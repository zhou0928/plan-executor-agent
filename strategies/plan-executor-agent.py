"""Plan-Executor Agent strategy: wires Planner -> Executor -> Replanner.

Error handling follows design doc §5, three categories:
1. invalid plan      -> Replanner regenerates (its own max_invalid_plan budget)
2. tool step failure -> failure recorded in scratchpad, Replanner takes over
3. systemic failure  -> streamed error naming the layer (planning/execution/replan exhausted)
When the replan budget is exhausted, fall back to a best-effort answer
explicitly labelled as such.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any

from dify_plugin.core.runtime import Session
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessageRole,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File, FileType
from dify_plugin.interfaces.agent import AgentStrategy, ToolEntity

from core.budget import BoundedCaller, BudgetTimeout
from core.execution_metadata import ExecutionMetadata
from core.executor import ExecutionOutcome, Executor
from core.params import to_int
from core.plan import InvalidPlanError, Plan
from core.planner import Planner
from core.replanner import BudgetExhaustedError, Replanner
from core.scratchpad import Scratchpad, validate_plan
from core.subagent_pool import LocalStepExecutor, StepExecutionError
from core.text import ThinkFilter, render_partial, strip_think, truncate_middle
from core.tool_allowlist import filter_allowed_tools

FINAL_ANSWER_SYSTEM = (
    "你是任务执行者。原计划未能完整执行。请基于以下已有成果，"
    "对用户目标给出尽力而为的回答，并明确说明哪些部分未能完成。"
)

# Used by the no-tools fast path (see _invoke).
FAST_PATH_SYSTEM = "你是任务执行者。直接完成用户目标并给出结果本身，不要描述你会怎么做。"

# Log label for progress messages (log messages never enter the node's text output).
PROGRESS_LABEL = "计划执行"

# Declared in the strategy yaml's output_schema, so it shows up as a node variable.
ANSWER_VAR = "answer"

# Per-step time budget. The yaml deliberately exposes no step_timeout param:
# every workflow would just set it wrong. ponytail: hard-coded 600s ceiling
# for a single tool/LLM step; lower it here if a workflow needs tighter.
_STEP_TIMEOUT_SECONDS = 600

# The daemon runs an invocation under PLUGIN_MAX_EXECUTION_TIMEOUT (600s) and
# kills the whole SSE stream past it, throwing away every step result. The
# default budget below stays under that so the plugin can answer with what it
# has; _RESERVE_FINAL_ANSWER_SECONDS is left for that closing model call (when
# less than that remains the accumulated results are returned as-is).
_DEFAULT_EXECUTION_SECONDS = 480
_RESERVE_FINAL_ANSWER_SECONDS = 60


def _flatten_content(content: Any) -> str:
    """SDK message content is either a plain string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "data", "") or "" for part in content)


def _coerce_history_message(raw: dict[str, Any]) -> Any:
    """Lift a history dict (role/content) into its SDK PromptMessage subclass.

    Mirrors AgentModelConfig.convert_prompt_messages; unknown roles fall back
    to UserPromptMessage so a bad turn can't crash the whole step.
    """
    role = raw.get("role")
    if role == PromptMessageRole.ASSISTANT.value:
        return AssistantPromptMessage(**raw)
    if role == PromptMessageRole.SYSTEM.value:
        return SystemPromptMessage(**raw)
    if role == PromptMessageRole.TOOL.value:
        return ToolPromptMessage(**raw)
    return UserPromptMessage(**raw)


def _chunk_text(chunk: Any) -> str:
    delta = getattr(chunk, "delta", None)
    if delta is None:
        return ""
    return _flatten_content(getattr(getattr(delta, "message", None), "content", None))


def _plan_with_validation(
    planner: Planner,
    goal: str,
    tools: list[ToolEntity],
    instruction: str,
    initial_vars: set[str],
    budget: int,
    deadline: float | None = None,
) -> Plan:
    """Plan with a retry budget for invalid plans.

    Without the loop a single bad plan would fall straight through to the
    category-3 "规划层失败" exit and return an empty answer; the loop converts
    the same budget into re-plan attempts. Model-level errors re-raise and are
    labelled by the caller (they are not plan-quality issues).
    Retries stop once ``deadline`` passes: another plan attempt on a slow model
    costs minutes the invocation does not have.
    """
    last_error: InvalidPlanError | None = None
    for attempt in range(budget + 1):
        if attempt and deadline is not None and time.monotonic() >= deadline:
            assert last_error is not None
            raise last_error
        try:
            plan = planner.plan(goal, tools, instruction, sorted(initial_vars))
            validate_plan(plan, initial_vars)
            return plan
        except InvalidPlanError as e:
            last_error = e
    assert last_error is not None  # budget >= 0 => loop ran at least once
    raise last_error


def _render_tool_message(message: Any) -> str:
    """Render one ToolInvokeMessage to text without repr()ing blobs.

    ``message`` is the SDK wrapper: the type sits on it, the content on
    ``.message`` (TextMessage.text / JsonMessage.json_object / BlobMessage.blob).
    Non-text payloads carry binary data, so they render as a terse label plus
    whatever the tool put in ``meta`` - repr()ing them would dump raw base64
    into the answer.
    """
    msg_type = getattr(message, "type", None)
    payload = getattr(message, "message", None)
    if msg_type == ToolInvokeMessage.MessageType.TEXT:
        return str(getattr(payload, "text", "") or "")
    if msg_type == ToolInvokeMessage.MessageType.JSON:
        data = getattr(payload, "json_object", None)
        return json.dumps(data, ensure_ascii=False) if data is not None else ""
    label = getattr(msg_type, "value", None) or "result"
    meta = getattr(message, "meta", None) or {}
    name = meta.get("filename") or meta.get("mime_type") or ""
    blob = getattr(payload, "blob", None)
    size = f" ({len(blob) / 1024:.0f} KB)" if isinstance(blob, bytes | bytearray) else ""
    return f"[{label}] {name}{size}".rstrip()


def _collect_tool_parts(response: Iterable[Any]) -> list[str]:
    """Render every tool message, dropping blank ones.

    A tool answering with an empty TEXT/None JSON counts as "no usable output":
    if nothing non-blank survives, the step fails and takes the replan path
    instead of silently propagating "" into dependent steps.
    """
    parts: list[str] = []
    for msg in response:
        if (part := _render_tool_message(msg)):
            parts.append(part)
    return parts


class _LLMCaller:
    """Callable ``(system, user) -> str`` plus ``.stream()`` for incremental text.

    Planner/Replanner/step code only call it, so they stay unaware of streaming.
    Token usage accumulates into ``usage`` so the node's usage panel can be
    filled once at the end.
    """

    def __init__(self, invoke: Any, stream: Any, usage: dict[str, Any]) -> None:
        self._invoke = invoke
        self._stream = stream
        self.usage = usage

    def __call__(self, system: str, user: str) -> str:
        # Planner/Replanner/reasoning-step calls all route through here, so a
        # transient model-api error (429 / network blip) gets one retry instead
        # of derailing the whole node. The stream path is deliberately not
        # retried: partial output has already been yielded to the user.
        # ponytail: 1 retry + 0.5s backoff; raise both if a flakier provider appears.
        last: Exception | None = None
        for attempt in range(_LLM_RETRY_ATTEMPTS):
            try:
                return self._invoke(system, user)
            except Exception as e:  # noqa: BLE001 - retried below, then re-raised
                last = e
                if attempt < _LLM_RETRY_ATTEMPTS - 1:
                    time.sleep(_LLM_BACKOFF_SECONDS)
        assert last is not None
        raise last

    def stream(self, system: str, user: str) -> Iterator[str]:
        return self._stream(system, user)


_LLM_RETRY_ATTEMPTS = 2
_LLM_BACKOFF_SECONDS = 0.5


class PlanExecutorAgentAgentStrategy(AgentStrategy):
    _HEARTBEAT_SECONDS = 5

    def __init__(self, runtime, session: Session) -> None:
        super().__init__(runtime, session)
        self._session = session

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        model_config = parameters["model"]
        mounted_tools = self._prepare_tools(parameters.get("tools") or [])
        tools = filter_allowed_tools(mounted_tools, parameters.get("allowed_tools"))
        if mounted_tools and not tools:
            yield self.create_log_message(
                PROGRESS_LABEL,
                {"warning": "allowed_tools 与已挂载工具名无一匹配，已按无工具模式执行"},
            )
        goal = str(parameters.get("query") or "")
        instruction = parameters.get("instruction") or ""
        max_steps = self._clamp(to_int(parameters.get("max_steps"), 20), 1, 50, 20)
        max_replan = self._clamp(to_int(parameters.get("max_replan"), 3), 0, 10, 3)
        max_invalid_plan = self._clamp(to_int(parameters.get("max_invalid_plan"), 2), 0, 10, 2)
        max_parallel = self._clamp(to_int(parameters.get("max_parallel_steps"), 1), 1, 8, 1)
        planning_prompt = parameters.get("planning_prompt") or None
        verbose = bool(parameters.get("verbose"))
        output_variable = (parameters.get("output_variable") or "output").strip() or "output"
        files = [f for f in (parameters.get("files") or []) if f]
        # `features: [history-messages]` makes Dify populate past turns here; the
        # strategy is responsible for putting them back into prompt_messages.
        history = list((model_config or {}).get("history_prompt_messages") or [])
        context_items = parameters.get("context") or []
        # 0 keeps full tool output; anything else caps what a single step may
        # paste into the next prompt (a scraped page would otherwise dominate it).
        result_chars = to_int(parameters.get("max_result_chars"), 8000)
        result_limit = None if result_chars <= 0 else self._clamp(result_chars, 500, 200_000, 8000)
        # 0 disables the budget (and with it the protection against the
        # daemon's PLUGIN_MAX_EXECUTION_TIMEOUT killing the whole invocation).
        budget_seconds = to_int(parameters.get("max_execution_seconds"), _DEFAULT_EXECUTION_SECONDS)
        deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None

        # Two funnels over one usage sink: planning and the final answer carry the
        # chat history, per-step calls do not (that would re-send the whole
        # conversation once per step).
        usage: dict[str, Any] = {"usage": None}
        usage_lock = threading.Lock()
        plan_llm = self._make_llm_caller(
            model_config,
            history=history,
            context=context_items,
            files=files,
            usage=usage,
            usage_lock=usage_lock,
        )
        step_llm = self._make_llm_caller(
            model_config,
            context=context_items,
            files=files,
            usage=usage,
            usage_lock=usage_lock,
        )
        # Planning and replanning are non-streaming single calls: a slow model
        # can spend minutes on one of them, which is exactly how an invocation
        # outlives the daemon's PLUGIN_MAX_EXECUTION_TIMEOUT. Bound them.
        planner_llm = BoundedCaller(plan_llm, deadline)
        scratchpad = Scratchpad(initial={"query": goal})
        initial_vars = {"query"}

        # No tools -> there is nothing to sequence, so planning only buys latency
        # (one LLM call per step) and produces meta-steps like "解析需求/分析信息".
        # ponytail: no tools = answer directly. Add any tool to re-enable planning.
        if not tools:
            answer = ""
            try:
                for piece in plan_llm.stream(instruction or FAST_PATH_SYSTEM, goal):
                    answer += piece
                    yield self.create_text_message(piece)
            except Exception as e:
                yield self.create_text_message(f"[计划执行 Agent] 模型调用失败：{e}\n")
                yield from self._emit_tail("", context_items, usage)
                return
            scratchpad.set(output_variable, answer)
            yield from self._emit_tail(answer, context_items, usage)
            return

        # --- Phase 1: planning (category 3 error exits here) ---
        planner = Planner(llm=planner_llm, planning_prompt=planning_prompt, max_steps=max_steps)
        try:
            plan = _plan_with_validation(
                planner, goal, tools, instruction, initial_vars, budget=max_invalid_plan, deadline=deadline
            )
        except InvalidPlanError as e:
            yield self.create_text_message(f"[计划执行 Agent] 规划层失败：{e}\n")
            yield from self._emit_tail("", context_items, usage)
            return
        except Exception as e:  # model unavailable etc.
            yield self.create_text_message(f"[计划执行 Agent] 模型调用失败（规划层）：{e}\n")
            yield from self._emit_tail("", context_items, usage)
            return

        if verbose:
            yield self.create_text_message(f"📋 初始计划：\n{self._render_plan(plan)}\n")

        # --- Phase 2+3: execute / replan loop ---
        replanner = Replanner(llm=planner_llm, max_replan=max_replan, max_invalid_plan=max_invalid_plan)
        replan_count = 0
        current_plan = plan
        start_index = 0

        streamed: dict[str, bool] = {}
        while True:
            outcome = yield from self._execute_with_progress(
                step_llm,
                tools,
                current_plan,
                scratchpad,
                start_index,
                verbose,
                max_parallel,
                streamed,
                max_result_chars=result_limit,
                deadline=deadline,
            )

            if outcome.timed_out:
                yield from self._final_answer(
                    plan_llm,
                    goal,
                    scratchpad,
                    output_variable,
                    context_items,
                    usage,
                    reason=f"执行时间预算（{budget_seconds} 秒）已用尽，剩余步骤已跳过",
                    deadline=deadline,
                )
                return

            if outcome.completed:
                break

            failed = outcome.failed_step
            assert failed is not None

            if deadline is not None and time.monotonic() >= deadline:
                # A replan is another model call; past the budget it only eats
                # into the daemon's hard timeout without ever being used.
                yield from self._final_answer(
                    plan_llm,
                    goal,
                    scratchpad,
                    output_variable,
                    context_items,
                    usage,
                    reason=f"执行时间预算（{budget_seconds} 秒）已用尽，剩余步骤已跳过",
                    deadline=deadline,
                )
                return

            # Category 2: step failure -> try replanning
            try:
                yield self.create_log_message(
                    PROGRESS_LABEL,
                    {"warning": f"步骤 {failed.id}「{failed.description}」失败：{outcome.error}，尝试重规划"},
                )
                new_plan = replanner.replan(
                    original=current_plan,
                    failed_step=failed,
                    error=outcome.error or "unknown",
                    scratchpad=scratchpad,
                    remaining_goal=goal,
                )
                replan_count += 1
                start_index = 0
                current_plan = new_plan
                if verbose:
                    yield self.create_text_message(f"📋 新计划（第 {replan_count} 次重规划）：\n{self._render_plan(new_plan)}\n")
            except InvalidPlanError as e:
                if replanner.invalid_budget_left <= 0:
                    yield from self._final_answer(
                        plan_llm,
                        goal,
                        scratchpad,
                        output_variable,
                        context_items,
                        usage,
                        reason=f"重规划生成的计划无效：{e}",
                    )
                    return
                yield self.create_log_message(
                    PROGRESS_LABEL, {"warning": f"重规划生成的计划无效（{e}），重试"}
                )
            except (BudgetExhaustedError, BudgetTimeout) as e:
                yield from self._final_answer(
                    plan_llm,
                    goal,
                    scratchpad,
                    output_variable,
                    context_items,
                    usage,
                    reason=str(e),
                    deadline=deadline,
                )
                return

        # Success: emit the final variable's content. The plan's output_var names
        # are chosen by the model, so fall back to the last step's variable.
        final_var = output_variable if scratchpad.has(output_variable) else current_plan.steps[-1].output_var
        final_text = strip_think(str(scratchpad.get(final_var, "")))
        if not streamed.get("answer"):
            yield self.create_text_message(final_text)
        yield from self._emit_tail(final_text, context_items, usage)

    # ------------------------------------------------------------------
    # helpers

    def _execute_with_progress(
        self,
        llm: _LLMCaller,
        tools: list[ToolEntity],
        plan: Plan,
        scratchpad: Scratchpad,
        start_index: int,
        verbose: bool = False,
        max_parallel: int = 1,
        streamed: dict[str, bool] | None = None,
        max_result_chars: int | None = None,
        deadline: float | None = None,
    ) -> Generator[AgentInvokeMessage, None, ExecutionOutcome]:
        """Run the executor on a worker thread while keeping the node stream alive.

        Progress goes out as LOG messages by default: text messages accumulate
        into the node's `text` output (Dify streams them raw, no separator), so
        they would glue onto the answer. With `verbose` on, steps go to the
        answer instead, one per line.
        Tokens from the streamed final step go out as text and set
        ``streamed["answer"]`` so the caller does not repeat the answer.
        Files a tool produced go out as node file outputs (see _artifact_message).
        The heartbeat is always emitted (a silent generator can idle the node).
        """
        streamed = streamed if streamed is not None else {}
        events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        outcome: list[ExecutionOutcome] = []
        failure: list[BaseException] = []
        answer_filter = ThinkFilter()

        def on_token(text: str) -> None:
            visible = answer_filter.feed(text)
            if not visible:
                return
            streamed["answer"] = True
            events.put(("token", visible))

        def on_artifact(message: AgentInvokeMessage) -> None:
            events.put(("artifact", message))

        pool = LocalStepExecutor(
            llm=llm,
            llm_stream=llm.stream,
            on_token=on_token,
            tool_invoker=lambda name, params: self._invoke_tool(
                name, params, tools, limit=max_result_chars, on_artifact=on_artifact
            ),
        )
        executor = Executor(
            pool,
            on_progress=lambda sid, done, total, desc: events.put(("step", f"步骤 {done}/{total}：{desc}")),
            max_parallel=max_parallel,
            step_timeout=_STEP_TIMEOUT_SECONDS,
            deadline=deadline,
        )

        def work() -> None:
            try:
                outcome.append(executor.run(plan, scratchpad, start_index=start_index))
            except BaseException as e:  # noqa: BLE001 - re-raised on the caller thread
                failure.append(e)
            finally:
                events.put(None)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while True:
            try:
                event = events.get(timeout=self._HEARTBEAT_SECONDS)
            except queue.Empty:
                yield self.create_log_message(PROGRESS_LABEL, {"status": "执行中"})
                continue
            if event is None:
                break
            kind, payload = event
            if kind == "token":
                yield self.create_text_message(payload)
            elif kind == "artifact":
                yield payload
            elif verbose:
                yield self.create_text_message(f"{payload}\n")
            else:
                yield self.create_log_message(PROGRESS_LABEL, {"progress": payload})
        worker.join()
        tail = answer_filter.flush()
        if tail:
            streamed["answer"] = True
            yield self.create_text_message(tail)
        if failure:
            raise failure[0]
        return outcome[0]

    def _make_llm_caller(
        self,
        model_config: dict,
        history: list | None = None,
        context: Any = None,
        files: list | None = None,
        usage: dict[str, Any] | None = None,
        usage_lock: threading.Lock | None = None,
    ) -> _LLMCaller:
        """Build one funnel every LLM call of a phase goes through.

        Chat history, knowledge-retrieval context and current-turn images are
        injected here so no caller has to know about them. History items arrive
        as plain dicts (role/content); coerce them into SDK message objects
        here, mirroring AgentModelConfig.convert_prompt_messages.

        Callers that should carry history pass it; the step caller passes none,
        because a long chat would otherwise be re-sent once per step. Both share
        one ``usage`` dict (and its lock), so the node's usage panel still sees
        every call.
        """
        history = [_coerce_history_message(m) for m in (history or [])]
        context_block = self._context_block(context)
        attachments = self._attachment_parts(files or [])
        usage = usage if usage is not None else {"usage": None}
        lock = usage_lock if usage_lock is not None else threading.Lock()

        def build(system: str, user: str) -> list[Any]:
            if context_block:
                system = f"{system}\n\n参考资料：\n{context_block}"
            prompt: Any = user
            if attachments:
                prompt = [*attachments, TextPromptMessageContent(data=user)]
            return [SystemPromptMessage(content=system), *history, UserPromptMessage(content=prompt)]

        def record(chunk_usage: Any) -> None:
            if chunk_usage is not None:
                with lock:  # steps may run concurrently and share this dict
                    self.increase_usage(usage, chunk_usage)

        def invoke(system: str, user: str) -> str:
            messages = build(system, user)
            result = self._session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=messages,
                stream=False,
            )
            text = _flatten_content(result.message.content)
            if text.strip():
                record(result.usage)
                return text
            # Some providers only deliver content through stream chunks.
            parts: list[str] = []
            for chunk in self._session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=messages,
                stream=True,
            ):
                record(getattr(getattr(chunk, "delta", None), "usage", None))
                parts.append(_chunk_text(chunk))
            return "".join(parts)

        def stream(system: str, user: str) -> Iterator[str]:
            for chunk in self._session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=build(system, user),
                stream=True,
            ):
                record(getattr(getattr(chunk, "delta", None), "usage", None))
                piece = _chunk_text(chunk)
                if piece:
                    yield piece

        return _LLMCaller(invoke, stream, usage)

    @staticmethod
    def _attachment_parts(files: list) -> list[Any]:
        """Image files from the current turn, as prompt content parts.

        Non-image files are skipped on purpose (the plan pipeline is text-only).
        A file that cannot be read raises: silent skips hide a real
        misconfiguration behind a plausible-looking answer.
        """
        parts: list[Any] = []
        for item in files:
            file = item if isinstance(item, File) else File(**item)
            if file.type != FileType.IMAGE:
                continue
            fmt = (file.extension or "").lstrip(".").lower()
            if not fmt and file.mime_type and "/" in file.mime_type:
                fmt = file.mime_type.split("/", maxsplit=1)[1]
            if fmt == "jpg":
                fmt = "jpeg"
            fmt = fmt or "png"
            parts.append(
                ImagePromptMessageContent(
                    format=fmt,
                    base64_data=base64.b64encode(file.blob).decode("ascii"),
                    mime_type=file.mime_type or f"image/{fmt}",
                    filename=file.filename or "",
                    detail=ImagePromptMessageContent.DETAIL.LOW,
                )
            )
        return parts

    def _emit_tail(
        self,
        answer: str,
        context: Any,
        usage: dict[str, Any],
    ) -> Generator[AgentInvokeMessage]:
        """Everything Dify expects after the answer: the declared node variable,
        the retrieval citations, and the usage the node's panel reads."""
        yield self.create_variable_message(ANSWER_VAR, answer)
        yield from self._emit_retriever_resources(context)
        yield self.create_json_message(
            {"execution_metadata": ExecutionMetadata.from_llm_usage(usage.get("usage")).to_dict()}
        )

    @staticmethod
    def _context_block(context: Any) -> str:
        """Render knowledge-retrieval items into a prompt-ready text block."""
        if not isinstance(context, list):
            return ""
        parts: list[str] = []
        for item in context:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            title = item.get("title") or ""
            parts.append(f"【{title}】\n{item['content']}" if title else str(item["content"]))
        return "\n\n".join(parts)

    def _emit_retriever_resources(self, context: Any) -> Generator[AgentInvokeMessage]:
        """Surface the retrieved segments so Dify renders the citation footer."""
        if not isinstance(context, list) or not context:
            return
        resources = []
        for item in context:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            resources.append(
                ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource(
                    content=str(item.get("content", "")),
                    position=meta.get("position"),
                    dataset_id=meta.get("dataset_id"),
                    dataset_name=meta.get("dataset_name"),
                    document_id=meta.get("document_id"),
                    document_name=meta.get("document_name"),
                    data_source_type=meta.get("document_data_source_type"),
                    segment_id=meta.get("segment_id"),
                    retriever_from=meta.get("retriever_from"),
                    score=meta.get("score"),
                    hit_count=meta.get("segment_hit_count"),
                    word_count=meta.get("segment_word_count"),
                    segment_position=meta.get("segment_position"),
                    index_node_hash=meta.get("segment_index_node_hash"),
                    page=meta.get("page"),
                    doc_metadata=meta.get("doc_metadata"),
                )
            )
        if resources:
            yield self.create_retriever_resource_message(
                retriever_resources=resources,
                context="",
            )

    def _invoke_tool(
        self,
        name: str,
        params: dict[str, Any],
        tools: list[ToolEntity],
        limit: int | None = None,
        on_artifact: Callable[[AgentInvokeMessage], None] | None = None,
    ) -> str:
        tool = next((t for t in tools if t.identity.name == name), None)
        if tool is None:
            raise StepExecutionError(f"计划引用了不可用工具：{name}")
        # The plan's params ride on top of the tool's configured runtime values
        # (webscraper user_agent, mind_map theme, ...). They must be merged in
        # here: only `parameters` reaches the provider, and Dify ships those
        # settings alongside the tool exactly because the plugin has to forward
        # them. credential_id likewise, for tools that carry a credential.
        parameters = {**(tool.runtime_parameters or {}), **params}
        try:
            response = self._session.tool.invoke(
                provider_type=tool.provider_type,
                provider=tool.identity.provider or "",
                tool_name=name,
                parameters=parameters,
                credential_id=tool.credential_id,
            )
        except Exception as e:  # noqa: BLE001 - normalised so the step fails via the replan path, like empty results
            raise StepExecutionError(f"工具 {name} 调用失败：{e}") from e
        messages = list(response)  # rendered once for text and once for artifacts
        if on_artifact is not None:
            for message in messages:
                if (artifact := self._artifact_message(message)) is not None:
                    on_artifact(artifact)
        parts = _collect_tool_parts(messages)
        if not parts:
            raise StepExecutionError(f"工具 {name} 返回空结果")
        return truncate_middle("\n".join(parts), limit)

    def _artifact_message(self, message: Any) -> AgentInvokeMessage | None:
        """Turn a tool's binary/link output into a node file output.

        Text results already reach the plan; without this a mind-map PNG or a
        converted docx would exist only as a "[blob] x.png" note in the answer.
        Types are matched by value, not by enum member: a newer API can send a
        type this SDK does not know (e.g. binary_link), and an unknown type must
        stay a no-op rather than crash the step.
        """
        msg_type = getattr(getattr(message, "type", None), "value", None)
        payload = getattr(message, "message", None)
        meta = getattr(message, "meta", None) or {}
        blob = getattr(payload, "blob", None)
        if msg_type == "blob" and isinstance(blob, bytes | bytearray):
            return self.create_blob_message(blob=bytes(blob), meta=meta)
        url = getattr(payload, "text", None)
        if not url:
            return None
        if msg_type in ("image", "image_link"):
            return self.create_image_message(str(url))
        if msg_type in ("link", "binary_link"):
            return self.create_link_message(str(url))
        return None

    def _prepare_tools(self, tool_entities) -> list[ToolEntity]:
        """Normalise the incoming tool config into SDK ToolEntity objects.

        Dify sends tools as plain dicts (api side: ``tool_runtime.entity.
        model_dump`` plus ``runtime_parameters``/``credential_id``); the SDK's
        prompt-tool conversion expects ToolEntity and silently drops dicts. The
        entities are kept as-is further down instead of being converted to
        PromptMessageTool: invoking a tool needs ``identity.provider`` and
        ``provider_type``, which the prompt-message view does not carry.
        """
        return [
            t if isinstance(t, ToolEntity) else ToolEntity.model_validate(t)
            for t in (tool_entities or [])
        ]

    @staticmethod
    def _clamp(value: int, lo: int, hi: int, default: int) -> int:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    @staticmethod
    def _render_plan(plan: Plan) -> str:
        lines = []
        for s in plan.steps:
            tool = s.tool or "（纯推理）"
            lines.append(f"  步骤{s.id}: {s.description} [工具: {tool}] → {s.output_var}")
        return "\n".join(lines)

    def _final_answer(
        self,
        llm: _LLMCaller,
        goal: str,
        scratchpad: Scratchpad,
        output_variable: str,
        context_items: Any,
        usage: dict[str, Any],
        reason: str,
        deadline: float | None = None,
    ) -> Generator[AgentInvokeMessage]:
        """Best-effort answer with an explicit failure notice (design doc §5.1).

        When ``deadline`` leaves less than the reserve, the closing model call is
        skipped and the accumulated results are returned verbatim: something
        beats an invocation the daemon kills mid-flight.
        """
        state = scratchpad.variables()
        results = {k: v for k, v in state.items() if not k.endswith("_error")}
        errors = {k: v for k, v in state.items() if k.endswith("_error")}
        payload = json.dumps({"成果": results, "失败": errors}, ensure_ascii=False, default=str)
        yield self.create_text_message(f"❌ {reason}，以下为尽力回答：\n\n")
        if deadline is not None and deadline - time.monotonic() < _RESERVE_FINAL_ANSWER_SECONDS:
            digest = render_partial(results, skip={"query"})
            yield self.create_text_message(digest)
            scratchpad.set(output_variable, digest)
            yield from self._emit_tail(digest, context_items, usage)
            return
        answer = ""
        answer_filter = ThinkFilter()
        for piece in llm.stream(
            FINAL_ANSWER_SYSTEM,
            f"目标：{goal}\n失败原因：{reason}\n已有成果：{payload}",
        ):
            visible = answer_filter.feed(piece)
            if visible:
                answer += visible
                yield self.create_text_message(visible)
        answer += answer_filter.flush()
        answer = strip_think(answer)
        scratchpad.set(output_variable, answer)
        yield from self._emit_tail(answer, context_items, usage)
