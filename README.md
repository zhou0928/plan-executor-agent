# Plan-Executor Agent

A **Plan-and-Execute** agent strategy for [Dify](https://dify.ai): the model writes a structured
step-by-step plan first, then executes that plan step by step with tool awareness, and replans
automatically when a step fails or its result drifts away from the plan.

- **Source repository:** <https://github.com/zhou0928/plan-executor-agent>
- **Author:** zhou0928
- **Plugin type:** agent strategy

## Features

| Feature | Detail |
|---|---|
| Plan first | A structured plan is produced before any tool is called |
| Tool awareness | The planner only sees the tools mounted on the node, narrowed by `allowed_tools` |
| Automatic replanning | A failed step or a tool error triggers a replan (`max_replan` times); malformed plans draw on a separate budget |
| Normalised failures | Tool exceptions and reasoning-LLM errors become step errors and take the replan path; a planning-layer failure exits gracefully with a notice |
| Streaming output | The final answer streams token by token; `verbose` also streams intermediate plans and per-step details |
| Parallel steps | `max_parallel_steps > 1` runs steps that have no dependency between them concurrently |
| Image input | Images passed through `files` are attached to every model call |
| Execution metadata | Token usage is accumulated and reported to the node's usage panel |
| File artefacts | Binary and link results from tools become node file outputs (blob → file, image_link → image, link → link) |

## Installation

1. In your Dify workspace open **Plugins → Install plugin → Via local file** and upload the
   `plan-executor-agent-<version>.difypkg` package, or install it from the Marketplace.
2. Add an **Agent** node to a Chatflow or Workflow and select **Plan-Executor Agent** as the
   agent strategy.
3. Choose the model that should plan and execute, and attach the tools the agent may call.

Nothing else needs to be configured.

## Credentials and connection requirements

The plugin stores **no credentials** and talks to no service directly. Every model request and
tool invocation is proxied back to Dify through the plugin SDK, so it inherits the provider
credentials, base URLs and network access already configured in your workspace. The only
connections it makes are the ones the selected model provider and the selected tools require.

## Usage

Bind the node's **Query** to the user input, attach some tools, and run it:

```
Query:        对比各地区季度营收趋势
Instruction:  Answer in Chinese. Conclusion first, then the evidence.
Tools:        any tools you want the agent to be able to call
```

That produces a plan such as:

```
1. Fetch per-region revenue data
2. Aggregate by quarter
3. Compare the trends and summarise
```

Each step is executed in order, and a failure triggers a replan instead of aborting the run. The
final answer is written to the node's `answer` variable and to the scratchpad variable named by
`output_variable` (`output` by default).

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `model` | LLM provider and model used for planning and execution | — |
| `query` | The user goal: input for both planning and execution | — |
| `tools` | Tools the node exposes to the executor | — |
| `allowed_tools` | Optional allowlist. Pass a JSON array of exact tool names, e.g. `["web_scraper"]`; a single bare name also works. Empty = every attached tool. A comma-separated string is **not** parsed and matches no tool. | all |
| `context` | Knowledge-retrieval results. Injected into every model call and rendered as citations. | — |
| `files` | Files for the current turn. Images are attached to every model call. | — |
| `instruction` | Extra requirements for the agent, e.g. answer style or constraints | — |
| `max_steps` | Maximum number of steps in one plan (1-50) | 20 |
| `max_replan` | How many times a failed step may trigger a replan (0-10). When exhausted the node answers best-effort. | 3 |
| `max_invalid_plan` | Separate budget for replans whose output could not be parsed (0-10). Does not consume `max_replan`. | 2 |
| `max_parallel_steps` | Concurrency for steps with no dependency between them (1-8). 1 = sequential. | 1 |
| `planning_prompt` | Fully replaces the built-in Chinese planning prompt | built-in |
| `output_variable` | Scratchpad variable the final answer is written to | `output` |
| `max_result_chars` | Tool or retrieval output longer than this is clipped, keeping both ends. 0 = no limit. | 8000 |
| `max_execution_seconds` | Wall-clock budget: no new step is started after it and the node answers with what it has. 0 = no limit. | 480 |
| `verbose` | Stream intermediate plans and per-step details into the reply | off |

## Known limitations and tuning notes

### Step output length

Tool and retrieval results longer than `max_result_chars` are clipped, keeping the head and the
tail and marking how many characters were dropped in the middle. Set it to `0` to disable. This
is the main knob for controlling token cost.

### The execution budget is not a generation limit

The plugin daemon kills an invocation that outruns `PLUGIN_MAX_EXECUTION_TIMEOUT` (600 s by
default) and every intermediate result is lost. `max_execution_seconds` defaults to 480 s so
the node answers with what it already produced instead of being killed. It does **not** cap how
much a single call generates: a slow local model can emit tens of thousands of tokens in one
reply, and a call already in flight cannot be interrupted. Tune the model's own generation
limit (`num_predict` for Ollama) or raise the daemon timeout as well.

### History only reaches the planning phase

Multi-turn history is sent with the planning and final-answer calls, not with per-step calls, so
a long conversation does not multiply token usage by the number of steps.

### `verbose` and a clean reply are mutually exclusive

Observed on Dify 1.17.1: a "direct reply" node that references the agent node's `answer`
variable does not stream (the reply comes out empty) and has to use `.text` instead, and `.text`
contains the plan and step text whenever `verbose` is on. Keep `verbose=false` for a clean
reply and follow progress in the node's log panel.

### `binary_link` compatibility

Since Dify 1.17 the API can send `binary_link` tool messages, a type the `dify_plugin` 0.9/0.10
enum does not know. The plugin itself matches message types by string value and treats unknown
types as a no-op, but the SDK fails while parsing such a message (for example when the built-in
web scraper emits one), so the step still fails. This needs an SDK bump or a version alignment.

### Model failures

A planning-layer failure prints a notice and writes an empty answer:

```
[计划执行 Agent] 模型调用失败（规划层）：<upstream error>
```

Upstream quota and authentication errors are surfaced verbatim.

## Privacy

See [PRIVACY.md](./PRIVACY.md). The plugin collects no personal data; it only forwards your
query, context, tools and files to the model and the tools you configured on the node.

## Development

```bash
git clone https://github.com/zhou0928/plan-executor-agent
cd plan-executor-agent
pip install -r requirements.txt
python -m pytest
```

Core tests run without the SDK installed; the strategy-level tests skip when `dify_plugin` is
missing.
