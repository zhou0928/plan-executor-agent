## plan-executor-agent

**Author:** xiaozhou
**Version:** 0.2.0
**Type:** agent-strategy

### Description

A **Plan-and-Execute** agent strategy for Dify. The model first produces a
structured step-by-step plan, then executes each step with tool awareness.
When a step fails or the result diverges from the plan, the strategy
automatically replans. Supports streaming, parallel step execution, file
inputs, and execution metadata reporting.

### Features

- **Plan first, execute after**: the model generates a structured plan before
  calling any tool
- **Tool awareness**: the executor only uses the tools available in the node
  and matching the `allowed_tools` allowlist
- **Replanning**: on plan failure or tool error, the strategy replans up to
  `max_replan` times (invalid plans draw from a separate budget)
- **Streaming**: intermediate plans, step results, and the final answer are
  streamed back to the workflow
- **Parallel execution**: set `max_parallel_steps` to run independent steps in
  parallel (default: 1, sequential)
- **File inputs**: pass image files through the `files` parameter; they are
  attached to every LLM call
- **Execution metadata**: token usage is accumulated and reported to the Dify
  usage panel

### Setup

1. Install the plugin in your Dify workspace (Agent strategies section).
2. Add the Chatflow / Workflow **Agent** node and choose **Plan-Executor Agent**.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | LLM provider and model used for planning and execution | — |
| `query` | User query to answer | — |
| `tools` | Tools available to the executor | — |
| `allowed_tools` | Optional comma-separated allowlist of tool names | all |
| `context` | Extra context appended to the user message | — |
| `files` | Optional image files attached to every LLM call | — |
| `instruction` | Optional extra instructions appended to the planning prompt | — |
| `max_steps` | Maximum number of steps per plan | 5 |
| `max_replan` | Maximum replans on valid-but-failed plans | 2 |
| `max_invalid_plan` | Maximum replans on invalid plans (separate budget) | 2 |
| `max_parallel_steps` | Parallelism for independent steps (1 = sequential) | 1 |
| `planning_prompt` | Optional custom planning prompt override | default |
| `output_variable` | Name of the workflow variable holding the answer | `answer` |
| `verbose` | Stream intermediate plans and step details | false |

### Usage Example

A query like "Compare quarterly revenue trends across regions" produces a
plan such as:

1. Gather revenue data per region
2. Aggregate by quarter
3. Compare trends and summarize

Each step runs with the available tools; failures trigger replanning instead
of stopping. The final answer is emitted into `output_variable` (default
`answer`).

### Privacy

See [PRIVACY.md](./PRIVACY.md). This plugin does not collect personal data and
only forwards your query, context, tools, and files to the LLM model selected
in the node.
