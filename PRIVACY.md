## Privacy Policy

**Last updated:** 2026-09-14

This privacy policy applies to the `plan-executor-agent` plugin.

### Data Collection

This plugin does **not** collect any personal data. It processes only the data
you explicitly provide inside your Dify workflow:

- The `query` (user question) and optional `context` you pass to the Agent node
- The `tools` selected in the node (tool names and their invocation parameters
  at runtime)
- Optional `files` (images) you attach through the `files` parameter
- The `model` you select in the node — no API keys are stored by the plugin;
  model credentials remain managed by your Dify workspace

### Data Usage

The data above is used solely to fulfill your request: it is sent to the LLM
model you configured in the node (e.g. OpenAI, Anthropic) and, when executing
tool steps, to the tools you selected. This plugin does not store, log, or
share the data with any third party beyond the model provider and tools you
explicitly configure in Dify.

### Data Retention

The plugin keeps no persistent data. All inputs and outputs exist only for the
duration of a single workflow run in memory and are discarded when the run
ends. To delete any related data, delete the corresponding workflow runs in
your Dify workspace.

### Contact

For privacy-related questions, contact the plugin author through the contact
channel of your Dify workspace community.
