# unreal-agent-runner

Run an AI agent from a prompt or JSON request. It writes events to stdout as
JSONL and exits when the task finishes.

Install with Go 1.27+:

```sh
go install github.com/unreallabsai/unreal-agent/cmd/unreal-agent-runner@latest
```

Set an OpenAI API key and run a prompt in the current directory:

```sh
export OPENAI_API_KEY="..."
unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

Or run from source at the repository root:

```sh
go run ./cmd/unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

Choose a workspace and save the output:

```sh
unreal-agent-runner -workspace ./my-project -p 'Summarize this project.' > run.jsonl
```

You can also pass a JSON request as an argument or through stdin:

```sh
unreal-agent-runner '{"prompt":"Summarize this project."}'
unreal-agent-runner < request.json
```

OpenAI is the default provider. Set `UNREAL_HARNESS_LLM_PROVIDER` to `openai`,
`openai-codex`, `anthropic`, `anthropic-subscription`, `openrouter`, `fireworks`,
or `ollama`, and
`UNREAL_HARNESS_LLM_MODEL` to choose a model.

Use Anthropic with an API key, an `ant auth login` profile, or workload identity:

```sh
export UNREAL_HARNESS_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="..." # omit when using a profile or workload identity
unreal-agent-runner -p 'Inspect this project.'
```

To use a Claude Code subscription token created by `claude setup-token`, select
the subscription provider explicitly so it cannot silently fall back to API
billing:

```sh
export UNREAL_HARNESS_LLM_PROVIDER=anthropic-subscription
export CLAUDE_CODE_OAUTH_TOKEN="..."
unreal-agent-runner -p 'Inspect this project.'
```

Anthropic requests default to `claude-opus-5` and 32,000 output tokens. Override
those with `UNREAL_HARNESS_LLM_MODEL` and the JSON request field
`max_output_tokens`. LLM credentials are removed from the environment inherited
by Bash tool processes.

Run `unreal-agent-runner -h` for options and the JSON request fields.

## Docker

The `unrea1labs/unreal-agent` image supports Linux on AMD64 and ARM64. Run it
with a project mounted as the workspace:

```sh
docker run --rm -i --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD:/workspace" \
  unrea1labs/unreal-agent:latest -p 'Summarize this project.'
```

Each release also publishes its Git tag (for example, `v0.1.0`) for version pinning.
