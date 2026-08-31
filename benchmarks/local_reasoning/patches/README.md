# Ollama runtime surgery patches

`ollama-v0.32.13-fixed-draft-depth.patch` applies to Ollama v0.32.13 at commit
`0f25c31bd53b64dc3fcc8fce0bde954159a67a58`. It adds an experimental
`OLLAMA_MLX_FIXED_DRAFT_DEPTH` environment variable to the MLX speculative
decoder. An unset variable leaves the production adaptive controller unchanged;
an integer at or above zero pins every speculative round to that draft depth.
The patch applies cleanly to that exact commit, and its focused
`TestDepthController` suite passes there.

This is a measurement instrument, not a proposed production default. It makes
depth 0 a no-speculation control without changing `think: true`, and lets the
reasoning benchmark measure the actual Metal cost/acceptance curve at depths
1, 2, 3, and so on. The controller still records its normal acceptance and cost
diagnostics.

Apply and run its focused tests from an Ollama v0.32.13 checkout:

```sh
git apply /absolute/path/to/ollama-v0.32.13-fixed-draft-depth.patch
gofmt -w x/mlxrunner/speculate_depth.go x/mlxrunner/speculate_depth_test.go
go test ./x/mlxrunner -run 'TestDepthController' -count=1
```

When starting a patched server, keep it on a separate local port and point it
at the existing read-only model store. For example:

```sh
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/path/to/.ollama/models \
OLLAMA_MLX_FIXED_DRAFT_DEPTH=3 \
patched-ollama serve
```
