# Agent transport benchmark

This corpus measures a narrow but important part of agent reliability: can a
model produce one syntactically valid Python script whose bridge calls preserve
the requested runtime values?

Generalist currently exposes code mode as an OpenAI-style function with a JSON
object argument, `{ "code": "..." }`. That forces the model's tool-call channel
to represent Python inside a JSON string. The benchmark compares that boundary
with two controls:

- `json_tool`: the current JSON object wrapper on `/chat/completions`;
- `plain_text`: raw Python in assistant text, which isolates code generation
  from tool-call serialization but is not a suitable production capability
  boundary; and
- `responses_custom`: an OpenAI Responses custom tool whose input is freeform
  text. Provider support is probed, never assumed.

The checker parses generated source with Python's AST and compares bridge call
names, order, and runtime literal values with the corpus. It deliberately does
**not execute provider-generated code**. As a result, the benchmark establishes
transport and static script correctness, not tool semantics, sandbox safety, or
successful end-to-end execution.

The corpus declares whether Generalist preloads its generated bridge. Current
runs accept either direct `tools.<name>(...)` use or an explicit `import tools`,
and advertise compact Python-like call signatures; full JSON Schemas remain a
runtime `__doc__` fallback.

## Corpus

[`tasks.json`](tasks.json) includes a simple control plus quote, backslash,
Unicode, nested data, argv, unified-diff, delimiter, batching, dataflow, and
retry cases. Fixed payloads must remain visible as Python literals; `$any`
marks a value that should be computed from an earlier tool result.

Every attempt records the sanitized request, full provider response, extracted
source, checker result, latency, token usage, and model-argument byte overhead.
No authorization header or API key is written. JSONL records are flushed and
synced after every attempt, and an interrupted file remains summarizable.
Treat result files as append-only evidence: start a new file for a changed
corpus, prompt, provider route, or implementation rather than rewriting an old
run.

Chat-completions runs default to `temperature=0` and `seed=1`; Responses runs
use the same temperature but omit the unsupported seed field. The exact values
are recorded in both the run manifest and request. Use `--no-seed` only when a
provider rejects that optional control; a seed is advisory and some upstreams
can remain nondeterministic.

## Reproducible first pass

Run the static tests first:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/agent_transport -p 'test_*.py'
```

The installed local comparison model is currently `qwen3.6:35b-a3b`:

```sh
python3 benchmarks/agent_transport/run.py \
  --provider ollama \
  --model qwen3.6:35b-a3b \
  --tag smoke \
  --transport json_tool \
  --transport plain_text \
  --reasoning-effort none
```

OpenRouter credentials are read only from `OPENROUTER_API_KEY`. A bounded
smoke comparison for the two remote models is:

```sh
python3 benchmarks/agent_transport/run.py \
  --provider openrouter \
  --model moonshotai/kimi-k3 \
  --tag smoke \
  --transport json_tool \
  --transport plain_text \
  --reasoning-effort low

python3 benchmarks/agent_transport/run.py \
  --provider openrouter \
  --model qwen/qwen3.8-max \
  --tag smoke \
  --transport json_tool \
  --transport plain_text \
  --reasoning-effort low
```

Probe custom/freeform support with only the ASCII control before spending on a
full run:

```sh
python3 benchmarks/agent_transport/run.py \
  --provider openrouter \
  --model moonshotai/kimi-k3 \
  --task simple_ascii \
  --transport responses_custom \
  --reasoning-effort low
```

An HTTP or extraction failure is a benchmark result and does not stop the
remaining matrix. Use `--require-pass` only when a green result should act as a
CI gate. Once a smoke route works, omit `--tag smoke` for the full corpus and
use `--repeat 3` only for the finalists; this keeps the exploratory paid run
bounded.

Summarize one or more result files without modifying them:

```sh
python3 benchmarks/agent_transport/summarize.py \
  benchmarks/agent_transport/results/*.jsonl
```

The summary reports exact-call pass rate, failure taxonomy, median latency,
mean token counts, median model-argument overhead, and provider-reported cost
when available. Raw JSONL remains the authoritative evidence.

When the checker becomes stricter, derive a new result file instead of editing
old evidence:

```sh
python3 benchmarks/agent_transport/recheck.py \
  benchmarks/agent_transport/results/OLDER-RUN.jsonl
```

The derived attempts retain the original request, response, metrics, run ID,
and classification under `derived_from`, while recording hashes for every
source file and checker artifact.
