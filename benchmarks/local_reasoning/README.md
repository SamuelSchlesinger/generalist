# Local reasoning inference benchmark

This benchmark optimizes performance *for reasoning*. Reasoning is not an
optional speed feature here: every native Ollama request sends `think: true`,
and an attempt passes only when all three quality guards hold:

- the model emitted a non-empty thinking stream;
- generation completed without hitting the output limit; and
- the checked final answer is correct.

The prompt asks for only the final marker after thinking. Compliance is reported
as a separate compact-answer count and as post-answer tail latency; extra prose
does not turn a correct reasoning result into a quality failure.

`--answer-format json-schema` keeps `think: true` and asks Ollama to constrain the
response channel to a single `{"final":"<answer>"}` object. The benchmark parses
a correct trailing object but counts it as compact only when no prose surrounds
it, so current structured-output failures remain visible. This tests whether
structured decoding can remove duplicated answer prose after the model has
already done the work in its thinking stream. It is a reasoning-preserving
latency optimization, not a no-thinking mode.

The benchmark reports quality and completion alongside time to first thinking,
time to final answer, prompt prefill throughput, decode throughput, output-token
count, post-answer tail latency, load time, and resident model size reported by
Ollama. Ollama's `eval_count` covers the complete generated stream, so
`decode_tokens_per_second` includes reasoning and answer tokens on ordinary
single-pass requests; it is not a count of visible answer tokens. When the
runtime log exposes multiple inference passes, the benchmark also sums them and
reports `runtime_estimated_emitted_tokens` as iterations plus accepted drafts.
This makes two-pass reasoning/format routes visible even when Ollama's API
counters describe only the final pass.
On macOS, the runner also tails `~/.ollama/logs/server.log` around each request
and records Ollama's direct MTP drafted/accepted counts, acceptance ratio, and
adaptive depth history. Use `--ollama-log` if the log lives elsewhere. Avoid
concurrent Ollama traffic during a benchmark so a log record cannot be
misattributed.

## Static gate

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/local_reasoning -p 'test_*.py'
```

## Baseline runs

The authentic model-default profile omits sampling overrides and repeats the
small exact-answer suite:

```sh
python3 benchmarks/local_reasoning/run.py \
  --model qwen3.8:27b-mlx \
  --variant official-nvfp4-default \
  --tag smoke \
  --repeat 3 \
  --require-pass
```

For a lower-noise paired surgery comparison, pin the same sampling controls on
every variant. This still performs full reasoning:

```sh
python3 benchmarks/local_reasoning/run.py \
  --model qwen3.8:27b-mlx \
  --variant official-nvfp4-controlled \
  --tag smoke \
  --repeat 3 \
  --temperature 0.6 \
  --seed 1 \
  --reload-between \
  --cooldown-seconds 30 \
  --require-pass
```

Measure a reasoning-preserving no-MTP control by adding
`--mtp-mode off-logprobs`. Ollama's native logprobs request parks speculative
decoding while `think: true` remains fixed. Returning selected-token logprobs
adds a small control-path overhead, so use this to estimate MTP's benefit, not
as a perfectly overhead-free plain-decoding score.

Use `--padding-lines` to sweep prefill independently. `--reload-between`
resets the model process and measures load cost separately from prompt/decode
cost; combine it with `--cooldown-seconds` so sequential variants do not merely
measure a progressively hotter machine. Interleave variant order across
repetitions as well. `--inspect-only` records Ollama/model metadata without
generating. There is intentionally no no-reasoning option.

Measure the compact structured-answer path with the same quality gates:

```sh
python3 benchmarks/local_reasoning/run.py \
  --model qwen3.8:27b-mlx \
  --variant official-nvfp4-json-final \
  --answer-format json-schema \
  --temperature 0 \
  --seed 1 \
  --require-pass
```

Each attempt stores the complete thinking and answer streams, the exact request
profile, native Ollama counters, and a strict classification. JSONL is flushed
after each record; interrupted runs remain inspectable. Start a new result file
for each model artifact or changed benchmark contract.

Summarize variants side by side:

```sh
python3 benchmarks/local_reasoning/summarize.py \
  benchmarks/local_reasoning/results/*.jsonl
```

Inspect the exact on-disk artifact composition before changing weights or
removing a component:

```sh
python3 benchmarks/local_reasoning/inspect_manifest.py \
  ~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx
```

This reports referenced bytes and blob counts separately for the language
model, MTP head, vision tower, and non-tensor metadata. It complements the
logical tensor types returned by `/api/show`; the two representations need not
have the same count.

## Model-surgery ladder

Change one axis at a time, keep task order and sampling controls fixed, and
reject a variant unless its reasoning, completion, and exact-answer counts are
at least as good as the baseline.

1. Official dense 27B NVFP4 MLX artifact with direct MTP acceptance evidence.
2. Instrumented runtime sweep with fixed draft depths, including depth zero,
   before changing weights.
3. MTP precision variants only if acceptance data supports them; do not assume
   that restoring `mtp.fc` or the full MTP head to BF16 is beneficial.
4. Text-only artifact with the unused vision tower removed, for load-time and
   memory measurement rather than expected decode gains.
5. Kernel/runtime changes, measured separately for prefill and decode.

The model snapshot records language/vision/MTP tensor type counts, the MTP
projection's precision, hybrid attention-layer indices, model digest, and
Ollama version. That makes a result attributable to an actual artifact instead
of a mutable model tag. Direct MTP acceptance is not currently exposed by the
Ollama API, so the benchmark obtains it from the runtime log and keeps the API
decode rate as a separate metric.
