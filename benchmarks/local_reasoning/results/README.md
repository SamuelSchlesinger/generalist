# Results

Keep benchmark JSONL files append-only. A useful comparison must preserve
reasoning, complete the generation, and pass the exact-answer checks; a faster
failed or truncated run is not an optimization win.

Record model-surgery results here only after the runner's model snapshot has
captured the concrete model digest, tensor precision mix, and Ollama version.

## Dense Qwen3.8 Metal baseline - 2026-08-31

All results in this section use `qwen3.8:27b-mlx` at digest
`5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e`
on an Apple M4 Pro with 48 GB unified memory. Every scored request sent native
Ollama `think: true`; a run counted only if thinking was non-empty, generation
completed, and the final answer passed its exact check. The installed runtime
was Ollama 0.32.13 unless a row explicitly says `custom`.

The model-default three-task run passed 3/3, with median wall time 34.862 s,
prompt prefill 93.922 tokens/s, decode 34.289 tokens/s, and 1,200 generated
tokens at the median. This is the production-oriented baseline in
`20260831T043644Z-qwen3.8-27b-mlx-official-nvfp4-default.jsonl`.

### Artifact structure

The API inspection and local manifest census describe a dense 27.781B-parameter
hybrid-attention model, not a mixture-of-experts model:

| Component | Structure | Manifest blobs | Bytes |
|---|---|---:|---:|
| Language model | 64 layers, width 5,120 | 851 | 16,994,585,388 |
| Linear attention/GDN | 48 of the 64 language layers | included above | included above |
| Full attention | layers 3, 7, ..., 63 (16 total) | included above | included above |
| MTP draft head | embedded self-draft head; `mtp.fc.weight` is NVFP4 | 15 | 238,934,656 |
| Vision tower | optional for text-only agent work | 333 | 921,497,176 |

The manifest has 1,199 tensor blobs totaling 18,155,017,220 bytes, plus ten
metadata/license/parameter blobs. Ollama's verbose model API reports 1,033
logical tensors because the manifest-level NVFP4 representation and the loaded
logical representation are different accounting layers. The artifact declares
a 262,144-token architectural maximum; the benchmark fixed runtime allocation
to 32,768 tokens.

This structure makes the priorities fairly clear. Removing the unused vision
tower could save about 0.92 GB (5.1% of the tag) and improve load/headroom, but
it should not materially accelerate steady-state text decode. The 0.24 GB MTP
head is small and produces the largest measured decode gain, so removing or
starving it would be counterproductive.

### MTP causal control

The paired `temperature=0`, `seed=1` recurrence task held full reasoning and
the exact answer fixed. Native logprobs parked speculation for the control while
leaving `think: true` unchanged:

| Runtime path | Pass | Wall time | Decode | MTP acceptance |
|---|---:|---:|---:|---:|
| Stock adaptive MTP | yes | 31.470 s | 36.394 tok/s | 89% |
| MTP parked | yes | 72.684 s | 14.033 tok/s | n/a |

MTP therefore delivered 2.59x decode throughput and cut wall time by 56.7% in
this controlled trace. The evidence files are
`20260831T044558Z-qwen3.8-27b-mlx-mtp-auto-temp0.jsonl` and
`20260831T044638Z-qwen3.8-27b-mlx-mtp-off-logprobs-temp0.jsonl`. The parked path
also asks the runtime for logprobs, so it is a close causal bound rather than a
perfectly overhead-free plain decoder.

### Fixed-depth runtime surgery

The experimental Ollama patch pinned speculative draft depth without touching
reasoning, sampling, model weights, or the prompt. All five attempts below
passed and produced byte-identical thinking plus answer output (SHA-256
`f56dc7c33287c39d918b8d0cf46916a5a43e043b67c33338e996af1c7c77bb87`):

| Fixed draft depth | Wall time | Decode | Acceptance |
|---:|---:|---:|---:|
| 0 | 71.515 s | 14.253 tok/s | n/a |
| 2 | 40.240 s | 26.926 tok/s | 95.0% |
| 3 | 31.384 s | 36.810 tok/s | not captured |
| 3, replication | 29.403 s | 38.497 tok/s | 89.5% |
| 4 | 39.388 s | 29.150 tok/s | 83.0% |

Depth 3 is the measured local optimum bracketed by depths 2 and 4. Against
depth 0, its replication was 2.70x faster in decode and 58.9% lower in wall
time. However, stock adaptive MTP was already within 1.1-5.8% of the two fixed
depth-3 observations. That is not enough evidence to ship a pinned fork: the
current production choice remains Ollama's adaptive controller, while the patch
stays as a reproducible measurement instrument.

### Confounds and negative results

The four-task cold run passed 4/4 but decode fell in execution order from 38.104
to 31.351 to 23.685 to 22.176 tokens/s despite 88-93% MTP acceptance. Its median
was 27.518 tokens/s. This makes thermal/runtime history a large enough confound
that future variant comparisons need balanced order, unload-and-cool intervals,
and multiple repetitions rather than a sequential sweep.

A schema-constrained final-answer experiment kept reasoning on and was correct,
but Ollama used two inference passes and emitted prose before the JSON object.
The 21.243 s observation versus a 27.622 s warm marker observation is therefore
exploratory, not a production win: it changed the reasoning trace, failed the
compact-output contract, and the API token counter covered only the final pass.
The runner now records runtime pass count and estimated total emitted tokens so
this accounting mismatch stays visible.

Current MLX-LM evidence also gives no reason to rebuild the head in BF16: its
[Qwen3.5 MTP work](https://github.com/ml-explore/mlx-lm/pull/990) reports that
quantizing `mtp.fc` changed acceptance by only tenths of a percentage point and
speed within noise. A BF16-head experiment should wait for a measured
acceptance problem in this artifact.

### Decision

1. Use the official dense `qwen3.8:27b-mlx` artifact and preserve reasoning.
2. Keep embedded MTP enabled; it is the dominant proven throughput gain.
3. Keep adaptive depth in production; use the fixed-depth patch only for
   controlled research until a balanced agent-shaped corpus shows a repeatable
   win.
4. Pin the Ollama server to a measured context allocation (currently 32k) and
   compact Generalist history at 24k/8k. Ollama's current
   [context policy](https://github.com/ollama/ollama/blob/main/docs/context-length.mdx)
   can otherwise select a much larger allocation from available memory.
5. Next measure agent/tool workloads with balanced variant order, prefix-cache
   reuse, cooling, and sustained-power telemetry. Then investigate
   phase-aware answer/tool emission; do not optimize by suppressing reasoning.
