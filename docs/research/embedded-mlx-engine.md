# Embedded Qwen3.8 dense inference engine

Status: first production agent path operational, 2026-08-31.

`generalist --embedded` now exercises this design end to end. The shipped path
uses the promoted exact profile only: full batched prefill, exact fused NVFP4,
the winning single-row/multi-row QMV geometries, and adaptive
depth-two-through-four MTP. The
research harness remains the evidence trail; rejected runtime variants are not
selectable from the agent.

## Objective

Maximize performance *for reasoning* on this M4 Pro (48 GiB unified memory,
20-core GPU), using the dense Qwen3.8 27B model. Reasoning stays enabled and is
part of every acceptance gate. The target hot path has no Ollama daemon, HTTP,
JSON, or generic serving layer:

```text
Agent -> EmbeddedProvider -> dedicated inference thread -> tokenizer/parser
      -> persistent Qwen session -> MLX graph + Metal kernels -> sampler/MTP
      -> reasoning/text deltas -> Agent
```

Model weights remain external data. MLX, MLX-C, the Qwen forward path, cache
management, sampler, and speculative-decoding controller are linked into or
owned by Generalist.

## Checked baseline

The current official `qwen3.8:27b-mlx` artifact is a 27.781B-parameter dense
hybrid model with 64 language layers: 48 gated-delta/linear-attention layers
and 16 full-attention layers. The local artifact accounts for about 16.995 GB
of language weights, 239 MB of MTP data, and 921 MB of vision data.

The reasoning benchmark measured the following controlled decode rates:

- target-only/fixed MTP depth 0: 14.253 generated tokens/s;
- fixed MTP depth 3: 38.497 generated tokens/s;
- observed gain: 2.70x, with byte-identical reasoning and answer on the paired
  replication task.

That is evidence for preserving and tuning MTP, not evidence that all prompts
or draft depths are equivalent. The full benchmark remains the quality gate.

The native feasibility spike is complete:

- MLX commit `3abd0fd6b3eb9d9d3a34cb65e8a2189c57260399`;
- MLX-C commit `fba4470b89073180056c9ea46c443051375f7399`;
- both compiled as static archives with the Metal 4 kernel set;
- a statically linked process selected `Device(gpu, 0)` and evaluated a Metal
  graph successfully;
- `otool -L` showed no MLX/MLX-C dylib dependency;
- the execution thread reported `QOS_CLASS_USER_INITIATED` (`0x19`).

The reproducible probe lives in `native/mlx`.

The language engine has now crossed the first full-model correctness boundary:

- the installed Ollama OCI manifest is normalized into a content-addressed
  `ModelPack`, and all 851 unique language blobs are loaded into one resident
  native model owner;
- the 64-layer token-42 zero-state decode is exact against the pinned Ollama
  implementation: token `1219`, hidden fingerprint `6dfc18bc2b1c2044`, and
  logits fingerprint `4efefa793523b51f`;
- a persistent session owns the 48 convolution/delta-state pairs and 16
  full-attention KV pairs, committing all 128 state roots together with the
  sampled token in one synchronized MLX evaluation;
- the second cached position is also exact: input `1219`, output `73`, hidden
  fingerprint `66245d3b40e34351`, and logits fingerprint
  `f1c93f5eefb964c8`;
- a 127-position continuation ends at input `13`, output `457`, with hidden
  fingerprint `ec8df38e485727c3`, logits fingerprint `b992c48f0705f8e4`, and
  continuation fingerprint `e1185ef3ebbb60ae`;
- reset returns the same session to the exact position-zero checkpoint; and
- the token-only API advances the real cache without copying the hidden vector
  or 248,320 logits back to the CPU for diagnostic summaries.

The accepted two-token checkpoint run measured 75.021 ms for Generalist's
position-one step and 75.367 ms for Ollama's, both including full hidden/logit
summaries. That is parity, not yet a durable throughput win: it is a single
contemporaneous measurement, and the faster token-only path still needs a
clean idle-machine run. Later runs with an unchanged zero-state control at
227 ms instead of 72--75 ms were rejected as system-load contamination.

The first ordered-kernel trace found 2,792 Metal dispatches in a steady cached
token. Exactly 161 BF16 scalar additions rebuilt the model's zero-centered
RMSNorm weights on every token, each paired with an F32-to-BF16 scalar cast.
Materializing those 161 invariant weights with the resident model removed all
322 launches, reducing the steady token graph to 2,470 dispatches without
changing either full-tensor fingerprint or a 15-token greedy continuation.
A focused 127-position run measured a paired speedup of only 1.000587, however,
so the earlier short-run gains were noise-scale. The precompute remains as
exact, low-cost graph cleanup, not as a claimed throughput win.

Memory-neutral gate/up packing across all 64 dense MLPs removed another 192
dispatches and 64 QMV launches per token, reducing the steady graph to 2,278.
Its focused 127-position paired speedup was 1.002680. Packing full-attention
Q/K/V was also exact, but its paired speedup was 0.999801, so that experiment
was reverted rather than generalized from dispatch count alone.

The first material kernel surgery is exact NVFP4 global-scale fusion. The
reference graph rounds the F32 QMV accumulator to BF16, promotes that rounded
value for an F32 scale multiply, then rounds to BF16 again. Generalist performs
those same two conversions in the QMV kernel for scalar and packed per-output
scales. This removes three follow-up launches from each of 289 matvecs: 867
dispatches per token and a 2,278-to-1,411 reduction. Three accepted
127-position replicas measured paired speedups of 1.025312, 1.023959, and
1.025925, winning 125, 123, and 124 pairs. All reproduced the frozen
position-128 hidden fingerprint `ec8df38e485727c3`, logits fingerprint
`b992c48f0705f8e4`, and continuation fingerprint `e1185ef3ebbb60ae`. The
pre-geometry seven-round production validation measured a 68.243 ms cached
median, or 14.654 tokens/s.

The next promoted surgery changes Metal kernel geometry rather than model
arithmetic. Large NVFP4 decode projections now compute two output results per
SIMD-group instead of four, reducing register pressure on this M4 Pro. It
retiled 257 of 289 QMV launches per token; the 32 projections below output
width 4,096 remain wide. Dispatch count is unchanged at 1,411. Four accepted
127-position replicas measured paired speedups of 1.010018, 1.010187,
1.007341, and 1.009854, winning 106, 111, 101, and 106 pairs, while
reproducing every frozen position-128 tensor and continuation fingerprint. A
separate Instruments run observed median GPU busy time of 65.663 ms versus
66.063 ms for the prior exact-scale trace; this cross-run comparison supports,
but does not replace, the paired evidence. The refreshed seven-round
production validation measured 68.261 ms per cached token, or 14.650 tokens/s.

A subsequent shape-by-shape sweep tested r1, r4, and r8 alternatives over all
seven QMV shapes. Every short-screen candidate either reversed at 127 positions
or, for the r1 LM head, produced mixed results in three final 255-position
replicas. All tensor and continuation fingerprints remained exact, but no
geometry override met the wall-clock promotion criterion. The negative result
is frozen in `benchmarks/embedded_inference/results/qmv_geometry_sweep.json`.

The native engine now also owns target-verified MTP decoding and realistic
prompt prefill. On the frozen 101-token medium-effort, think-enabled coding fixture,
one target batch prefills at 91.690 tok/s versus 34.830 tok/s for the original
eight-token chunks and 88.244 tok/s for the contemporaneous Ollama reference.
Target and MTP caches finish at the same prompt position, and depth-two output
is identical to ordinary greedy target decoding over all 192 generated tokens.

The first stage profile ruled out MTP proposal/correction as the primary next
target: multi-row target verification accounted for 2,018.845 of 2,302.305 ms,
or 87.69%. QMV geometry is shape-regime-specific. Although r2 remains best for
single-token large-N decode, r4 won all three paired 192-token runs for batched
verification. Its paired median gain was 1.028902 end to end and 1.033051 for
verification, with the same 126/128 draft acceptance and exact output digest.
Generalist therefore scopes r4 to verification and rejection replay while
retaining r2 for ordinary decode. The next surgery target is inside batched
target verification, especially per-shape QMV policy and fused multi-row layer
graphs, rather than reducing reasoning or draft quality.

The promoted production path now uses an exact-scale wide QMV that reuses
packed weight groups across verification rows, compact recurrent rollback,
and an online controller over draft depths two through four. Its cheap MTP
head evaluates a 65,536-token core, control tokens, and up to 256 high-ID rows
selected from the prompt. Full 248,320-row target verification still governs
every commit, and sustained generated IDs outside the adaptive head switch the
draft back to the full vocabulary. On a stable three-replica multilingual
fixture, prompt-local rows preserved the full-head 47/60 acceptance and target
token hash while improving the paired median decode time by 7.88%.

One exactness result materially changes how surgery must be evaluated. A
mathematically equivalent unfused `silu(gate) * up` first diverged from Ollama
in layer 62 because it introduced a BF16 rounding boundary. Ollama's shapeless
compiled SwiGLU keeps that boundary inside the compiled graph. Reusing compiled
shapeless SiLU/SwiGLU graphs restored exactness through every layer. Operation
graphs and dtype boundaries are therefore part of the oracle, not incidental
implementation detail.

## Runtime ownership

`EmbeddedProvider` is the agent integration seam. Its initializer and all
model calls run on one permanent `generalist-inference` thread. A bounded
channel streams deltas back to the async agent, providing backpressure without
letting MLX objects cross threads. Dropping a completion or rejecting a delta
sets a cancellation flag that the backend must inspect between prefill chunks
and decode steps.

The provider remains semantically request-pure: a request must produce the
same distribution whether or not an internal cache hits. The engine may retain:

- mapped weights and immutable tensor metadata;
- compiled prefill/decode graph variants;
- attention KV and gated-delta recurrent state;
- tokenized prompt prefixes and their cache fingerprints;
- sampler state scoped to one generation;
- MTP acceptance history used only to choose speculative depth.

Every reusable cache entry includes the model digest, tokenizer/template
digest, exact token prefix, relevant inference options, and numerical profile.
Any mismatch invalidates the entry. The provider retains at most two immutable
prefix roots after a successful request: a conservative prompt checkpoint for
identical retries and a longer response checkpoint when the parser-normalized
history is token-identical. It restores the longest matching branch. Cache
replacement is transactional, so cancellation or failed parsing cannot erase
the last usable state.

## Model pack and redistribution

Qwen3.8-27B and the exact Ollama model artifact carry Apache-2.0. Redistribution
is allowed if recipients receive the license, upstream notices/attribution are
retained, modified artifacts are marked, and Qwen marks are used only for
reasonable source identification.

Ordinary Git history will not contain the roughly 18 GB model. Generalist will
own a signed model-pack manifest containing:

- a stable pack format/version and upstream revision;
- shard paths, byte lengths, SHA-256 digests, tensor index, and alignment;
- tokenizer, chat template, reasoning/tool parser configuration;
- language, optional vision, and MTP components declared separately;
- quantization/numerical profile and modification notice;
- Apache-2.0 license plus any upstream NOTICE material.

An installer places verified shards in a profile-scoped model store. An offline
distribution may place the identical pack beside the binary. The engine maps
shards directly; it does not copy weights into the Mach-O image.

## Scheduler and system policy

Only the inference worker receives `QOS_CLASS_USER_INITIATED`. The TUI/input
lane remains separately schedulable. We will not use Mach time-constraint
threads, negative nice values, private APIs, or pretend that thread affinity
can pin work to Apple performance cores. Long-running real-time priority could
starve the UI, audio, and system work and induce worse thermal throttling.

During a foreground generation, the macOS application layer may hold an
`NSActivityUserInitiatedAllowingIdleSystemSleep` activity to prevent App Nap
without preventing idle system sleep. It must end on completion, cancellation,
or failure.

Thermal state is benchmark metadata, not a reason to remove reasoning. Under
sustained heat, the controller may adjust semantics-preserving execution knobs
such as prefill chunk size or MTP draft depth. Any precision change is a new
model variant and must pass the quality suite.

## Compilation experiment matrix

The portable baseline is pinned upstream Release mode (`-O3 -DNDEBUG`) with
MLX's default safe Metal compilation (`-fno-fast-math`). Each experiment gets a
fresh build directory and immutable build manifest.

CPU/link candidates:

1. baseline Release;
2. `-mcpu=native` for this M4 Pro;
3. interprocedural optimization/LTO, with and without native CPU targeting;
4. Rust `target-cpu=native`, thin/fat LTO, one codegen unit, and symbol stripping;
5. profile-guided optimization only after a representative agent trace exists.

Metal/runtime candidates:

1. upstream safe math baseline;
2. relaxed or fast math per custom kernel, never as an unmeasured global flip;
3. NAX versus non-NAX quantized GEMV/GEMM variants on the 20-core M4 Pro GPU;
4. threadgroup shape, vector width, tile size, and split-K sweeps by operation
   shape (prefill and single-token decode are different regimes);
5. ahead-of-time graph variants for common context and batch shapes;
6. `MLX_METAL_FAST_SYNCH` as a separately measured synchronization experiment;
7. model-specific fused kernels rather than broad compiler flags when traces
   show launch or intermediate-memory overhead.

CPU flags are expected to move tokenizer/sampler/orchestration latency more
than dense decode. Metal kernels, tensor layout, cache reuse, and accepted MTP
tokens are expected to dominate decode throughput. Measurements decide.

## Benchmark contract

Every run records:

- source commits, source diff digest, compiler/Xcode/SDK versions, full flags,
  metallib digest, binary digest, and model-pack digest;
- hardware model, CPU/GPU core counts, memory, power mode, QoS, and thermal
  state before/after;
- exact prompt bytes, rendered token IDs, sampler settings, seed, reasoning
  effort, context limit, output limit, and cache state;
- cold load time, tokenizer time, time to first reasoning token, prefill
  tokens/s, target forward steps/s, drafted/accepted tokens, emitted tokens/s,
  peak resident/mapped/Metal memory, and sustained throughput;
- complete reasoning, answer, stop reason, tool calls, and correctness labels.

Microbenchmarks also carry an unchanged exact zero-state control. On this
machine the accepted reference median is 72.549 ms; a run is labeled
`contaminated` when that control is more than 1.65x slower. Semantic parity
from such a run remains useful, but none of its latency, throughput, or speedup
numbers may be promoted.

The first direct production-provider comparison is frozen in
`benchmarks/embedded_inference/results/embedded_vs_ollama.json`. It sends the
same 632-byte production-rendered xhigh-reasoning prompt to both engines with
greedy sampling, alternates AB/BA order over three replicas, unloads Ollama
around embedded samples, and runs each engine sample in a fresh process. Warm
paired medians favor embedded by 8.54% after the first delta and 11.36% end to
end; median first-delta latency is 70.558 ms versus 188.281 ms. The embedded
spread is 2.79%, smaller than the measured gain. Every pre-sample gate passed,
but this evidence is deliberately labeled `gated-normal-desktop-load`: the
machine still had ordinary background processes, telemetry covers only the
before/after boundaries, and macOS thermal pressure was not recorded. It is
credible evidence for this exact workload, not yet a broad or clean-room claim.

For subsequent captures, the harness now also requires nominal
`NSProcessInfo` thermal state before and after each isolated child, caps the
largest pre-sample background process at 50% of one CPU core, and measures
system-wide CPU idle across the full child lifetime from Mach host tick
deltas. This closes the obvious mid-run CPU blind spot without pretending that
unprivileged macOS supplies reliable per-process GPU attribution. The original
artifact retains its weaker label. To avoid treating AGX's retained initial
statistics window as current work, GPU admission discards one priming read and
then applies the 5% limit to the maximum of three fresh samples 250 ms apart.

The experiment ladder is:

1. microkernels: quantized projection, GDN step, attention, normalization,
   sampler, and MTP head;
2. model parity: logits/top-k agreement at layer and token checkpoints;
3. engine throughput: cold/warm prefill and decode across context lengths;
4. agent quality: frozen reasoning and tool-use corpus;
5. sustained operation: interleaved variants with cooldown and a thermal soak.

An optimization is accepted only if the reasoning/completion/exact-answer/tool
protocol gates do not regress. Reference paths remain bit-identical at their
frozen tensor boundaries. A production kernel with a different parallel
reduction tree may be promoted when it computes the same dense-model equations,
keeps deterministic finite outputs, preserves full target-token streams on the
differential corpus, and passes reasoning, tool, cache, and cancellation gates.
Sampled runs use paired seeds and a larger task set; speed alone cannot promote
a numerically different variant.

## Implementation ladder

1. **Native boundary (complete):** static MLX/MLX-C build, Metal resource, GPU
   probe, QoS worker, async streaming/cancellation adapter.
2. **Model-pack ingestion (language path complete):** parse and bound the
   installed manifest, enforce content-addressed regular files, inspect/hash
   shards explicitly, classify language/MTP/vision data, and load all language
   arrays into a resident owner. The distributable signed pack, tokenizer, and
   template bundle remain.
3. **Full-model parity (complete through position 128):** all 64 layers, final
   norm, LM head, greedy token, recurrent state transition, KV transition, and
   reset are covered by exact frozen Ollama oracles.
4. **Streaming decode (complete):** `EmbeddedProvider` owns tokenizer decode,
   reasoning/text/tool parsing, stops, usage, host limits, bounded streaming,
   and cooperative cancellation on its permanent inference thread.
5. **Hybrid cache (complete):** full-attention KV and GDN recurrent state
   persist and grow correctly; 64-token cancellable prefill transactions,
   exact suffix reuse, response-boundary promotion, and transactional fallback
   are active across user turns and tool loops.
6. **MTP (adaptive path complete):** target/MTP prefill, proposal, batched
   full-target verification, compact rejection rollback, correction, and
   atomic commit are live under an online depth-two-through-four controller.
7. **Surgery (underway):** shifted norms, dense gate/up layouts, exact NVFP4
   global scaling, exact-scale wide verification, and the prompt-adaptive draft
   head are promoted. Further model-specific kernels remain individually gated
   by frozen oracles and focused paired benchmarks.
8. **Product path (complete):** `--embedded` selects the in-process engine;
   `--local` remains the separate Ollama/OpenAI-compatible path and Ollama is
   retained as a differential performance and semantics oracle.

The first useful performance target is not merely beating the HTTP path. It is
matching Ollama's token semantics and MTP throughput while owning persistent
cross-turn state. The larger gains should then come from Generalist-specific
prefix reuse, narrower compiled shapes, and fused Qwen kernels.
