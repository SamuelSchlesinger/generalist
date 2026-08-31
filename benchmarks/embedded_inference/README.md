# Embedded inference differential checkpoints

These checkpoints compare Generalist's statically linked inference path with
the exact Ollama wrapper used by `qwen3.8:27b-mlx`. They freeze operation-level
parity before optimization, so a faster kernel cannot silently change model
semantics.

The first checkpoint uses the real layer-zero input normalization tensor and a
deterministic 5120-element activation. Both paths perform the same operations:

1. cast the activation to BF16;
2. add one to the checkpoint's zero-centered BF16 normalization weight;
3. run MLX fast RMSNorm with epsilon `1e-6`;
4. cast the BF16 output to float32 and fingerprint every element.

The paired result in `results/rms_norm_layer0.json` is exact, including the
full-output fingerprint. `validate_results.py` makes that evidence part of the
ordinary benchmark gate.

The second checkpoint looks up token `42` in the real 2.54 GB BF16 embedding
table and feeds that row into the same shifted layer-zero RMSNorm. Both the raw
embedding and normalized activation match Ollama over all 5120 elements.

The third checkpoint sends that normalized activation through layer zero's
real `in_proj_qkv` matrix: 10,240 rows of packed NVFP4 weights, 16-value E4M3
group scales, and one F32 global scale. It freezes both the raw MLX
`quantized_matmul` result and Ollama's post-scale BF16 output. All elements
match exactly. The split boundary is intentional: Ollama currently applies the
global scale after the matrix-vector kernel. Generalist's promoted QMV retains
that BF16 boundary inside the kernel, applies the F32 scale, and rounds to BF16
again; the unfused sequence remains the regression oracle.

The remaining checkpoints extend the same chain rather than replacing it:

4. packed QKV/Z and beta/alpha projections for layer zero;
5. zero-state causal convolution and gated-delta recurrence, including both
   next-state tensors;
6. the complete layer-zero residual, GDN output projection, post norm, dense
   SwiGLU MLP, and final residual;
7. all 64 language layers, final RMSNorm, 248,320-way LM head, and greedy token;
8. a persistent two-token session covering every recurrent/KV transition,
   reset, hidden/logit fingerprints, and greedy outputs at positions 0 and 1.

The cached oracle exposed a numerically important execution boundary. Ollama's
SwiGLU is a compiled shapeless MLX graph. Writing the same expression as an
ordinary unfused graph introduced a BF16 rounding boundary and first diverged
at layer 62. Generalist now compiles shapeless SiLU and SwiGLU graphs as well;
the position-one result is exact over all 5120 hidden values and all 248,320
logits, with output token `73`.

The Ollama side is intentionally a source fixture rather than copied runtime
code. Place `ollama_rms_checkpoint_test.go` in
`x/mlxrunner/mlx/generalist_rms_checkpoint_test.go` in a pinned Ollama v0.32.13
checkout, then run:

```sh
OLLAMA_LLM_LIBRARY=mlx_metal_v4 \
GENERALIST_RMS_WEIGHT_PATH="$HOME/.ollama/models/blobs/sha256-ee8700267f26047c63e54902301adeca2d8c0a26784c0242190f689f7ed06114" \
GENERALIST_RMS_WEIGHT_NAME=model.language_model.layers.0.input_layernorm.weight \
GENERALIST_EMBEDDING_WEIGHT_PATH="$HOME/.ollama/models/blobs/sha256-10986d0f406a9fc981e29dc2bf90157ff505ddc74bdb19aef52a40d283247db6" \
GENERALIST_EMBEDDING_WEIGHT_NAME=model.language_model.embed_tokens.weight \
go test ./x/mlxrunner/mlx -run '^TestGeneralist(Embedding|RMS)Checkpoint$' -count=1 -v
```

For the NVFP4 checkpoint, place `ollama_nvfp4_checkpoint_test.go` in
`x/models/qwen3_5/generalist_nvfp4_checkpoint_test.go`, then run:

```sh
OLLAMA_LLM_LIBRARY=mlx_metal_v4 \
GENERALIST_RMS_WEIGHT_PATH="$HOME/.ollama/models/blobs/sha256-ee8700267f26047c63e54902301adeca2d8c0a26784c0242190f689f7ed06114" \
GENERALIST_RMS_WEIGHT_NAME=model.language_model.layers.0.input_layernorm.weight \
GENERALIST_EMBEDDING_WEIGHT_PATH="$HOME/.ollama/models/blobs/sha256-10986d0f406a9fc981e29dc2bf90157ff505ddc74bdb19aef52a40d283247db6" \
GENERALIST_EMBEDDING_WEIGHT_NAME=model.language_model.embed_tokens.weight \
GENERALIST_NVFP4_WEIGHT_PATH="$HOME/.ollama/models/blobs/sha256-46256a4834040a6bc66a237a060f502e6e4e5a1f3bd7c1efaea11a31d9aa2452" \
GENERALIST_NVFP4_WEIGHT_NAME=model.language_model.layers.0.linear_attn.in_proj_qkv.weight \
GENERALIST_NVFP4_SCALE_NAME=model.language_model.layers.0.linear_attn.in_proj_qkv.weight.scale \
GENERALIST_NVFP4_GLOBAL_SCALE_NAME=model.language_model.layers.0.linear_attn.in_proj_qkv.weight.global_scale \
go test ./x/models/qwen3_5 -run '^TestGeneralistNVFP4ProjectionCheckpoint$' -count=1 -v
```

The Generalist side is produced by the feature-gated
`embedded_native_probe` example documented in `native/mlx/README.md`.

For the full and cached model oracles, copy
`ollama_full_decode_checkpoint_test.go` and
`ollama_cached_decode_checkpoint_test.go` into `x/mlxrunner/mlx` in the same
pinned Ollama checkout, then run:

```sh
OLLAMA_LLM_LIBRARY=mlx_metal_v4 \
GENERALIST_MODEL=qwen3.8:27b-mlx \
go test ./x/mlxrunner/mlx \
  -run '^TestGeneralist(Full|Cached)DecodeCheckpoint$' -count=1 -v
```

`results/full_decode.json` preserves the original lazy-load first-token timing
as historical evidence. `results/cached_decode.json` is the current exact
stateful oracle. Its accepted contemporaneous checkpoint run measured
75.021 ms for Generalist and 75.367 ms for Ollama at position one, with full
hidden/logit summaries included. This is evidence of parity, not a general
throughput claim.

`results/metal_dispatch_profile.json` freezes the cumulative trace-guided
optimizations. The debug-only MLX trace labels every ordered compute dispatch;
`analyze_metal_trace.py` segments cached tokens at their `argmax_bfloat16`
pipeline and optionally joins Xcode's encoder-level GPU intervals.
Precomputing 161 shifted RMSNorm weights removes 322 of 2,792 steady dispatches,
but its focused 127-position paired speedup was 1.000587: exact cleanup with a
neutral wall-clock result. Memory-neutral dense gate/up packing then removes
192 dispatches and 64 QMV launches, with a focused paired speedup of 1.002680.

Exact NVFP4 global-scale fusion removes the three post-QMV launches for each of
289 matvecs, taking steady decode from 2,278 to 1,411 dispatches. Three accepted
127-position replicas measured paired speedups of 1.025312, 1.023959, and
1.025925. All reproduced the position-128 hidden/logit fingerprints and the
127-token continuation fingerprint. This path is promoted; the unfused graph
is retained as a benchmark oracle. Dispatch count is still not a linear timing
model: the cumulative exact surgeries nearly halve launches while the latest
kernel surgery improves wall-clock decode by about 2.4--2.6%.

The next exact optimization keeps all 1,411 dispatches but changes large-N
NVFP4 QMV geometry from four to two output results per SIMD-group. It retiled
257 of 289 matvecs per token; 32 small projections remain wide. Four accepted
127-position replicas added paired speedups of 1.010018, 1.010187, 1.007341,
and 1.009854, all exact through position 128. The four-result path remains the
benchmark oracle. Xcode's cross-run median GPU busy time moved from 66.063 ms
to 65.663 ms; the paired wall-clock replicas remain the stronger evidence.

`results/qmv_geometry_sweep.json` records the follow-up exhaustive shape
screen. It inventories all seven decode matvec shapes and tests one, two, four,
and eight results per SIMD-group where they differ from production. Several
15-position signals disappeared at 127 positions. The only remaining signal,
an r1 LM head, then reversed in two of three 255-position replicas. No override
was promoted, so the simpler r2-for-large/r4-for-small policy remains in place.

`results/mtp_depth_control.json` isolates the source of Ollama's remaining
throughput advantage. A locally instrumented build of the pinned Ollama runner
forces speculative draft depths zero through three while retaining its normal
MLX graph and acceptance path. On one deterministic 192-token coding prompt,
all 15 fixed/adaptive replicas produced the same output SHA-256. Fixed depth
zero measured a 14.843 tok/s median, essentially matching Generalist's prior
14.650 tok/s production result. Fixed depth one reached 23.414 tok/s and depth
two reached 26.646 tok/s; depth three fell slightly to 26.164 tok/s as draft
acceptance declined. The production adaptive controller settled near depth two
at 26.554 tok/s. This closes the base-engine question and makes exact MTP
drafting plus batched target verification the next engine priority.

`results/native_speculative_decode.json` freezes that native implementation on
the exact 101-token medium-effort, think-enabled chat fixture. A single
512-token-capable target prefill batch raises prompt throughput from 34.830 to
91.690 tok/s while preserving the completion stream. The complete native MTP
path then reproduces ordinary greedy target decoding for all 192 output tokens.
A stage profile assigns 87.69% of depth-two time to batched target verification.
That evidence motivated a verification-only QMV geometry sweep: r4 beat r2 in
all three interleaved replicas, with paired median gains of 1.028902 end to end
and 1.033051 inside verification. Single-token decode remains r2. The validator
recomputes every speedup, throughput, acceptance ratio, and stage share from
the raw measurements and rejects a disabled-reasoning protocol or output-hash
drift.

`results/adaptive_production.json` freezes the promoted six-patch engine after
compact rollback, exact-scale wide verification, adaptive draft depth, and the
prompt-local draft head were integrated. Its rotating multilingual comparison
recomputes the 34.604 versus 37.368 tok/s medians, 1.078845 paired speedup,
1.012613 maximum spread, identical 64-token target digest, and identical 47/60
acceptance. The same checkpoint records the public provider's xhigh reasoning
run, 4.376x exact prefix-cache gain, 556 ms maximum native transaction, and
519 ms real cancellation/shutdown observation. The machine was not completely
isolated, so the evidence relies on paired mode rotation and low measured
within-mode spread rather than treating an absolute one-off rate as universal.

`results/embedded_vs_ollama.json` is the first direct public-provider
comparison. It uses Generalist's production Qwen renderer once and sends the
same 632 raw prompt bytes to the embedded provider and Ollama with xhigh
reasoning, greedy sampling, a 32,768-token context, and a fixed 128-token output
budget. Every engine sample runs in a fresh child process, Ollama is explicitly
unloaded around embedded samples, and the three replicas rotate AB/BA order.
Warm paired medians measured:

- post-first-delta decode: 32.559 embedded versus 30.014 Ollama tok/s, an 8.54%
  embedded gain;
- end-to-end output: 32.229 versus 28.961 tok/s, an 11.36% embedded gain; and
- first content delta: 70.558 ms embedded versus 188.281 ms Ollama.

The result passed its explicit pre-sample gates: 89.43--94.98% CPU idle, 0--1%
GPU use, and 90--92% free memory. Embedded's 2.79% throughput spread is below
the 8.54% paired advantage. This is nevertheless `gated-normal-desktop-load`
evidence, not a clean-room maximum: ordinary background processes were visible,
the harness sampled load before and after rather than continuously, it did not
enforce a per-process CPU ceiling, and thermal pressure was unavailable. The
validator preserves that boundary as well as the raw samples.

New captures use stricter admission than that frozen first result: the harness
requires nominal `NSProcessInfo` thermal state before and after every child, no
pre-sample background process above 50% of one CPU core, and at least 80%
system-wide CPU idle over the entire child lifetime using Mach host tick
deltas. The old JSON remains labeled under its original weaker contract; these
new checks do not retroactively upgrade it. Unprivileged macOS APIs still do
not expose sufficiently reliable per-process GPU accounting, so paired order,
replication, and spread remain necessary defenses against GPU interference.
The harness checks all pre-sample gates before model load and aborts on the
first contaminated replica, avoiding both misleading output and needless heat.
Its GPU check discards one priming AGX statistics read and then applies the 5%
ceiling to the maximum of three fresh samples spaced 250 ms apart.

Build the paired harness against the persistent native cache, then write a new
JSON capture with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
GENERALIST_MLX_SOURCE_DIR="$generalist_mlx_root/mlx" \
GENERALIST_MLX_C_SOURCE_DIR="$generalist_mlx_root/mlx-c" \
GENERALIST_MLX_BUILD_DIR="$generalist_mlx_root/build/baseline" \
cargo build --release --features embedded-mlx-native \
  --example embedded_ollama_benchmark

./target/release/examples/embedded_ollama_benchmark \
  128 3 5000 /tmp/generalist-embedded-vs-ollama.json
```

The hot benchmark carries a separate exact zero-state control. Its accepted
reference median on the target M4 Pro is 72.549 ms and the run is labeled
`contaminated` above a 1.65x slowdown. A contaminated run can still prove
fingerprints but cannot support a performance comparison. Validate all frozen
evidence without running the model via:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/embedded_inference -p 'test_*.py'
```
