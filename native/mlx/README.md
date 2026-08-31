# Static MLX boundary

This directory owns the reproducible native boundary for the embedded Qwen
engine. It is intentionally outside Cargo's default build: ordinary Generalist
development must not download or compile MLX.

`PINS.env` records the exact MLX and MLX-C commits used by Ollama v0.32.13 for
`qwen3.8:27b-mlx`. `build_static_probe.sh` checks out those revisions, builds
both libraries statically, links a process-local GPU probe, places the required
`mlx.metallib` beside it, and confirms with `otool` that no MLX dylib remains.
It also builds a tensor probe that uses MLX-C to load a real safetensors blob,
enumerates every packed array, and forces a small selected tensor through a
Metal reduction.

Six ordered source patches extend the pinned MLX checkout. The first exposes
Ollama's encoded global-scale QMV mode for the deliberately approximate
experiment. The second adds release-inert dispatch labels. The third adds the
production exact-scale QMV variant: it preserves the reference graph's
F32-accumulator-to-BF16 rounding boundary inside the kernel, applies either a
scalar or per-output F32 scale, and rounds to BF16 again. The fourth makes the
large-N NVFP4 QMV output tile selectable by the model wrapper, allowing this
M4 Pro path to use two rather than four results per SIMD-group. The fifth
materializes the exact r1/r2/r4/r8 kernel family needed to tune multi-row
verification independently from single-token decode. Generalist keeps r2 for
ordinary large-N decode. The sixth adds the promoted exact-scale wide-QMV
kernel: it reuses each packed weight group across the small target-verification
batch while preserving the BF16/global-scale/BF16 numerical boundary. Its
parallel reduction tree is mathematically equivalent to the independent-row
oracle but need not be bit-identical.

The user-facing source build is one command from the repository root:

```sh
make embedded
```

Use `make install-embedded` instead when the result should be installed through
Cargo and available as `generalist` on `PATH`.

The wrapper never replaces a dirty source checkout. It reuses
`$XDG_CACHE_HOME/generalist/embedded-mlx` when `XDG_CACHE_HOME` is set, otherwise
`$HOME/Library/Caches/generalist/embedded-mlx`. An explicit
`GENERALIST_MLX_WORK_ROOT` overrides that location. Native developers can run
`native/mlx/build_static_probe.sh` directly; it uses the same cache policy.
The builder also verifies the combined diff of the MLX checkout against the
promoted six-patch fingerprint and requires MLX-C to remain clean. Extra local
source edits are rejected instead of silently becoming a `baseline` build.

To prove direct access to the installed Qwen layer-zero normalization tensor:

```sh
GENERALIST_MLX_TENSOR_PROBE_PATH="$HOME/.ollama/models/blobs/sha256-ee8700267f26047c63e54902301adeca2d8c0a26784c0242190f689f7ed06114" \
GENERALIST_MLX_TENSOR_PROBE_NAME=model.language_model.layers.0.input_layernorm.weight \
native/mlx/build_static_probe.sh
```

Machine-specific CPU experiments must use a distinct variant/build directory:

```sh
GENERALIST_MLX_VARIANT=m4-pro-native-lto \
GENERALIST_MLX_CPU_FLAGS='-O3 -DNDEBUG -mcpu=native' \
GENERALIST_MLX_IPO=ON \
native/mlx/build_static_probe.sh
```

Those flags affect MLX's CPU-side work. Metal kernels are compiled separately;
kernel flag or math-mode experiments require a pinned source patch and the
reasoning-quality gates in `benchmarks/local_reasoning`.

On 2026-08-31, the baseline probe was compiled and run on an M4 Pro with one
Metal GPU. It reported MLX 0.32.1, `Device(gpu, 0)`, QoS class `0x19`, and the
expected GPU result. The linked executable was 19 MiB and depended only on
Apple frameworks/system libraries; the colocated Metal library was 167 MiB.

The build requires Xcode's separately downloadable Metal compiler:

```sh
xcodebuild -downloadComponent MetalToolchain
```

MLX and MLX-C are MIT-licensed, but their compiled closure also retains
third-party code under MIT, BSD, Zlib, and Apache-2.0 terms. The complete
notices are embedded from `THIRD_PARTY_NOTICES.md` and exposed by
`generalist --licenses`. The native build verifies both the fetched dependency
set and the copyright markers in the pinned source before producing a usable
bundle, so an upstream pin or dependency change cannot silently outrun the
notice set.

## Resident Qwen model and cached decode

### Production agent path

The promoted engine is available to the ordinary Generalist binary as
`generalist --embedded`. Its runtime configuration is deliberately fixed:

- precomputed shifted RMS norms;
- packed dense gate/up projections;
- exact fused NVFP4 global scaling;
- r2 single-token QMV for large output widths and r4 for small widths;
- exact-scale wide QMV with weight reuse for batched target verification;
- retained recurrent-state rollback instead of rejection replay;
- exact prompt-boundary target/MTP checkpoints with suffix-only reuse;
- cancellable 64-token prompt transactions;
- adaptive depth-two-through-four MTP speculation; and
- a 65,536-row draft core plus control tokens and up to 256 prompt-local
  high-ID rows, with a full-draft fallback for sustained uncovered IDs.

The target verifier and all committed corrections retain the complete
248,320-row head. The draft shortlist affects only proposal acceptance and
scheduling; it cannot bypass full-model verification.

Rejected row-reuse work and geometry sweeps are not part of this provider
path. The diagnostic benchmarks retain comparison oracles only to make the
promoted result reproducible.

Build the native stack and actual agent together:

```sh
make embedded

./target/release/generalist --embedded
```

Cargo copies the compiled `mlx.metallib` into its output and `include_bytes!`
places it in the Mach-O image. At runtime Generalist verifies and materializes
those bytes for MLX. Weights are discovered from the installed Ollama pack and
are never copied into the executable. The daemon itself is not used.

With the static archives already built, Cargo can compile the feature-gated
Qwen engine directly into a Generalist executable:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
GENERALIST_MLX_SOURCE_DIR="$generalist_mlx_root/mlx" \
GENERALIST_MLX_C_SOURCE_DIR="$generalist_mlx_root/mlx-c" \
GENERALIST_MLX_BUILD_DIR="$generalist_mlx_root/build/baseline" \
cargo build --features embedded-mlx-native --example embedded_decode_benchmark
```

The native model owner loads every unique language safetensors blob once,
prepackages QKV/Z weights and recurrent constants, evaluates the arrays while
the model is resident, and raises MLX's wired-memory limit only to the smaller
of active model memory and Metal's recommended working set. Target-only loads
still exclude the vision tower and MTP weights for controlled base benchmarks;
`MlxQwenModel::load_with_mtp` additionally retains the 15-tensor, one-layer MTP
head and shares the target embedding and LM head. The first target/MTP/target
checkpoint is exact against the pinned Ollama runner at every hidden and logits
boundary.

The sequence owner lazily creates one cache per layer. Linear-attention layers
retain a BF16 `[1, 3, 10240]` convolution window and F32
`[1, 48, 128, 128]` delta state. Full-attention layers retain BF16 key/value
buffers shaped `[1, 4, capacity, 256]`, growing in 256-token blocks. A step
evaluates the greedy token plus all 128 next-state arrays together before it
advances the position, so a failed evaluation cannot partially commit state.

The native API accepts up to 512 prompt tokens per target batch. Production
uses 64-token transactions: this both improved the tested M4 Pro prefill time
and bounds how long synchronous Metal work can delay Ctrl+C. Batched GDN
projection uses the exact custom path through eight rows and MLX's throughput
matmul above that size; both implement the same recurrence and preserve the
downstream greedy stream. Prompt prefill projects only the final target and MTP
rows through the 248,320-entry language head because intermediate logits are
unused. For known nonterminal prompt tokens, the one-layer MTP path retains
only the attention K/V state; its query/output projection, MLP, final norm, and
vocabulary head cannot affect a future token and are skipped. The retained
full-row path remains in target verification and checkpoint diagnostics where
every row is observable.

During prefill, the provider clones the immutable MLX cache roots immediately
before the prompt's final token as a conservative checkpoint. At a valid
response boundary it also reconstructs the parser-normalized assistant history
and compares that complete token prefix with the raw generated stream. A match
retains a second, longer response checkpoint; a mismatch leaves only the
conservative prompt boundary usable. A later request selects the longest
checkpoint whose processed prefix and held token match exactly, then prefills
the suffix. Retaining both bounded branches makes an identical retry cheap
without sacrificing the longer tool-loop continuation. Replacement is
transactional, so cancellation, parser failure, and host-limit rejection
preserve the prior committed checkpoints. Prefix divergence from compaction,
steering, or changed tools resets both target and MTP state. The mutable
post-generation session is released after success or failure; immutable MLX
cache roots are shared by the two retained prefixes rather than copying a
redundant KV history.

The earlier 512-row target-only optimization moved the 101-token fixture from
34.830 to 91.690 prompt tok/s, a 2.633x gain and slightly above the frozen
88.244 tok/s Ollama observation. On the clean six-patch 2026-08-31 build, a
fresh 256-token production-style chunked prefill took 2.280 s versus 2.875 s as
one transaction. Extending that prompt by 64 tokens took 0.625 s from the
255-token checkpoint versus 2.734 s fresh, a 4.376x speedup. All four paths
produced the same checked eight-token greedy continuation; the longest
64-token native transaction was 0.556 s.

A rotating three-replica multilingual decode validation compared the full MTP
head with the production 65K-plus-prompt head at fixed depth four. Both modes
accepted 47 of 60 drafts and produced the same 64-token SHA-256 in every run.
The full-head median was 34.604 tok/s; the adaptive-head median was 37.368
tok/s, with a 1.078845 paired median speedup and only 1.012613 maximum timing
spread. A separate public-provider probe measured 128 production tokens at
23.125 end-to-end tok/s and approximately 30.306 tok/s after the first streamed
delta; cold time to that delta was 1.344 s. Cancelling a real long prefill and
joining the inference worker took 0.519 s, consistent with the 64-token
transaction bound. These are local M4 Pro measurements, not cross-machine
performance guarantees.

After the bounded two-checkpoint policy, a process-isolated three-replica
comparison sent the same 632-byte production-rendered xhigh-reasoning prompt to
embedded and Ollama. Warm medians were 32.559 versus 30.014 tok/s after the
first delta and 32.229 versus 28.961 tok/s end to end; paired median gains were
8.54% and 11.36%. Embedded first-delta latency was 70.558 ms versus 188.281 ms.
All six pre-sample snapshots had 89.43--94.98% CPU idle, 0--1% GPU use, and
90--92% free memory, and each engine ran in a fresh process. The run is still
classified as gated normal desktop load rather than a clean-room maximum:
background processes were present, no per-process ceiling was enforced during
the run, telemetry was sampled only before/after inference, and thermal state
was not recorded.

The current harness tightens future captures with nominal macOS thermal-state
checks, a 50%-of-one-core pre-sample background-process ceiling, and Mach CPU
tick accounting over each complete child run. The frozen comparison remains
under its original contract rather than inheriting those checks after the
fact. GPU admission first primes AGX's retained statistics window, then
requires the maximum of three fresh samples spaced 250 ms apart to remain
within the 5% limit.

`MlxQwenModel::speculative_prefill` initializes target and MTP state at the
same prompt position. `MlxQwenSpeculativeSession` then proposes with the real
one-layer MTP head, verifies all proposed inputs in one target graph, commits
only the accepted prefix, and replays/corrects a rejection atomically. Stage
timers cover proposal, target verification, rejection replay, and correction.

Run the exact, experimental-fusion, profile, cached-checkpoint, reset, and
token-only hot-path checks with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
./target/debug/examples/embedded_decode_benchmark \
  "$generalist_mlx_root/bundle/baseline/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" \
  7
```

The final argument is the number of interleaved exact/fused rounds and greedy
hot-path steps. The benchmark fingerprints both cached positions against
Ollama before reporting timing. It also compares its exact zero-state median
with the accepted 72.549 ms control; `timing_status=contaminated` means the
semantic checks passed but all performance numbers from that run must be
discarded.

Run the realistic native MTP depth and verification-geometry fixture with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
./target/release/examples/embedded_speculative_benchmark \
  "$generalist_mlx_root/build/baseline/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" 192 3 4 standard

./target/release/examples/embedded_speculative_benchmark \
  "$generalist_mlx_root/build/baseline/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" 192 3 4 geometry-sweep

./target/release/examples/embedded_speculative_benchmark \
  "$generalist_mlx_root/build/baseline/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" 64 3 4 shortlist-focused multilingual
```

The prompt token IDs, prompt digest, reasoning mode, output digest, timing
scope, and rotating order are part of the emitted protocol. Prefill is timed
separately; the decode timer begins after prefill produces the first greedy
completion token.

Run the cache/cancellation benchmark with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
GENERALIST_MLX_SOURCE_DIR="$generalist_mlx_root/mlx" \
GENERALIST_MLX_C_SOURCE_DIR="$generalist_mlx_root/mlx-c" \
GENERALIST_MLX_BUILD_DIR="$generalist_mlx_root/build/baseline" \
cargo run --release --features embedded-mlx-native \
  --example embedded_prefix_cache_benchmark -- \
  "$generalist_mlx_root/build/baseline/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" 256 64 64
```

The three trailing values are base prompt tokens, appended tokens, and maximum
transaction tokens. The benchmark rejects any chunked or restored-prefix run
whose greedy continuation differs from a fresh full prefill and reports the
longest native transaction as the observed cancellation polling bound.

Run the same adaptive provider and cancellation path used by
`generalist --embedded` with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
GENERALIST_MLX_SOURCE_DIR="$generalist_mlx_root/mlx" \
GENERALIST_MLX_C_SOURCE_DIR="$generalist_mlx_root/mlx-c" \
GENERALIST_MLX_BUILD_DIR="$generalist_mlx_root/build/baseline" \
cargo run --release --features embedded-mlx-native \
  --example embedded_provider_benchmark -- 128 1024 50
```

The arguments are requested output tokens, long-prompt repetitions, and the
delay before dropping the completion. The output separates model load,
time-to-first-delta, end-to-end rate, approximate post-first-delta rate, and
the real worker-join cancellation latency.

Ordinary builds remain portable because they do not enable
`embedded-mlx-native`. On Apple Silicon, explicitly requesting that feature
without a configured native build fails with a `make embedded` instruction.
Repository-wide portable checks opt into the unavailable-runtime stub with
`GENERALIST_MLX_ALLOW_STUB=1`; user builds should not set it.

## Ordered Metal kernel trace

The pinned MLX patch `0002-metal-kernel-trace-labels.patch` is inert in release
builds. With `MLX_METAL_DEBUG` enabled, it labels compute pipelines and can
emit the exact ordered pipeline list for each MLX compute encoder. Build the
separate debug variant with:

```sh
generalist_mlx_root="${XDG_CACHE_HOME:-$HOME/Library/Caches}/generalist/embedded-mlx"
GENERALIST_MLX_WORK_ROOT="$generalist_mlx_root" \
GENERALIST_MLX_VARIANT=metal-debug \
GENERALIST_MLX_METAL_DEBUG=ON \
GENERALIST_MLX_RUN_PROBE=0 \
native/mlx/build_static_probe.sh

CARGO_TARGET_DIR="$generalist_mlx_root/cargo/metal-debug" \
GENERALIST_MLX_SOURCE_DIR="$generalist_mlx_root/mlx" \
GENERALIST_MLX_C_SOURCE_DIR="$generalist_mlx_root/mlx-c" \
GENERALIST_MLX_BUILD_DIR="$generalist_mlx_root/build/metal-debug" \
cargo build --features embedded-mlx-native --example embedded_decode_trace
```

Set `GENERALIST_MLX_TRACE_KERNELS=1` only for the traced process. The trace
example checks the exact position-zero oracle before emitting eight cached
tokens, and the MLX process writes `MLX_METAL_ENCODER` records at teardown:

```sh
mkdir -p "$generalist_mlx_root/traces"
GENERALIST_MLX_TRACE_KERNELS=1 \
"$generalist_mlx_root/cargo/metal-debug/debug/examples/embedded_decode_trace" \
  "$generalist_mlx_root/bundle/metal-debug/mlx.metallib" \
  "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx" \
  "$HOME/.ollama/models/blobs" 8 > "$generalist_mlx_root/traces/qwen-kernels.log"

python3 benchmarks/embedded_inference/analyze_metal_trace.py \
  "$generalist_mlx_root/traces/qwen-kernels.log"
```

The omitted final path argument traces the production exact-scale kernel.
Append `baseline` after the step count to trace the retained unfused-scale
oracle instead.

For measured encoder-level GPU time, launch the same command through Xcode's
`Metal System Trace`, redirect target stdout to the kernel log, export the
`metal-gpu-intervals` table, and pass it with `--gpu-xml`. The analyzer joins
the two ordered encoder inventories only when their unique counts agree. It
unions overlapping intervals and deliberately reports no per-kernel timing;
pipeline counts are exact, while available timing is encoder- and step-level.

The ordered trace now freezes four cumulative, exact surgeries. Precomputing
all 161 zero-centered RMSNorm weights removes 322 dispatches per cached token
(2,792 to 2,470), but the focused 127-position A/B measured only a 1.000587
paired speedup, so its wall-clock effect is neutral. Packing each dense MLP's
gate/up projections removes another 192 dispatches and 64 QMV launches per
token (2,470 to 2,278); its focused paired speedup was 1.002680.

The promoted exact-scale QMV is the material win. The old graph converted each
of 289 NVFP4 results from BF16 to F32, launched a scale multiply, then converted
back to BF16. Retaining those same two rounding boundaries in the QMV removes
all three follow-up dispatches: 867 launches per token, taking steady decode
from 2,278 to 1,411. Three accepted 127-position replicas measured paired
speedups of 1.025312, 1.023959, and 1.025925, with 125, 123, and 124 wins out of
127. Every replica reproduced the frozen position-128 hidden/logit fingerprints
and continuation fingerprint. The former unfused graph remains available only
as the A/B oracle. Frozen counts and timing evidence live in
`benchmarks/embedded_inference/results/metal_dispatch_profile.json`.

The promoted QMV geometry then reduces register pressure without changing the
graph or dispatch count. For the 257 projections per token whose output width
is at least 4,096, production uses two results per SIMD-group; the remaining 32
small projections stay at four. Four accepted 127-position replicas measured
paired speedups of 1.010018, 1.010187, 1.007341, and 1.009854, with exact
position-128 fingerprints and continuation fingerprints. The retained
four-result path is the A/B oracle. An independent Instruments trace kept
steady dispatches at 1,411 and observed median GPU busy time fall from 66.063
ms to 65.663 ms; that GPU comparison is cross-run, while the paired replicas
are the primary timing evidence.

That r2 result is specific to a single input row. A traced depth-two MTP step
showed target verification consuming 2,018.845 of 2,302.305 ms (87.69%) and
launching 289 multi-row QMV kernels. Interleaving r2 and r4 verification over
three complete 192-token runs made r4 win all three pairs: the paired median
end-to-end speedup was 1.028902 and the verification-only speedup was 1.033051.
All modes produced the same 192-token SHA-256 and accepted 126 of 128 drafts.
The promotion is deliberately scoped to target verification and rejection
replay; proposal, correction, and ordinary decode retain the prior geometry.
The validated evidence lives in
`benchmarks/embedded_inference/results/native_speculative_decode.json`.
