//! Production Qwen3.8 dense backend for the in-process MLX provider.
//!
//! This module intentionally exposes one production numerical/runtime profile:
//! the promoted exact-scale NVFP4 graph, fused small-batch verification QMV,
//! and cost-aware depth-two through depth-four MTP decoding. Floating-point reduction
//! trees may differ from token-at-a-time evaluation, but every path evaluates
//! the same dense Qwen3.8 model equations.

#[cfg(generalist_mlx_native_linked)]
use super::mlx::MlxRuntime;
use super::mlx::{MlxQwenModel, MlxQwenSpeculativeSession, MlxSpeculativeStep};
#[cfg(generalist_mlx_native_linked)]
use super::model_pack::ModelPack;
use super::{GenerationOutput, LocalCompletionRequest, LocalInferenceBackend};
use crate::error::{Error, Result};
use crate::types::{
    CompletionDelta, CompletionLimits, CompletionResponse, ContentBlock, Message, StopReason,
    ToolDef, Usage,
};
#[cfg(generalist_mlx_native_linked)]
use ring::digest::{Context, SHA256};
use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeSet, VecDeque};
#[cfg(generalist_mlx_native_linked)]
use std::env;
use std::fmt::Display;
#[cfg(generalist_mlx_native_linked)]
use std::fs::{self, File};
#[cfg(generalist_mlx_native_linked)]
use std::io::{Read, Write};
#[cfg(generalist_mlx_native_linked)]
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;
use uuid::Uuid;

pub const DEFAULT_EMBEDDED_MODEL: &str = "qwen3.8:27b-mlx";

#[cfg(generalist_mlx_native_linked)]
const MODEL_NAME: &str = "qwen3.8";
#[cfg(generalist_mlx_native_linked)]
const MODEL_TAG: &str = "27b-mlx";
#[cfg(generalist_mlx_native_linked)]
const CONTEXT_TOKENS: usize = 262_144;
const DEFAULT_OUTPUT_TOKENS: usize = 32_768;
const MTP_MIN_DRAFT_DEPTH: usize = 2;
const MTP_MAX_DRAFT_DEPTH: usize = 4;
const MTP_DRAFT_DEPTH_COUNT: usize = MTP_MAX_DRAFT_DEPTH - MTP_MIN_DRAFT_DEPTH + 1;
const MTP_WARMUP_DEPTHS: [usize; 6] = [2, 2, 3, 3, 4, 4];
const MTP_PROBE_INTERVAL_BASE: usize = 32;
const MTP_PROBE_INTERVAL_MAX: usize = 512;
const MTP_PROBE_CYCLES: usize = 2;
const MTP_COST_EWMA_ALPHA: f64 = 0.30;
const MTP_ACCEPTANCE_EWMA_ALPHA: f64 = 0.10;
const MTP_COST_INNOVATION_LIMIT: f64 = 0.25;
const MTP_MIN_ACCEPTANCE_SAMPLES: usize = 10;
const MTP_DEPTH_HYSTERESIS: f64 = 0.03;
const MTP_DRAFT_VOCAB_LIMIT: usize = 65_536;
const MTP_DRAFT_SPECIAL_START: i32 = 248_000;
const MTP_DRAFT_EXTRA_TOKEN_LIMIT: usize = 256;
const MTP_OUTPUT_VOCAB_WINDOW: usize = 16;
const MTP_OUTPUT_FULL_VOCAB_MIN_SAMPLES: usize = 8;
const MTP_OUTPUT_FULL_VOCAB_MIN_HIGH_TOKENS: usize = 3;
const VERIFICATION_QMV_RESULTS: i32 = 4;
const MAX_NATIVE_COMMIT_TOKENS: usize = 8;
// Bound one uninterruptible Metal prefill transaction. The production value is
// benchmarked below MLX's 512-token maximum so Ctrl+C can stop sustained GPU
// work promptly even for a very long history.
const PREFILL_CHUNK_TOKENS: usize = 64;
const IM_START_TOKEN: &str = "<|im_start|>";
const IM_END_TOKEN: &str = "<|im_end|>";
#[cfg(generalist_mlx_native_linked)]
const END_OF_TEXT_TOKEN: &str = "<|endoftext|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const ASSISTANT_GENERATION_PREFIX: &str = "<|im_start|>assistant\n<think>\n";
const XHIGH_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

#[cfg(generalist_mlx_native_linked)]
static EMBEDDED_METALLIB: &[u8] = include_bytes!(env!("GENERALIST_EMBEDDED_MLX_METALLIB_PATH"));

/// Resident dense Qwen3.8 model, tokenizer, and reusable exact speculative
/// cache. The complete value lives only on Generalist's inference thread.
pub struct QwenEmbeddedBackend {
    tokenizer: Tokenizer,
    model: MlxQwenModel,
    session: MlxQwenSpeculativeSession,
    stop_token_ids: BTreeSet<u32>,
    max_context_tokens: usize,
    prompt_caches: Vec<PromptCacheCheckpoint>,
    depth_controller: AdaptiveMtpController,
    draft_vocab_limit: usize,
    draft_extra_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
struct AcceptanceEstimate {
    probability: f64,
    reached: usize,
}

impl Default for AcceptanceEstimate {
    fn default() -> Self {
        Self {
            probability: 1.0,
            reached: 0,
        }
    }
}

impl AcceptanceEstimate {
    fn observe(&mut self, accepted: bool) {
        let sample = if accepted { 1.0 } else { 0.0 };
        if self.reached == 0 {
            self.probability = sample;
        } else {
            self.probability = (1.0 - MTP_ACCEPTANCE_EWMA_ALPHA) * self.probability
                + MTP_ACCEPTANCE_EWMA_ALPHA * sample;
        }
        self.reached += 1;
    }
}

/// Online controller for the draft depths that improved throughput on the
/// target M4 Pro. It optimizes expected committed target tokens per measured
/// wall-clock second, including rejection recovery and host overhead.
struct AdaptiveMtpController {
    acceptance: [AcceptanceEstimate; MTP_MAX_DRAFT_DEPTH],
    cost_seconds: [Option<f64>; MTP_DRAFT_DEPTH_COUNT],
    warmup_index: usize,
    selected_depth: usize,
    last_depth: Option<usize>,
    cycles_since_probe: usize,
    probe_depth: Option<usize>,
    probe_cycles_remaining: usize,
    probe_interval: usize,
}

impl Default for AdaptiveMtpController {
    fn default() -> Self {
        Self {
            acceptance: [AcceptanceEstimate::default(); MTP_MAX_DRAFT_DEPTH],
            cost_seconds: [None; MTP_DRAFT_DEPTH_COUNT],
            warmup_index: 0,
            selected_depth: MTP_MIN_DRAFT_DEPTH,
            last_depth: None,
            cycles_since_probe: 0,
            probe_depth: None,
            probe_cycles_remaining: 0,
            probe_interval: MTP_PROBE_INTERVAL_BASE,
        }
    }
}

impl AdaptiveMtpController {
    fn next_depth(&mut self) -> usize {
        if let Some(depth) = MTP_WARMUP_DEPTHS.get(self.warmup_index) {
            return *depth;
        }
        if self.probe_cycles_remaining != 0 {
            return self
                .probe_depth
                .expect("an active MTP probe always has a depth");
        }

        let previous_selection = self.selected_depth;
        self.select_fastest_depth();
        if self.selected_depth != previous_selection {
            self.probe_interval = MTP_PROBE_INTERVAL_BASE;
            self.cycles_since_probe = 0;
        }
        if self.cycles_since_probe >= self.probe_interval {
            if let Some(depth) = self.next_probe_depth() {
                self.probe_depth = Some(depth);
                self.probe_cycles_remaining = MTP_PROBE_CYCLES;
                return depth;
            }
        }
        self.selected_depth
    }

    fn observe(
        &mut self,
        requested_depth: usize,
        drafted: usize,
        accepted: usize,
        elapsed_seconds: f64,
    ) {
        // Tail-limited calls do not represent the requested kernel shape and
        // must not train either the scheduler or its transition-cost filter.
        if drafted != requested_depth || !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
            self.last_depth = None;
            return;
        }

        for position in 0..requested_depth.min(MTP_MAX_DRAFT_DEPTH) {
            // A later draft is reached only if every preceding draft matched.
            if accepted < position {
                break;
            }
            self.acceptance[position].observe(accepted > position);
        }

        // The first cycle after changing M is polluted by graph compilation
        // and allocator transitions. Only compare steady same-shape cycles.
        if self.last_depth == Some(requested_depth) {
            let index = depth_index(requested_depth);
            self.cost_seconds[index] = Some(match self.cost_seconds[index] {
                None => elapsed_seconds,
                Some(previous) => {
                    let bounded = elapsed_seconds.clamp(
                        previous * (1.0 - MTP_COST_INNOVATION_LIMIT),
                        previous * (1.0 + MTP_COST_INNOVATION_LIMIT),
                    );
                    (1.0 - MTP_COST_EWMA_ALPHA) * previous + MTP_COST_EWMA_ALPHA * bounded
                }
            });
        }
        self.last_depth = Some(requested_depth);

        if self.warmup_index < MTP_WARMUP_DEPTHS.len() {
            debug_assert_eq!(MTP_WARMUP_DEPTHS[self.warmup_index], requested_depth);
            self.warmup_index += 1;
            return;
        }
        if self.probe_cycles_remaining != 0 && self.probe_depth == Some(requested_depth) {
            self.probe_cycles_remaining -= 1;
            if self.probe_cycles_remaining == 0 {
                self.probe_depth = None;
                let previous_selection = self.selected_depth;
                self.select_fastest_depth();
                self.probe_interval = if self.selected_depth == previous_selection {
                    (self.probe_interval * 2).min(MTP_PROBE_INTERVAL_MAX)
                } else {
                    MTP_PROBE_INTERVAL_BASE
                };
                self.cycles_since_probe = 0;
            }
            return;
        }
        self.cycles_since_probe = self.cycles_since_probe.saturating_add(1);
    }

    fn select_fastest_depth(&mut self) {
        let Some(current_score) = self.score(self.selected_depth) else {
            if let Some((depth, _)) = self.best_scored_depth() {
                self.selected_depth = depth;
            }
            return;
        };
        let Some((challenger, challenger_score)) = (MTP_MIN_DRAFT_DEPTH..=MTP_MAX_DRAFT_DEPTH)
            .filter(|depth| *depth != self.selected_depth)
            .filter_map(|depth| self.score(depth).map(|score| (depth, score)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            return;
        };
        if challenger_score > current_score * (1.0 + MTP_DEPTH_HYSTERESIS) {
            self.selected_depth = challenger;
        }
    }

    fn best_scored_depth(&self) -> Option<(usize, f64)> {
        (MTP_MIN_DRAFT_DEPTH..=MTP_MAX_DRAFT_DEPTH)
            .filter_map(|depth| self.score(depth).map(|score| (depth, score)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
    }

    fn next_probe_depth(&self) -> Option<usize> {
        (self.selected_depth < MTP_MAX_DRAFT_DEPTH).then_some(self.selected_depth + 1)
    }

    fn score(&self, depth: usize) -> Option<f64> {
        let cost = self.cost_seconds[depth_index(depth)]?;
        let mut expected_committed = 1.0;
        let mut prefix_probability = 1.0;
        let mut inherited_probability = 1.0;
        for estimate in self.acceptance.iter().take(depth) {
            if estimate.reached >= MTP_MIN_ACCEPTANCE_SAMPLES {
                inherited_probability = estimate.probability.clamp(0.0, 1.0);
            }
            prefix_probability *= inherited_probability;
            expected_committed += prefix_probability;
        }
        Some(expected_committed / cost)
    }
}

fn depth_index(depth: usize) -> usize {
    debug_assert!((MTP_MIN_DRAFT_DEPTH..=MTP_MAX_DRAFT_DEPTH).contains(&depth));
    depth - MTP_MIN_DRAFT_DEPTH
}

fn is_high_ordinary_token(token_id: i32) -> bool {
    (MTP_DRAFT_VOCAB_LIMIT as i32..MTP_DRAFT_SPECIAL_START).contains(&token_id)
}

fn draft_vocab_observation(token_id: i32, extra_token_ids: &[i32]) -> Option<bool> {
    (0..MTP_DRAFT_SPECIAL_START).contains(&token_id).then_some(
        is_high_ordinary_token(token_id) && extra_token_ids.binary_search(&token_id).is_err(),
    )
}

fn prompt_draft_extra_token_ids(token_ids: &[i32]) -> Vec<i32> {
    let mut selected = BTreeSet::new();
    for &token_id in token_ids.iter().rev() {
        if is_high_ordinary_token(token_id) {
            selected.insert(token_id);
            if selected.len() == MTP_DRAFT_EXTRA_TOKEN_LIMIT {
                break;
            }
        }
    }
    selected.into_iter().collect()
}

fn output_window_prefers_full_draft_vocab(window: &VecDeque<bool>) -> bool {
    if window.len() < MTP_OUTPUT_FULL_VOCAB_MIN_SAMPLES {
        return false;
    }
    let high = window.iter().filter(|&&high_id| high_id).count();
    high >= MTP_OUTPUT_FULL_VOCAB_MIN_HIGH_TOKENS && high * 4 >= window.len()
}

/// An exact checkpoint immediately before one held token. Reuse is
/// conditional on an exact comparison with the next fully rendered prompt, so
/// parser normalization cannot make a merely similar history enter the native
/// cache.
struct PromptCacheCheckpoint {
    session: MlxQwenSpeculativeSession,
    processed_token_ids: Vec<i32>,
    held_token_id: i32,
}

impl QwenEmbeddedBackend {
    pub fn load(model_name: &str) -> Result<Self> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = model_name;
            Err(embedded_error(
                "the native engine was not linked; run `make embedded` from the Generalist \
                 repository on Apple Silicon macOS",
            ))
        }

        #[cfg(generalist_mlx_native_linked)]
        {
            validate_model_name(model_name)?;
            let (manifest, blob_root) = resolve_model_pack_paths()?;
            let pack = ModelPack::open_ollama(&manifest, &blob_root)?;
            validate_model_config(&pack)?;

            let tokenizer_descriptor = pack.document("tokenizer.json")?;
            let tokenizer_path = pack.blob_path(tokenizer_descriptor);
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
                embedded_error(format!(
                    "failed to load tokenizer {}: {error}",
                    tokenizer_path.display()
                ))
            })?;
            validate_tokenizer(&tokenizer)?;

            let metallib_path = materialize_embedded_metallib()?;
            let runtime = MlxRuntime::initialize(&metallib_path)?;
            let (mut model, checkpoint) = MlxQwenModel::load_with_mtp(&runtime, &pack)?;
            if checkpoint.layer_count != 64 || checkpoint.mtp_layer_count != 1 {
                return Err(embedded_error(format!(
                    "model load produced {} language layers and {} MTP layers; expected 64 and 1",
                    checkpoint.layer_count, checkpoint.mtp_layer_count
                )));
            }
            let mut session = model.start_speculative_session()?;
            session.set_verification_qmv_results_per_simdgroup(VERIFICATION_QMV_RESULTS)?;
            session.set_verification_wide_qmv(true)?;
            session.set_mtp_draft_vocab_limit(MTP_DRAFT_VOCAB_LIMIT)?;
            session.set_direct_cache_rollback(true)?;

            let stop_token_ids = [IM_END_TOKEN, END_OF_TEXT_TOKEN]
                .into_iter()
                .map(|token| required_token_id(&tokenizer, token))
                .collect::<Result<_>>()?;
            Ok(Self {
                tokenizer,
                model,
                session,
                stop_token_ids,
                max_context_tokens: CONTEXT_TOKENS,
                prompt_caches: Vec::new(),
                depth_controller: AdaptiveMtpController::default(),
                draft_vocab_limit: MTP_DRAFT_VOCAB_LIMIT,
                draft_extra_token_ids: Vec::new(),
            })
        }
    }

    fn set_draft_vocab_limit(&mut self, vocab_limit: usize) -> Result<()> {
        if self.draft_vocab_limit == vocab_limit {
            return Ok(());
        }
        self.session.set_mtp_draft_vocab_limit(vocab_limit)?;
        self.draft_vocab_limit = vocab_limit;
        // Proposal cost and acceptance both change with the head. Do not let
        // measurements from one vocabulary regime steer the other one.
        self.depth_controller = AdaptiveMtpController::default();
        Ok(())
    }

    fn prefill_prompt(
        &mut self,
        input_token_ids: &[i32],
        output: &GenerationOutput<'_>,
    ) -> Result<(MlxSpeculativeStep, usize, Option<PromptCacheCheckpoint>)> {
        let cache_read_tokens = best_reusable_prefix_len(
            self.prompt_caches.iter().map(|checkpoint| {
                (
                    checkpoint.processed_token_ids.as_slice(),
                    checkpoint.held_token_id,
                )
            }),
            input_token_ids,
        );

        let mut cursor = if let Some(cache_read_tokens) = cache_read_tokens {
            let restored = self
                .prompt_caches
                .iter()
                .find(|checkpoint| {
                    reusable_prefix_len(
                        &checkpoint.processed_token_ids,
                        checkpoint.held_token_id,
                        input_token_ids,
                    ) == Some(cache_read_tokens)
                })
                .expect("cache length came from one of the bounded checkpoints")
                .session
                .try_clone()?;
            self.session = restored;
            cache_read_tokens
        } else {
            self.session.reset()?;
            0
        };
        self.session
            .set_verification_qmv_results_per_simdgroup(VERIFICATION_QMV_RESULTS)?;
        self.session.set_verification_wide_qmv(true)?;
        self.session
            .set_mtp_draft_vocab_limit(self.draft_vocab_limit)?;
        self.model
            .set_mtp_draft_extra_tokens(&mut self.session, &self.draft_extra_token_ids)?;
        self.session.set_direct_cache_rollback(true)?;

        let checkpoint_position = input_token_ids.len() - 1;
        while cursor < checkpoint_position {
            if output.is_cancelled() {
                return Err(cancelled());
            }
            let end = (cursor + PREFILL_CHUNK_TOKENS).min(checkpoint_position);
            let held_token_id = input_token_ids[end];
            let step = self.model.speculative_prefill_chunk(
                &mut self.session,
                &input_token_ids[cursor..end],
                Some(held_token_id),
            )?;
            validate_prefill_transition(&step, end - 1, input_token_ids[end - 1], held_token_id)?;
            cursor = end;
            if output.is_cancelled() {
                return Err(cancelled());
            }
        }

        // This conservative boundary is always a prefix of the current
        // rendered prompt. It remains available for an exact retry even when
        // a longer token-identical raw-response checkpoint is also retained.
        let next_prompt_cache = if checkpoint_position != 0 {
            Some(PromptCacheCheckpoint {
                session: self.session.try_clone()?,
                processed_token_ids: input_token_ids[..checkpoint_position].to_vec(),
                held_token_id: input_token_ids[checkpoint_position],
            })
        } else {
            None
        };

        if output.is_cancelled() {
            return Err(cancelled());
        }
        let seed = self.model.speculative_prefill_chunk(
            &mut self.session,
            &input_token_ids[checkpoint_position..],
            None,
        )?;
        if seed.position != checkpoint_position
            || seed.input_token_id != input_token_ids[checkpoint_position]
            || seed.output_token_ids.len() != 1
        {
            return Err(embedded_error(
                "native terminal prefill returned an inconsistent seed",
            ));
        }
        Ok((seed, cache_read_tokens.unwrap_or(0), next_prompt_cache))
    }

    fn complete_inner(
        &mut self,
        request: LocalCompletionRequest,
        output: &mut GenerationOutput<'_>,
    ) -> Result<CompletionResponse> {
        if output.is_cancelled() {
            return Err(cancelled());
        }

        let prompt = render_prompt(&request)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| embedded_error(format!("failed to tokenize chat prompt: {error}")))?;
        let input_token_ids = encoding
            .get_ids()
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id)
                    .map_err(|_| embedded_error("tokenizer emitted an ID outside native i32 range"))
            })
            .collect::<Result<Vec<_>>>()?;
        if input_token_ids.is_empty() {
            return Err(embedded_error(
                "rendered chat prompt tokenized to no tokens",
            ));
        }
        if input_token_ids.len() >= self.max_context_tokens {
            return Err(embedded_error(format!(
                "prompt uses {} tokens, exceeding the embedded context allowance of fewer than {}",
                input_token_ids.len(),
                self.max_context_tokens
            )));
        }
        let requested_output_tokens = request
            .max_tokens
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_OUTPUT_TOKENS);
        let output_limit = requested_output_tokens.min(
            self.max_context_tokens
                .saturating_sub(input_token_ids.len()),
        );
        if output_limit == 0 {
            return Err(embedded_error("no context remains for an output token"));
        }

        self.set_draft_vocab_limit(MTP_DRAFT_VOCAB_LIMIT)?;
        self.draft_extra_token_ids = prompt_draft_extra_token_ids(&input_token_ids);

        let (seed, cache_read_tokens, next_prompt_cache) =
            self.prefill_prompt(&input_token_ids, output)?;
        if output.is_cancelled() {
            return Err(cancelled());
        }

        let mut parser = QwenOutputParser::new(&request.tools, request.limits);
        let mut decode_stream = self.tokenizer.decode_stream(false);
        let mut decoded_token_ids = Vec::with_capacity(output_limit.min(4096));
        let mut generated_token_ids = Vec::with_capacity(output_limit.min(4096));
        let mut generated_tokens = 0usize;
        let mut stopped = false;
        let mut commit = seed.output_token_ids;
        let mut draft_vocab_window = VecDeque::with_capacity(MTP_OUTPUT_VOCAB_WINDOW);
        // A speculative call can return tokens beyond an EOS token. Keep its
        // exact input checkpoint so the response boundary can be replayed
        // with a shorter commit instead of caching the unobserved suffix.
        let mut commit_origin: Option<(MlxQwenSpeculativeSession, usize)> = None;
        let mut completion_session: Option<MlxQwenSpeculativeSession> = None;

        'generation: loop {
            if commit.is_empty() {
                return Err(embedded_error(
                    "native speculative decode committed no tokens",
                ));
            }
            for (commit_index, token_id) in commit.iter().copied().enumerate() {
                if output.is_cancelled() {
                    return Err(cancelled());
                }
                if generated_tokens == output_limit {
                    break 'generation;
                }
                let token_id = u32::try_from(token_id)
                    .map_err(|_| embedded_error("native decoder emitted a negative token ID"))?;
                generated_token_ids.push(token_id as i32);
                generated_tokens += 1;
                if let Some(uncovered) =
                    draft_vocab_observation(token_id as i32, &self.draft_extra_token_ids)
                {
                    if draft_vocab_window.len() == MTP_OUTPUT_VOCAB_WINDOW {
                        draft_vocab_window.pop_front();
                    }
                    draft_vocab_window.push_back(uncovered);
                }
                if self.stop_token_ids.contains(&token_id) {
                    stopped = true;
                    completion_session = Some(if commit_index + 1 == commit.len() {
                        self.session.try_clone()?
                    } else {
                        let (mut replay, draft_depth) = commit_origin.take().ok_or_else(|| {
                            embedded_error(
                                "initial speculative seed unexpectedly contained multiple tokens",
                            )
                        })?;
                        let replayed = self.model.speculative_step(
                            &mut replay,
                            draft_depth,
                            commit_index + 1,
                        )?;
                        if replayed.output_token_ids != commit[..=commit_index] {
                            return Err(embedded_error(
                                "shortened EOS replay changed the committed target prefix",
                            ));
                        }
                        replay
                    });
                    break 'generation;
                }
                decoded_token_ids.push(token_id);
                if let Some(fragment) = decode_stream.step(token_id).map_err(|error| {
                    embedded_error(format!(
                        "failed to decode generated token {token_id}: {error}"
                    ))
                })? {
                    emit_all(output, parser.push(&fragment)?)?;
                }
            }

            if generated_tokens == output_limit {
                completion_session = Some(self.session.try_clone()?);
                break;
            }
            if output.is_cancelled() {
                return Err(cancelled());
            }
            if self.draft_vocab_limit != 0
                && output_window_prefers_full_draft_vocab(&draft_vocab_window)
            {
                // Prompt-local rows cover repeated uncommon terms cheaply.
                // If the target instead sustains high-ID tokens absent from
                // that adaptive head, retain the full head for this turn.
                self.session.set_mtp_draft_vocab_limit(0)?;
                self.draft_vocab_limit = 0;
                self.depth_controller = AdaptiveMtpController::default();
            }
            let remaining = output_limit - generated_tokens;
            let draft_depth = self.depth_controller.next_depth();
            let step_started = Instant::now();
            let origin = self.session.try_clone()?;
            let step = self.model.speculative_step(
                &mut self.session,
                draft_depth,
                remaining.min(MAX_NATIVE_COMMIT_TOKENS),
            )?;
            self.depth_controller.observe(
                draft_depth,
                step.drafted,
                step.accepted,
                step_started.elapsed().as_secs_f64(),
            );
            commit = step.output_token_ids;
            commit_origin = Some((origin, draft_depth));
            if output.is_cancelled() {
                return Err(cancelled());
            }
        }

        // DecodeStream can retain an incomplete multi-token Unicode sequence.
        // Reconcile once, linearly, against the complete decode so no trailing
        // user-visible bytes disappear at EOS or the output limit.
        drop(decode_stream);
        let complete_decoded =
            self.tokenizer
                .decode(&decoded_token_ids, false)
                .map_err(|error| {
                    embedded_error(format!("failed to finalize generated text: {error}"))
                })?;
        let streamed_len = parser.decoded_source().len();
        if !complete_decoded.starts_with(parser.decoded_source()) {
            return Err(embedded_error(
                "incremental tokenizer decode disagreed with complete decode",
            ));
        }
        emit_all(output, parser.push(&complete_decoded[streamed_len..])?)?;
        let (trailing_deltas, content) = parser.finish()?;
        emit_all(output, trailing_deltas)?;

        let mut stop_reason = if stopped {
            StopReason::EndTurn
        } else {
            StopReason::MaxTokens
        };
        if stopped
            && content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        {
            stop_reason = StopReason::ToolUse;
        }
        let response = CompletionResponse {
            content,
            stop_reason,
            usage: Some(Usage {
                input_tokens: input_token_ids.len().saturating_sub(cache_read_tokens) as u64,
                output_tokens: generated_tokens as u64,
                cache_read_input_tokens: (cache_read_tokens != 0)
                    .then_some(cache_read_tokens as u64),
                cache_creation_input_tokens: None,
            }),
        };
        request.limits.validate_response(&response)?;
        let promote_response_checkpoint = response_checkpoint_matches_rendered_prefix(
            &self.tokenizer,
            &prompt,
            &response.content,
            &input_token_ids,
            &generated_token_ids,
        )?;
        let completion_session = completion_session.ok_or_else(|| {
            embedded_error("generation ended without an exact response-boundary checkpoint")
        })?;
        let held_token_id = *generated_token_ids
            .last()
            .ok_or_else(|| embedded_error("generation ended without a token"))?;
        let mut processed_token_ids =
            Vec::with_capacity(input_token_ids.len() + generated_token_ids.len() - 1);
        processed_token_ids.extend_from_slice(&input_token_ids);
        processed_token_ids
            .extend_from_slice(&generated_token_ids[..generated_token_ids.len().saturating_sub(1)]);
        let response_prompt_cache = PromptCacheCheckpoint {
            session: completion_session,
            processed_token_ids,
            held_token_id,
        };
        // Promote only after the complete response has passed every parser,
        // stream, and host limit. Cancellation or failure therefore preserves
        // the last committed conversation checkpoint for a correction/retry.
        // The working session and response checkpoint share their immutable
        // prefix roots. Release the mutable owner rather than retaining two
        // long context states between calls.
        self.session.reset()?;
        // Keep two bounded branches when safe: the conservative current-prompt
        // boundary supports an exact retry, while the longer raw-response
        // boundary supports the normal next turn. Their immutable prefix roots
        // are shared by the native session representation. A later request
        // always selects the longest token-identical prefix.
        self.prompt_caches.clear();
        self.prompt_caches.extend(next_prompt_cache);
        if promote_response_checkpoint {
            self.prompt_caches.push(response_prompt_cache);
        }
        Ok(response)
    }
}

fn best_reusable_prefix_len<'a>(
    checkpoints: impl IntoIterator<Item = (&'a [i32], i32)>,
    input_token_ids: &[i32],
) -> Option<usize> {
    checkpoints
        .into_iter()
        .filter_map(|(processed_token_ids, held_token_id)| {
            reusable_prefix_len(processed_token_ids, held_token_id, input_token_ids)
        })
        .max()
}

fn reusable_prefix_len(
    processed_token_ids: &[i32],
    held_token_id: i32,
    input_token_ids: &[i32],
) -> Option<usize> {
    let processed = processed_token_ids.len();
    (processed < input_token_ids.len()
        && input_token_ids.starts_with(processed_token_ids)
        && input_token_ids[processed] == held_token_id)
        .then_some(processed)
}

fn response_checkpoint_matches_rendered_prefix(
    tokenizer: &Tokenizer,
    current_prompt: &str,
    response_content: &[ContentBlock],
    input_token_ids: &[i32],
    generated_token_ids: &[i32],
) -> Result<bool> {
    let history_prefix = current_prompt
        .strip_suffix(ASSISTANT_GENERATION_PREFIX)
        .ok_or_else(|| embedded_error("rendered prompt lost its assistant generation prefix"))?;
    let mut completed_history = history_prefix.to_owned();
    render_assistant_message(
        &mut completed_history,
        &Message::assistant(response_content.to_vec()),
    )?;
    let normalized = tokenizer
        .encode(completed_history, false)
        .map_err(|error| {
            embedded_error(format!(
                "failed to tokenize completed response history: {error}"
            ))
        })?;
    let normalized = normalized.get_ids();
    Ok(normalized_response_starts_with_raw_tokens(
        normalized,
        input_token_ids,
        generated_token_ids,
    ))
}

fn normalized_response_starts_with_raw_tokens(
    normalized: &[u32],
    input_token_ids: &[i32],
    generated_token_ids: &[i32],
) -> bool {
    let expected_len = input_token_ids.len() + generated_token_ids.len();
    normalized.len() >= expected_len
        && normalized[..input_token_ids.len()]
            .iter()
            .copied()
            .eq(input_token_ids.iter().map(|token_id| *token_id as u32))
        && normalized[input_token_ids.len()..expected_len]
            .iter()
            .copied()
            .eq(generated_token_ids.iter().map(|token_id| *token_id as u32))
}

fn validate_prefill_transition(
    step: &MlxSpeculativeStep,
    expected_position: usize,
    expected_input_token_id: i32,
    expected_held_token_id: i32,
) -> Result<()> {
    if step.position == expected_position
        && step.input_token_id == expected_input_token_id
        && step.output_token_ids.as_slice() == [expected_held_token_id]
    {
        Ok(())
    } else {
        Err(embedded_error(
            "native incremental prefill returned an inconsistent transition",
        ))
    }
}

impl LocalInferenceBackend for QwenEmbeddedBackend {
    fn complete(
        &mut self,
        request: LocalCompletionRequest,
        output: &mut GenerationOutput<'_>,
    ) -> Result<CompletionResponse> {
        let result = self.complete_inner(request, output);
        if result.is_err() {
            // Cancellation and parser/host failures leave the committed
            // checkpoint untouched. Drop any partially extended working KV
            // roots immediately so an interrupted turn becomes truly idle.
            let _ = self.session.reset();
        }
        result
    }
}

fn emit_all(
    output: &mut GenerationOutput<'_>,
    deltas: impl IntoIterator<Item = CompletionDelta>,
) -> Result<()> {
    for delta in deltas {
        output.emit(delta)?;
    }
    Ok(())
}

fn render_prompt(request: &LocalCompletionRequest) -> Result<String> {
    if request.messages.is_empty() {
        return Err(embedded_error(
            "chat completion requires at least one message",
        ));
    }

    let system = request.system.as_deref().unwrap_or_default().trim();
    let mut prompt = String::new();
    if request.tools.is_empty() {
        prompt.push_str(IM_START_TOKEN);
        prompt.push_str("system\n");
        prompt.push_str(XHIGH_REASONING_INSTRUCTION);
        if !system.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(system);
        }
        prompt.push_str(IM_END_TOKEN);
        prompt.push('\n');
    } else {
        prompt.push_str(IM_START_TOKEN);
        prompt.push_str("system\n");
        prompt.push_str(XHIGH_REASONING_INSTRUCTION);
        prompt.push_str("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &request.tools {
            validate_tag_identifier(&tool.name, "tool name")?;
            prompt.push('\n');
            prompt.push_str(&serde_json::to_string(&json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            }))?);
        }
        prompt.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if !system.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(system);
        }
        prompt.push_str(IM_END_TOKEN);
        prompt.push('\n');
    }

    for message in &request.messages {
        render_message(&mut prompt, message)?;
    }
    prompt.push_str(ASSISTANT_GENERATION_PREFIX);
    Ok(prompt)
}

/// Render the exact raw prompt consumed by the embedded Qwen backend.
///
/// This is intentionally exposed for cross-runtime differential benchmarks:
/// another runtime can consume the same bytes without applying its own chat
/// template, keeping prompt construction out of performance comparisons.
pub fn render_qwen_prompt(request: &LocalCompletionRequest) -> Result<String> {
    render_prompt(request)
}

fn render_message(prompt: &mut String, message: &Message) -> Result<()> {
    match message.role.as_str() {
        "assistant" => render_assistant_message(prompt, message),
        "user" => render_user_message(prompt, message),
        role => Err(embedded_error(format!(
            "unsupported conversation role `{role}`"
        ))),
    }
}

fn render_assistant_message(prompt: &mut String, message: &Message) -> Result<()> {
    let reasoning = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.trim()),
            _ => None,
        })
        .filter(|thinking| !thinking.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let text = message.text();
    let text = text.trim();

    prompt.push_str(IM_START_TOKEN);
    prompt.push_str("assistant\n");
    prompt.push_str(THINK_OPEN);
    prompt.push('\n');
    prompt.push_str(&reasoning);
    prompt.push('\n');
    prompt.push_str(THINK_CLOSE);
    prompt.push_str("\n\n");
    prompt.push_str(text);

    let mut first_tool = true;
    for block in &message.content {
        let ContentBlock::ToolUse { name, input, .. } = block else {
            continue;
        };
        validate_tag_identifier(name, "tool-call name")?;
        if !first_tool || !text.is_empty() {
            prompt.push_str("\n\n");
        }
        first_tool = false;
        prompt.push_str(TOOL_OPEN);
        prompt.push('\n');
        prompt.push_str("<function=");
        prompt.push_str(name);
        prompt.push_str(">\n");
        let arguments = input.as_object().ok_or_else(|| {
            embedded_error(format!(
                "replayed tool call `{name}` has non-object arguments"
            ))
        })?;
        for (argument_name, argument_value) in arguments {
            validate_tag_identifier(argument_name, "tool argument name")?;
            prompt.push_str("<parameter=");
            prompt.push_str(argument_name);
            prompt.push_str(">\n");
            if let Some(text) = argument_value.as_str() {
                prompt.push_str(text);
            } else {
                prompt.push_str(&serde_json::to_string(argument_value)?);
            }
            prompt.push_str("\n</parameter>\n");
        }
        prompt.push_str("</function>\n");
        prompt.push_str(TOOL_CLOSE);
    }
    prompt.push_str(IM_END_TOKEN);
    prompt.push('\n');
    Ok(())
}

fn render_user_message(prompt: &mut String, message: &Message) -> Result<()> {
    let tool_results = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let has_tool_results = !tool_results.is_empty();
    if has_tool_results {
        prompt.push_str(IM_START_TOKEN);
        prompt.push_str("user");
        for result in tool_results {
            prompt.push_str("\n<tool_response>\n");
            prompt.push_str(result);
            prompt.push_str("\n</tool_response>");
        }
        prompt.push_str(IM_END_TOKEN);
        prompt.push('\n');
    }
    let text = message.text();
    let text = text.trim();
    if !text.is_empty() {
        prompt.push_str(IM_START_TOKEN);
        prompt.push_str("user\n");
        prompt.push_str(text);
        prompt.push_str(IM_END_TOKEN);
        prompt.push('\n');
    }
    if !has_tool_results && text.is_empty() {
        return Err(embedded_error("user message has no renderable content"));
    }
    Ok(())
}

fn validate_tag_identifier(value: &str, description: &str) -> Result<()> {
    if value.is_empty()
        || value
            .chars()
            .any(|character| matches!(character, '>' | '<' | '\n' | '\r'))
    {
        Err(embedded_error(format!(
            "{description} `{value}` cannot be represented in Qwen's tool format"
        )))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParserState {
    Thinking,
    Content,
    Tool,
}

struct QwenOutputParser<'a> {
    tools: &'a [ToolDef],
    limits: CompletionLimits,
    state: ParserState,
    buffer: String,
    decoded_source: String,
    retained_source_bytes: usize,
    reasoning: String,
    text: String,
    tool_uses: Vec<ContentBlock>,
    allow_leading_think: bool,
    trim_leading_content: bool,
}

impl<'a> QwenOutputParser<'a> {
    fn new(tools: &'a [ToolDef], limits: CompletionLimits) -> Self {
        Self {
            tools,
            limits,
            state: ParserState::Thinking,
            buffer: String::new(),
            decoded_source: String::new(),
            retained_source_bytes: 0,
            reasoning: String::new(),
            text: String::new(),
            tool_uses: Vec::new(),
            allow_leading_think: true,
            trim_leading_content: false,
        }
    }

    fn decoded_source(&self) -> &str {
        &self.decoded_source
    }

    fn push(&mut self, fragment: &str) -> Result<Vec<CompletionDelta>> {
        if fragment.is_empty() {
            return Ok(Vec::new());
        }
        self.retained_source_bytes = self
            .limits
            .checked_response_bytes(self.retained_source_bytes, fragment.len())?;
        self.decoded_source.push_str(fragment);
        self.buffer.push_str(fragment);
        self.drain()
    }

    fn drain(&mut self) -> Result<Vec<CompletionDelta>> {
        let mut deltas = Vec::new();
        loop {
            match self.state {
                ParserState::Thinking => {
                    if self.allow_leading_think {
                        let leading = leading_whitespace_bytes(&self.buffer);
                        let candidate = &self.buffer[leading..];
                        if THINK_OPEN.starts_with(candidate) {
                            return Ok(deltas);
                        }
                        if candidate.starts_with(THINK_OPEN) {
                            let remaining = self.buffer[leading + THINK_OPEN.len()..]
                                .trim_start_matches(char::is_whitespace)
                                .to_string();
                            self.buffer = remaining;
                            self.allow_leading_think = false;
                            continue;
                        }
                        self.allow_leading_think = false;
                    }

                    let close = self.buffer.find(THINK_CLOSE);
                    let tool = self.buffer.find(TOOL_OPEN);
                    match earliest_delimiter(close, tool) {
                        Some((index, THINK_CLOSE)) => {
                            let reasoning = self.buffer[..index].to_string();
                            self.emit_reasoning(&reasoning, &mut deltas);
                            self.buffer = self.buffer[index + THINK_CLOSE.len()..]
                                .trim_start_matches(char::is_whitespace)
                                .to_string();
                            self.state = ParserState::Content;
                            self.trim_leading_content = true;
                        }
                        Some((index, TOOL_OPEN)) => {
                            let reasoning = self.buffer[..index].to_string();
                            self.emit_reasoning(&reasoning, &mut deltas);
                            self.buffer = self.buffer[index + TOOL_OPEN.len()..].to_string();
                            self.state = ParserState::Tool;
                        }
                        _ => {
                            let retained =
                                longest_delimiter_overlap(&self.buffer, &[THINK_CLOSE, TOOL_OPEN]);
                            let safe = self.buffer.len() - retained;
                            let reasoning = self.buffer[..safe].to_string();
                            self.emit_reasoning(&reasoning, &mut deltas);
                            self.buffer = self.buffer[safe..].to_string();
                            return Ok(deltas);
                        }
                    }
                }
                ParserState::Content => {
                    if self.trim_leading_content {
                        self.buffer = self
                            .buffer
                            .trim_start_matches(char::is_whitespace)
                            .to_string();
                        if self.buffer.is_empty() {
                            return Ok(deltas);
                        }
                        self.trim_leading_content = false;
                    }
                    if let Some(index) = self.buffer.find(TOOL_OPEN) {
                        let content = self.buffer[..index].trim_end().to_string();
                        self.emit_text(&content, &mut deltas);
                        self.buffer = self.buffer[index + TOOL_OPEN.len()..].to_string();
                        self.state = ParserState::Tool;
                        continue;
                    }
                    let overlap = longest_delimiter_overlap(&self.buffer, &[TOOL_OPEN]);
                    let before_overlap = self.buffer.len() - overlap;
                    let safe = trailing_whitespace_start(&self.buffer[..before_overlap]);
                    let content = self.buffer[..safe].to_string();
                    self.emit_text(&content, &mut deltas);
                    self.buffer = self.buffer[safe..].to_string();
                    return Ok(deltas);
                }
                ParserState::Tool => {
                    let Some(index) = self.buffer.find(TOOL_CLOSE) else {
                        return Ok(deltas);
                    };
                    if self.tool_uses.len() == self.limits.max_tool_uses {
                        return Err(Error::Limit(format!(
                            "provider completion exceeded the host limit of {} tool calls",
                            self.limits.max_tool_uses
                        )));
                    }
                    let raw = self.buffer[..index].to_string();
                    let tool_use = parse_tool_call(&raw, self.tools)?;
                    self.tool_uses.push(tool_use);
                    self.buffer = self.buffer[index + TOOL_CLOSE.len()..]
                        .trim_start_matches(char::is_whitespace)
                        .to_string();
                    self.state = ParserState::Content;
                    self.trim_leading_content = true;
                }
            }
        }
    }

    fn finish(mut self) -> Result<(Vec<CompletionDelta>, Vec<ContentBlock>)> {
        let mut deltas = self.drain()?;
        match self.state {
            ParserState::Thinking => {
                if self.allow_leading_think {
                    let leading = leading_whitespace_bytes(&self.buffer);
                    if self.buffer[leading..].starts_with(THINK_OPEN) {
                        self.buffer = self.buffer[leading + THINK_OPEN.len()..]
                            .trim_start_matches(char::is_whitespace)
                            .to_string();
                    }
                }
                let remainder = std::mem::take(&mut self.buffer);
                self.emit_reasoning(&remainder, &mut deltas);
            }
            ParserState::Content => {
                let remainder = self.buffer.trim_end().to_string();
                self.buffer.clear();
                self.emit_text(&remainder, &mut deltas);
            }
            ParserState::Tool => {
                // Never execute a truncated tool call. Preserve it as answer
                // text so MaxTokens remains visible and recoverable.
                let remainder = format!("{TOOL_OPEN}{}", self.buffer);
                self.buffer.clear();
                self.emit_text(&remainder, &mut deltas);
            }
        }

        let mut content = Vec::new();
        if !self.reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                thinking: self.reasoning,
                signature: String::new(),
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        content.extend(self.tool_uses);
        crate::provider::ensure_unique_tool_call_ids(&mut content);
        Ok((deltas, content))
    }

    fn emit_reasoning(&mut self, text: &str, deltas: &mut Vec<CompletionDelta>) {
        if !text.is_empty() {
            self.reasoning.push_str(text);
            deltas.push(CompletionDelta::Reasoning(text.to_string()));
        }
    }

    fn emit_text(&mut self, text: &str, deltas: &mut Vec<CompletionDelta>) {
        if !text.is_empty() {
            self.text.push_str(text);
            deltas.push(CompletionDelta::Text(text.to_string()));
        }
    }
}

fn parse_tool_call(raw: &str, tools: &[ToolDef]) -> Result<ContentBlock> {
    let raw = raw.trim();
    let function_prefix = "<function=";
    let after_prefix = raw
        .strip_prefix(function_prefix)
        .ok_or_else(|| embedded_error("Qwen tool call is missing its <function=name> block"))?;
    let function_name_end = after_prefix
        .find('>')
        .ok_or_else(|| embedded_error("Qwen tool call has an unterminated function tag"))?;
    let function_name = after_prefix[..function_name_end].trim();
    validate_tag_identifier(function_name, "generated tool-call name")?;
    let mut cursor = &after_prefix[function_name_end + 1..];
    let mut arguments = Map::new();

    loop {
        cursor = cursor.trim_start_matches(char::is_whitespace);
        if let Some(remaining) = cursor.strip_prefix("</function>") {
            if !remaining.trim().is_empty() {
                return Err(embedded_error(
                    "Qwen tool call contains content after </function>",
                ));
            }
            break;
        }
        let parameter_prefix = "<parameter=";
        let after_parameter = cursor.strip_prefix(parameter_prefix).ok_or_else(|| {
            embedded_error("Qwen tool call contains content outside a parameter block")
        })?;
        let parameter_name_end = after_parameter
            .find('>')
            .ok_or_else(|| embedded_error("Qwen tool call has an unterminated parameter tag"))?;
        let parameter_name = after_parameter[..parameter_name_end].trim();
        validate_tag_identifier(parameter_name, "generated tool argument name")?;
        let value_and_rest = &after_parameter[parameter_name_end + 1..];
        let value_end = value_and_rest.find("</parameter>").ok_or_else(|| {
            embedded_error(format!(
                "Qwen tool argument `{parameter_name}` is missing </parameter>"
            ))
        })?;
        let raw_value = trim_one_newline(&value_and_rest[..value_end]);
        let value = parse_typed_tool_value(
            raw_value,
            schema_types(tools, function_name, parameter_name),
        );
        arguments.insert(parameter_name.to_string(), value);
        cursor = &value_and_rest[value_end + "</parameter>".len()..];
    }

    Ok(ContentBlock::ToolUse {
        name: function_name.to_string(),
        input: Value::Object(arguments),
        id: format!("call_{}", Uuid::new_v4().simple()),
    })
}

fn schema_types(tools: &[ToolDef], function_name: &str, parameter_name: &str) -> BTreeSet<String> {
    let Some(schema) = tools
        .iter()
        .find(|tool| tool.name == function_name)
        .and_then(|tool| tool.input_schema.get("properties"))
        .and_then(|properties| properties.get(parameter_name))
    else {
        return BTreeSet::new();
    };
    let mut types = BTreeSet::new();
    collect_schema_types(schema, &mut types);
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        for option in any_of {
            collect_schema_types(option, &mut types);
        }
    }
    types
}

fn collect_schema_types(schema: &Value, types: &mut BTreeSet<String>) {
    match schema.get("type") {
        Some(Value::String(kind)) => {
            types.insert(kind.clone());
        }
        Some(Value::Array(kinds)) => {
            for kind in kinds.iter().filter_map(Value::as_str) {
                types.insert(kind.to_string());
            }
        }
        _ => {}
    }
}

fn parse_typed_tool_value(raw: &str, types: BTreeSet<String>) -> Value {
    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    if types.is_empty() {
        return Value::String(raw.to_string());
    }
    if types.contains("boolean") {
        if raw.eq_ignore_ascii_case("true") {
            return Value::Bool(true);
        }
        if raw.eq_ignore_ascii_case("false") || types.len() == 1 {
            return Value::Bool(false);
        }
    }
    if types.contains("integer") {
        if let Ok(integer) = raw.parse::<i64>() {
            return Value::Number(Number::from(integer));
        }
        if let Ok(integer) = raw.parse::<u64>() {
            return Value::Number(Number::from(integer));
        }
        if types.len() == 1 {
            return Value::String(raw.to_string());
        }
    }
    if types.contains("number") {
        if let Ok(number) = raw.parse::<f64>() {
            if let Some(number) = Number::from_f64(number) {
                return Value::Number(number);
            }
        }
        if types.len() == 1 {
            return Value::String(raw.to_string());
        }
    }
    if types.contains("array") {
        if let Ok(value @ Value::Array(_)) = serde_json::from_str(raw) {
            return value;
        }
        if types.len() == 1 {
            return Value::String(raw.to_string());
        }
    }
    if types.contains("object") {
        if let Ok(value @ Value::Object(_)) = serde_json::from_str(raw) {
            return value;
        }
        if types.len() == 1 {
            return Value::String(raw.to_string());
        }
    }
    Value::String(raw.to_string())
}

fn trim_one_newline(mut value: &str) -> &str {
    value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

fn earliest_delimiter(
    first: Option<usize>,
    second: Option<usize>,
) -> Option<(usize, &'static str)> {
    match (first, second) {
        (Some(first), Some(second)) if first <= second => Some((first, THINK_CLOSE)),
        (Some(_), Some(second)) => Some((second, TOOL_OPEN)),
        (Some(first), None) => Some((first, THINK_CLOSE)),
        (None, Some(second)) => Some((second, TOOL_OPEN)),
        (None, None) => None,
    }
}

fn longest_delimiter_overlap(text: &str, delimiters: &[&str]) -> usize {
    delimiters
        .iter()
        .map(|delimiter| {
            (1..delimiter.len())
                .rev()
                .find(|length| text.ends_with(&delimiter[..*length]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

fn leading_whitespace_bytes(text: &str) -> usize {
    text.char_indices()
        .find_map(|(index, character)| (!character.is_whitespace()).then_some(index))
        .unwrap_or(text.len())
}

fn trailing_whitespace_start(text: &str) -> usize {
    text.char_indices()
        .rev()
        .find_map(|(index, character)| {
            (!character.is_whitespace()).then_some(index + character.len_utf8())
        })
        .unwrap_or(0)
}

#[cfg(generalist_mlx_native_linked)]
fn validate_model_name(model_name: &str) -> Result<()> {
    if model_name == DEFAULT_EMBEDDED_MODEL {
        Ok(())
    } else {
        Err(embedded_error(format!(
            "unsupported model `{model_name}`; this engine is specialized for dense {DEFAULT_EMBEDDED_MODEL}"
        )))
    }
}

#[cfg(generalist_mlx_native_linked)]
fn resolve_model_pack_paths() -> Result<(PathBuf, PathBuf)> {
    let manifest_override = env::var_os("GENERALIST_EMBEDDED_MANIFEST");
    let blob_override = env::var_os("GENERALIST_EMBEDDED_BLOB_ROOT");
    match (manifest_override, blob_override) {
        (Some(manifest), Some(blobs)) => Ok((manifest.into(), blobs.into())),
        (Some(_), None) | (None, Some(_)) => Err(embedded_error(
            "GENERALIST_EMBEDDED_MANIFEST and GENERALIST_EMBEDDED_BLOB_ROOT must be set together",
        )),
        (None, None) => {
            let model_root = match env::var_os("OLLAMA_MODELS") {
                Some(root) => PathBuf::from(root),
                None => {
                    let home = env::var_os("HOME").ok_or_else(|| {
                        embedded_error(
                            "HOME is unavailable; set OLLAMA_MODELS or both embedded model-pack overrides",
                        )
                    })?;
                    PathBuf::from(home).join(".ollama/models")
                }
            };
            Ok((
                model_root
                    .join("manifests/registry.ollama.ai/library")
                    .join(MODEL_NAME)
                    .join(MODEL_TAG),
                model_root.join("blobs"),
            ))
        }
    }
}

#[cfg(generalist_mlx_native_linked)]
fn validate_model_config(pack: &ModelPack) -> Result<()> {
    let config = pack.read_json_document("config.json")?;
    expect_json(
        &config,
        "/architectures/0",
        json!("Qwen3_5ForConditionalGeneration"),
    )?;
    expect_json(&config, "/model_type", json!("qwen3_5"))?;
    for (pointer, expected) in [
        ("/text_config/model_type", json!("qwen3_5_text")),
        ("/text_config/hidden_size", json!(5120)),
        ("/text_config/intermediate_size", json!(17408)),
        ("/text_config/num_hidden_layers", json!(64)),
        ("/text_config/num_attention_heads", json!(24)),
        ("/text_config/num_key_value_heads", json!(4)),
        ("/text_config/head_dim", json!(256)),
        ("/text_config/full_attention_interval", json!(4)),
        ("/text_config/linear_num_key_heads", json!(16)),
        ("/text_config/linear_num_value_heads", json!(48)),
        ("/text_config/linear_key_head_dim", json!(128)),
        ("/text_config/linear_value_head_dim", json!(128)),
        ("/text_config/vocab_size", json!(248320)),
        (
            "/text_config/max_position_embeddings",
            json!(CONTEXT_TOKENS),
        ),
    ] {
        expect_json(&config, pointer, expected)?;
    }
    Ok(())
}

#[cfg(generalist_mlx_native_linked)]
fn expect_json(config: &Value, pointer: &str, expected: Value) -> Result<()> {
    let observed = config.pointer(pointer).ok_or_else(|| {
        embedded_error(format!(
            "model config is missing required field `{pointer}`"
        ))
    })?;
    if observed == &expected {
        Ok(())
    } else {
        Err(embedded_error(format!(
            "model config `{pointer}` is {observed}, expected {expected}"
        )))
    }
}

#[cfg(generalist_mlx_native_linked)]
fn validate_tokenizer(tokenizer: &Tokenizer) -> Result<()> {
    for (token, expected) in [
        (END_OF_TEXT_TOKEN, 248_044),
        (IM_START_TOKEN, 248_045),
        (IM_END_TOKEN, 248_046),
    ] {
        let observed = required_token_id(tokenizer, token)?;
        if observed != expected {
            return Err(embedded_error(format!(
                "tokenizer maps `{token}` to {observed}, expected {expected}"
            )));
        }
    }
    let generation_prefix = tokenizer
        .encode("<|im_start|>assistant\n<think>\n", false)
        .map_err(|error| {
            embedded_error(format!(
                "failed to validate tokenizer generation prefix: {error}"
            ))
        })?;
    if generation_prefix.get_ids() != [248_045, 74_455, 198, 248_068, 198] {
        return Err(embedded_error(format!(
            "tokenizer generation prefix is {:?}, expected the Qwen3.8 prefix",
            generation_prefix.get_ids()
        )));
    }
    Ok(())
}

#[cfg(generalist_mlx_native_linked)]
fn required_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| embedded_error(format!("tokenizer is missing required token `{token}`")))
}

#[cfg(generalist_mlx_native_linked)]
fn materialize_embedded_metallib() -> Result<PathBuf> {
    let digest = ring::digest::digest(&SHA256, EMBEDDED_METALLIB);
    let hex = hex_bytes(digest.as_ref());
    let directory = env::temp_dir().join("generalist-embedded-mlx");
    fs::create_dir_all(&directory).map_err(|error| {
        embedded_error(format!(
            "failed to create Metal-library cache {}: {error}",
            directory.display()
        ))
    })?;
    let destination = directory.join(format!("mlx-{hex}.metallib"));
    if destination.exists() {
        verify_file_digest(&destination, digest.as_ref())?;
        return Ok(destination);
    }

    let mut temporary = tempfile::NamedTempFile::new_in(&directory).map_err(|error| {
        embedded_error(format!(
            "failed to create temporary Metal library in {}: {error}",
            directory.display()
        ))
    })?;
    temporary.write_all(EMBEDDED_METALLIB).map_err(|error| {
        embedded_error(format!("failed to write embedded Metal library: {error}"))
    })?;
    temporary.as_file().sync_all().map_err(|error| {
        embedded_error(format!("failed to flush embedded Metal library: {error}"))
    })?;
    match temporary.persist_noclobber(&destination) {
        Ok(_) => {}
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_file_digest(&destination, digest.as_ref())?;
        }
        Err(error) => {
            return Err(embedded_error(format!(
                "failed to install embedded Metal library {}: {}",
                destination.display(),
                error.error
            )))
        }
    }
    Ok(destination)
}

#[cfg(generalist_mlx_native_linked)]
fn verify_file_digest(path: &Path, expected: &[u8]) -> Result<()> {
    let mut file = File::open(path).map_err(|error| {
        embedded_error(format!(
            "failed to open cached Metal library {}: {error}",
            path.display()
        ))
    })?;
    let mut context = Context::new(&SHA256);
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            embedded_error(format!(
                "failed to hash cached Metal library {}: {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }
    if context.finish().as_ref() == expected {
        Ok(())
    } else {
        Err(embedded_error(format!(
            "cached Metal library {} does not match the executable; remove it and retry",
            path.display()
        )))
    }
}

#[cfg(generalist_mlx_native_linked)]
fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn cancelled() -> Error {
    embedded_error("inference was cancelled")
}

fn embedded_error(message: impl Display) -> Error {
    Error::Other(format!("Embedded Qwen error: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<Message>, tools: Vec<ToolDef>) -> LocalCompletionRequest {
        LocalCompletionRequest {
            system: Some("Be precise.".to_string()),
            messages,
            tools,
            max_tokens: Some(128),
            limits: CompletionLimits::default(),
        }
    }

    fn weather_tool() -> ToolDef {
        ToolDef {
            name: "weather".to_string(),
            description: "Read weather".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "filters": {"type": "array"}
                },
                "required": ["city"]
            }),
        }
    }

    #[test]
    fn draft_vocab_prompt_extras_are_recent_sorted_unique_high_ids() {
        assert_eq!(
            prompt_draft_extra_token_ids(&[70_000, 42, 80_000, 70_000, 248_000, 65_535]),
            vec![70_000, 80_000]
        );

        let high_ids =
            (65_536..65_536 + MTP_DRAFT_EXTRA_TOKEN_LIMIT as i32 + 1).collect::<Vec<_>>();
        let selected = prompt_draft_extra_token_ids(&high_ids);
        assert_eq!(selected.len(), MTP_DRAFT_EXTRA_TOKEN_LIMIT);
        assert_eq!(selected[0], 65_537);
        assert_eq!(selected.last(), Some(&65_792));
    }

    #[test]
    fn draft_vocab_output_fallback_requires_sustained_uncovered_high_ids() {
        let sparse = VecDeque::from([true, true, false, false, false, false, false, false]);
        assert!(!output_window_prefers_full_draft_vocab(&sparse));

        let dense = VecDeque::from([true, true, true, false, false, false, false, false]);
        assert!(output_window_prefers_full_draft_vocab(&dense));
        assert_eq!(draft_vocab_observation(65_535, &[]), Some(false));
        assert_eq!(draft_vocab_observation(65_536, &[]), Some(true));
        assert_eq!(draft_vocab_observation(65_536, &[65_536]), Some(false));
        assert_eq!(draft_vocab_observation(248_000, &[]), None);
    }

    #[test]
    fn adaptive_mtp_warms_each_shape_with_two_consecutive_cycles() {
        let mut controller = AdaptiveMtpController::default();
        for depth in MTP_WARMUP_DEPTHS {
            assert_eq!(controller.next_depth(), depth);
            let elapsed = match depth {
                2 => 0.10,
                3 => 0.12,
                4 => 0.16,
                _ => unreachable!(),
            };
            controller.observe(depth, depth, depth, elapsed);
        }

        assert_eq!(
            controller.cost_seconds,
            [Some(0.10), Some(0.12), Some(0.16)]
        );
        assert_eq!(controller.next_depth(), 3);
    }

    #[test]
    fn adaptive_mtp_uses_depth_three_when_the_third_draft_pays() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            cost_seconds: [Some(0.10), Some(0.12), Some(0.20)],
            ..AdaptiveMtpController::default()
        };
        controller.acceptance = [
            AcceptanceEstimate {
                probability: 0.95,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
        ];

        assert_eq!(controller.next_depth(), 3);
    }

    #[test]
    fn adaptive_mtp_uses_depth_four_when_the_fourth_draft_pays() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            cost_seconds: [Some(0.10), Some(0.12), Some(0.135)],
            ..AdaptiveMtpController::default()
        };
        controller.acceptance = [
            AcceptanceEstimate {
                probability: 0.95,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
        ];

        assert_eq!(controller.next_depth(), 4);
    }

    #[test]
    fn adaptive_mtp_does_not_probe_below_the_selected_maximum() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            cost_seconds: [Some(0.10), Some(0.12), Some(0.135)],
            ..AdaptiveMtpController::default()
        };
        controller.acceptance = [
            AcceptanceEstimate {
                probability: 0.95,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
        ];

        for _ in 0..(MTP_PROBE_INTERVAL_BASE + 4) {
            assert_eq!(controller.next_depth(), 4);
            controller.observe(4, 4, 4, 0.135);
        }
        assert_eq!(controller.probe_depth, None);
    }

    #[test]
    fn adaptive_mtp_probes_only_one_deeper_and_backs_off() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            cost_seconds: [Some(0.10), Some(0.20), Some(0.30)],
            ..AdaptiveMtpController::default()
        };
        controller.acceptance = [
            AcceptanceEstimate {
                probability: 0.95,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.10,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.10,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
        ];

        for _ in 0..MTP_PROBE_INTERVAL_BASE {
            assert_eq!(controller.next_depth(), 2);
            controller.observe(2, 2, 2, 0.10);
        }
        for _ in 0..MTP_PROBE_CYCLES {
            assert_eq!(controller.next_depth(), 3);
            controller.observe(3, 3, 3, 0.20);
        }
        assert_eq!(controller.selected_depth, 2);
        assert_eq!(controller.probe_interval, 2 * MTP_PROBE_INTERVAL_BASE);

        for _ in 0..(2 * MTP_PROBE_INTERVAL_BASE) {
            assert_eq!(controller.next_depth(), 2);
            controller.observe(2, 2, 2, 0.10);
        }
        assert_eq!(controller.next_depth(), 3);
    }

    #[test]
    fn adaptive_mtp_uses_depth_two_when_the_third_draft_does_not_pay() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            cost_seconds: [Some(0.10), Some(0.11), Some(0.13)],
            ..AdaptiveMtpController::default()
        };
        controller.acceptance = [
            AcceptanceEstimate {
                probability: 0.95,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.90,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.10,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
            AcceptanceEstimate {
                probability: 0.10,
                reached: MTP_MIN_ACCEPTANCE_SAMPLES,
            },
        ];

        assert_eq!(controller.next_depth(), 2);
    }

    #[test]
    fn adaptive_mtp_updates_only_reached_acceptance_positions() {
        let mut controller = AdaptiveMtpController {
            warmup_index: MTP_WARMUP_DEPTHS.len(),
            ..AdaptiveMtpController::default()
        };
        controller.observe(4, 4, 0, 0.1);
        assert_eq!(controller.acceptance[0].reached, 1);
        assert_eq!(controller.acceptance[1].reached, 0);
        assert_eq!(controller.acceptance[2].reached, 0);
        assert_eq!(controller.acceptance[3].reached, 0);

        controller.observe(4, 4, 3, 0.1);
        assert_eq!(controller.acceptance[0].reached, 2);
        assert_eq!(controller.acceptance[1].reached, 1);
        assert_eq!(controller.acceptance[2].reached, 1);
        assert_eq!(controller.acceptance[3].reached, 1);
    }

    #[test]
    fn prefix_cache_accepts_only_the_exact_held_continuation() {
        assert_eq!(reusable_prefix_len(&[10, 20], 30, &[10, 20, 30]), Some(2));
        assert_eq!(
            reusable_prefix_len(&[10, 20], 30, &[10, 20, 30, 40]),
            Some(2)
        );
        assert_eq!(reusable_prefix_len(&[10, 20], 30, &[10, 20]), None);
        assert_eq!(reusable_prefix_len(&[10, 20], 30, &[10, 21, 30]), None);
        assert_eq!(reusable_prefix_len(&[10, 20], 30, &[10, 20, 31]), None);
    }

    #[test]
    fn prefix_cache_selects_the_longest_exact_branch() {
        let short = [10, 20];
        let long = [10, 20, 30, 40];
        let checkpoints = [(short.as_slice(), 30), (long.as_slice(), 50)];
        assert_eq!(
            best_reusable_prefix_len(checkpoints, &[10, 20, 30, 40, 50, 60]),
            Some(4)
        );
        assert_eq!(
            best_reusable_prefix_len(checkpoints, &[10, 20, 30, 99]),
            Some(2)
        );
        assert_eq!(best_reusable_prefix_len(checkpoints, &[10, 99, 30]), None);
    }

    #[test]
    fn response_checkpoint_promotion_requires_the_complete_raw_prefix() {
        assert!(normalized_response_starts_with_raw_tokens(
            &[10, 20, 30, 40, 50],
            &[10, 20],
            &[30, 40]
        ));
        assert!(!normalized_response_starts_with_raw_tokens(
            &[10, 21, 30, 40, 50],
            &[10, 20],
            &[30, 40]
        ));
        assert!(!normalized_response_starts_with_raw_tokens(
            &[10, 20, 30, 41, 50],
            &[10, 20],
            &[30, 40]
        ));
        assert!(!normalized_response_starts_with_raw_tokens(
            &[10, 20, 30],
            &[10, 20],
            &[30, 40]
        ));
    }

    #[test]
    fn completed_turn_and_tool_loop_extend_the_prior_rendered_prompt() {
        let initial_messages = vec![Message::user_text("Weather?")];
        let initial =
            render_prompt(&request(initial_messages.clone(), vec![weather_tool()])).unwrap();
        let assistant = Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "Need weather.".to_string(),
                signature: String::new(),
            },
            ContentBlock::ToolUse {
                name: "weather".to_string(),
                input: json!({"city": "Boston"}),
                id: "call_1".to_string(),
            },
        ]);
        let tool_result = Message::user(vec![ContentBlock::ToolResult {
            content: "Sunny".to_string(),
            tool_use_id: "call_1".to_string(),
            is_error: None,
        }]);
        let tool_loop = render_prompt(&request(
            vec![initial_messages[0].clone(), assistant.clone(), tool_result],
            vec![weather_tool()],
        ))
        .unwrap();
        assert!(tool_loop.starts_with(&initial));

        let next_turn = render_prompt(&request(
            vec![
                initial_messages[0].clone(),
                assistant,
                Message::user_text("And tomorrow?"),
            ],
            vec![weather_tool()],
        ))
        .unwrap();
        assert!(next_turn.starts_with(&initial));
    }

    #[test]
    fn prompt_keeps_xhigh_reasoning_and_opens_thinking() {
        let rendered = render_prompt(&request(
            vec![Message::user_text("What is 2 + 2?")],
            Vec::new(),
        ))
        .unwrap();

        assert!(rendered.starts_with(&format!(
            "{IM_START_TOKEN}system\n{XHIGH_REASONING_INSTRUCTION}\n\nBe precise.{IM_END_TOKEN}\n"
        )));
        assert!(rendered.contains(&format!(
            "{IM_START_TOKEN}user\nWhat is 2 + 2?{IM_END_TOKEN}\n"
        )));
        assert!(rendered.ends_with(&format!("{IM_START_TOKEN}assistant\n{THINK_OPEN}\n")));
    }

    #[test]
    fn prompt_replays_reasoning_tools_and_grouped_results() {
        let assistant = Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "Need weather.".to_string(),
                signature: String::new(),
            },
            ContentBlock::ToolUse {
                name: "weather".to_string(),
                input: json!({"city": "Boston", "days": 2}),
                id: "call_1".to_string(),
            },
        ]);
        let results = Message::user(vec![ContentBlock::ToolResult {
            content: "Sunny".to_string(),
            tool_use_id: "call_1".to_string(),
            is_error: None,
        }]);
        let rendered = render_prompt(&request(
            vec![Message::user_text("Weather?"), assistant, results],
            vec![weather_tool()],
        ))
        .unwrap();

        assert!(rendered.contains("# Tools"));
        assert!(rendered.contains("<think>\nNeed weather.\n</think>"));
        assert!(rendered.contains("<function=weather>"));
        assert!(rendered.contains("<parameter=city>\nBoston\n</parameter>"));
        assert!(rendered.contains(&format!(
            "{IM_START_TOKEN}user\n<tool_response>\nSunny\n</tool_response>{IM_END_TOKEN}\n"
        )));
    }

    #[test]
    fn split_stream_extracts_reasoning_text_and_typed_tool_call() {
        let tool = weather_tool();
        let tools = [tool];
        let source = "work carefully</think>\n\nChecking.\n<tool_call>\n<function=weather>\n<parameter=city>\nBoston\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=filters>\n[\"rain\"]\n</parameter>\n</function>\n</tool_call>";
        let mut parser = QwenOutputParser::new(&tools, CompletionLimits::default());
        let mut deltas = Vec::new();
        for character in source.chars() {
            deltas.extend(parser.push(&character.to_string()).unwrap());
        }
        let (tail, content) = parser.finish().unwrap();
        deltas.extend(tail);

        assert_eq!(
            content[0],
            ContentBlock::Thinking {
                thinking: "work carefully".to_string(),
                signature: String::new()
            }
        );
        assert_eq!(
            content[1],
            ContentBlock::Text {
                text: "Checking.".to_string()
            }
        );
        let ContentBlock::ToolUse { name, input, id } = &content[2] else {
            panic!("expected tool use")
        };
        assert_eq!(name, "weather");
        assert!(id.starts_with("call_"));
        assert_eq!(input["city"], json!("Boston"));
        assert_eq!(input["days"], json!(3));
        assert_eq!(input["metric"], json!(true));
        assert_eq!(input["filters"], json!(["rain"]));
        assert!(deltas
            .iter()
            .any(|delta| matches!(delta, CompletionDelta::Reasoning(_))));
        assert!(deltas
            .iter()
            .any(|delta| matches!(delta, CompletionDelta::Text(_))));
    }

    #[test]
    fn truncated_tool_markup_is_never_executable() {
        let tools = [weather_tool()];
        let mut parser = QwenOutputParser::new(&tools, CompletionLimits::default());
        parser
            .push("reason</think><tool_call><function=weather><parameter=city>Boston")
            .unwrap();
        let (_, content) = parser.finish().unwrap();

        assert!(!content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. })));
        assert!(content.iter().any(|block| matches!(
            block,
            ContentBlock::Text { text } if text.starts_with("<tool_call>")
        )));
    }

    #[test]
    fn parser_enforces_decoded_payload_limit_before_growth() {
        let limits = CompletionLimits {
            max_response_bytes: 4,
            ..CompletionLimits::default()
        };
        let mut parser = QwenOutputParser::new(&[], limits);
        assert!(parser.push("1234").is_ok());
        assert!(matches!(parser.push("5"), Err(Error::Limit(_))));
    }
}
