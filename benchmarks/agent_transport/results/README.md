# Local baseline — 2026-08-04

These files are retained as append-only evidence from
`qwen3.6:35b-a3b` through Ollama's OpenAI-compatible endpoint. No generated
code was executed. They predate Generalist's preloaded bridge, so omission of
the then-required `import tools` is correctly classified as a historical
failure rather than retroactively rewritten as a pass.

## Controlled result

The controlled runs used `reasoning.effort=none`, a 1,200-token output limit,
and one attempt per task. The first full corpus was split across
`20260804T224826Z-...jsonl` and `20260804T224917Z-...jsonl`; the stricter
checker derived `20260804T225500Z-local-controlled-rechecked.jsonl` from those
immutable responses. `20260804T225320Z-...jsonl` is an independent full-corpus
replication produced directly with the stricter checker.

| Transport | Exact-call passes | Rate | Median latency | Mean input tokens | Median model-payload overhead |
|---|---:|---:|---:|---:|---:|
| JSON function `{code: string}` | 8/20 | 40% | 2,052.5 ms | 630.5 | 31.5 bytes |
| Raw assistant text control | 12/20 | 60% | 1,384.1 ms | 261.1 | 0 bytes |

Failure totals across the two repetitions were:

- JSON function: five argument mismatches, four missing/ignored tool calls,
  and three scripts that omitted `import tools`;
- raw text: five argument mismatches, two syntax errors, and one script that
  invented its own bridge instead of calling `tools`.

This is a small paired engineering benchmark, not a general model-quality
claim. Raw assistant text is only a generation control and is not an adequate
production capability boundary. The input-token and latency differences also
include the function schema and bridge documentation supplied to the JSON
route, so they do not isolate serialization alone. Still, the repeated 40%
versus 60% result and the quote-heavy failures justify testing a provider-native
freeform tool rather than assuming the current wrapper is harmless.

One `responses_custom` capability probe is retained in
`20260804T225028Z-...jsonl`. Ollama returned HTTP 200 but downgraded the custom
tool to a JSON `function_call` with the wrong argument shape. The current local
route therefore cannot be switched to freeform input; it needs the JSON
fallback.

## Diagnostic default-thinking run

`20260804T224213Z-...jsonl` was the initial smoke run with default model
thinking. Seven of ten attempts exhausted the 1,200-token output budget before
emitting code. The original runner labeled those as extraction errors;
`20260804T225501Z-local-default-thinking-rechecked.jsonl` preserves a derived
classification that identifies them as `output_limit`. This run motivated the
controlled reasoning setting and should not be pooled with the result above.

## Integrity

The controlled derived file records SHA-256 hashes for all three source JSONL
files, the corpus, runner, and rechecker. Its corpus hash is
`aa65127334954d65b9c0ccbbbac50becc4dc97b34e9443eed8781e03986b4225`.
The original files remain authoritative for provider responses; derived files
exist only to apply a newer checker without rewriting history.

## Unpinned post-change sample

`20260804T230710Z-...jsonl` and `20260804T230836Z-...jsonl` were generated
after bridge preloading and compact signatures but before the harness pinned
temperature and seed. Their JSON repetitions varied from 5/10 to 9/10. They
are retained as evidence of that protocol flaw and must not be presented as a
deterministic post-change score.

## Pinned post-change sample

`20260804T231154Z-...jsonl` contains one paired run at `temperature=0`,
`seed=1`; `20260804T231243Z-...jsonl` contains two more JSON repetitions with
the same controls. JSON passed 21/30 (70%) and raw text passed 8/10 (80%). All
three JSON repetitions produced the identical ordered task/classification/code
digest
`16443d8fa3e9d8a4145eecd5ad78fd1ea727fe02e786aa82409b797c827005ea`,
confirming that this Ollama route honored the pinned controls for this corpus.

## Pinned legacy transport control

`20260804T231555Z-...jsonl` directly compares the preloaded, compact-signature
JSON transport with a control that requires `import tools` and supplies the
verbose raw JSON schema. Both passed 7/10 at `temperature=0`, `seed=1`, so this
run provides no evidence of an exact-call pass-rate improvement. The compact
transport had a 2,161.3 ms median latency versus 2,792.2 ms for the legacy
control, and 22 versus 38 median bytes of model-payload overhead. Those timing
figures come from one alternated local run and may include warm-cache or order
effects; they are engineering observations, not a model-level performance
claim.

The compact route failed with two argument mismatches and one ignored tool
call. The legacy route ignored the tool call three times. This distinction is
still operationally useful: bridge preloading removes a real missing-import
runtime failure mode, while the benchmark shows that generation correctness
remains a separate problem.

## OpenRouter remote baseline

The one-call `responses_custom` probes in `20260804T232940Z-...jsonl` and
`20260804T232956Z-...jsonl` were rejected by OpenRouter with HTTP 400
`invalid_prompt` for Kimi K3 and Qwen 3.8 Max, respectively. Neither request
reached code extraction, so freeform custom tools are not a viable route for
these model/provider combinations.

`20260804T233101Z-...jsonl` is the pinned ten-task Kimi K3 matrix. Current JSON
and legacy JSON each passed 9/10; raw text passed 10/10. Current JSON exhausted
the 1,200-token output limit on `unified_diff`, while legacy JSON changed one
`regex_and_sql` argument. Provider-reported successful-call costs were
$0.058977, $0.057016, and $0.068213 for current JSON, legacy JSON, and raw text.

`20260804T233059Z-...jsonl` is the corresponding Qwen 3.8 Max matrix. Raw text
passed 10/10 at a provider-reported cost of $0.035896. Both JSON profiles were
rejected on all ten tasks because this benchmark forced the Python function
with an object-valued `tool_choice`, which Alibaba does not support in thinking
mode. This is an API-profile incompatibility, not a 0/10 code-generation
result: Generalist's OpenAI-compatible adapter omits `tool_choice`, and an
unforced JSON control is required before drawing a production conclusion.

## OpenRouter implicit-tool control

`20260804T234315Z-...jsonl` resolves the Qwen API-profile question using the
production-like `json_tool_implicit` transport, which omits `tool_choice` and
`parallel_tool_calls`. Qwen 3.8 Max passed 10/10 at a provider-reported cost of
$0.039186. The earlier forced-profile HTTP errors therefore do not apply to
Generalist's current request shape.

`20260804T234322Z-...jsonl` applies the same control to Kimi K3. It passed 7/10
at a provider-reported cost of $0.048652. On `argv_no_shell` and
`delimiter_collision`, Kimi returned text instead of calling Python; on
`batch_mixed_payloads`, it changed the second call's arguments. The forced
Kimi profile passed 9/10 in its first sample, suggesting that explicit tool
selection may improve call propensity, but one repetition per profile is not
enough to estimate that effect.

`20260804T234714Z-...jsonl` adds two Qwen implicit-tool repetitions. It passed
18/20, with output-limit failures on `unified_diff` and
`delimiter_collision`. Across all three complete Qwen repetitions, implicit
JSON passed 28/30 (93.3%).

`20260804T234712Z-...jsonl` planned two additional Kimi repetitions of forced
and implicit JSON, but was deliberately interrupted after 25/40 attempts when
provider latency became extreme. Its first repetition is complete: forced JSON
passed 6/10 with four output limits, while implicit JSON passed 8/10 with one
ignored tool call and one argument mismatch. Five completed attempts from the
second repetition all passed but are not treated as a full replicate. Combining
only complete repetitions with the earlier baseline gives 15/20 for forced
JSON and 15/20 for implicit JSON. The data therefore do not support adding a
Kimi-specific forced-tool policy.
