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
