# Validation record

Validated on 2026-09-06 03:33 UTC.

Baseline commit: `4cb8df7bc5d0b27d75c1bc6170ab769dfa2b7306`. The changes are local and uncommitted.
[source-hashes.sha256](source-hashes.sha256) records SHA-256 for all 293
tracked or visible untracked repository files outside `whitepaper/`, including
the newly shared TLC helper. Ignored build/runtime files are excluded. This
identifies the checked working contents, not just the baseline commit.

From the repository root, verify the source snapshot with:

```sh
shasum -a 256 -c whitepaper/evidence/source-hashes.sha256
```

## Commands and results

`GENERALIST_TLA_ACK=1 make check` completed with exit status 0. It includes
formatting, strict Clippy, Rust documentation generation, ShellCheck, profile
and traceability lints, research/benchmark and hook checks, all three exhaustive
TLC runs, observed traces, negative checks, and Rust tests including doc tests.

| Check | Result |
| --- | --- |
| Library tests | 318 passed; 2 explicitly ignored live catalog checks |
| Binary tests | 36 passed |
| OpenAI-compatible integration tests | 2 passed |
| Example tests | 6 passed |
| Documentation examples | 2 passed |
| Observed trace wrappers | 16 accepted |
| Invalid trace mutations | 25 rejected by a temporal counterexample |
| Model faults | 3 rejected by the required named invariant |

| Model | Generated states | Distinct states | Depth |
| --- | ---: | ---: | ---: |
| AsyncRuntime | 306655 | 65475 | 26 |
| MemoryRuntime | 7627 | 1690 | 16 |
| ArchiveScopeRuntime | 519516 | 59725 | 13 |

The three checked configurations contain respectively:

- Async: 3 prompt IDs, 2 request IDs, 2 tools, 2 rounds, 1 recoverable claim failure.
- Memory: 2 episode IDs, 3 scopes, one immutable current project.
- Archive: 3 scopes, 2 history IDs, 2 memory IDs.

Observed wrappers derive their identity domains from each recorded scenario;
all current async scenarios set the same two-round bound in Rust and TLC.
The base state-invariant lists are inherited, while finite trace completion
replaces base-specification temporal properties.

## Environment and limits

The run used macOS 26.5.2, Rust `1.99.0-nightly (d0babd8b6 2026-07-15)`, and
TLC 2.19 (8 August 2024), via the locally installed AdoptOpenJDK 14 runtime.
The TLC jar matches the repository's pinned v1.7.4 distribution checksum:
`936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`.
TLC and Python bridge tests required permitted local sockets; the initial
restricted-sandbox TLC launch failed at its RMI listener and was rerun with
that access. No provider credentials were needed for this gate.

The two ignored tests query live Cerebras/OpenRouter catalogs. This local run
does not execute those checks, a native MLX model, a Linux kernel sandbox, or a
new terminal PTY smoke test. The all-feature build explicitly uses the
unavailable native-runtime stub. Model results are finite checks; traces are
sampled and predominantly action-based, with concrete queue claim IDs and Rust
checkpoint assertions. No composed refinement proof, disk-crash proof,
external-effect rollback, or task-quality result follows from these counts.

## Regression evidence

The expanded pre-fix conformance run accepted the old happy paths but rejected
`AsyncRuntimeDeniedBatch`, `AsyncRuntimeDeniedPython`,
`AsyncRuntimeNestedDenial`, and `AsyncRuntimeUndeclaredTool`, each because the
observed trace could not complete. Their ordinary Rust outcomes remained
correct; this exposed model/instrumentation defects.

`cancellation_at_truncated_batch_boundary_preserves_steering` first failed
because its queued steer had already been consumed; it passes with the added
cancellation check. Context-measurement invalidation and saved-name redaction
have separate deterministic regressions. These distinctions are preserved in
the paper rather than counting every instrumentation correction as a runtime
bug discovered by TLA+.

## Accepted and rejecting cases

```text
PASS observed implementation trace: AsyncRuntimeToolAnswer
PASS observed implementation trace: AsyncRuntimeRefusal
PASS observed implementation trace: AsyncRuntimeProviderFailure
PASS observed implementation trace: AsyncRuntimeCancelBeforeProvider
PASS observed implementation trace: AsyncRuntimeDeniedBatch
PASS observed implementation trace: AsyncRuntimeDeniedPython
PASS observed implementation trace: AsyncRuntimeNestedDenial
PASS observed implementation trace: AsyncRuntimeUndeclaredTool
PASS observed implementation trace: AsyncRuntimeTruncatedBatch
PASS observed implementation trace: AsyncRuntimeCancelTruncatedBatch
PASS observed implementation trace: AsyncRuntimeIterationLimit
PASS observed implementation trace: AsyncRuntimeSteering
PASS observed implementation trace: AsyncRuntimeCancelledBatch
PASS observed implementation trace: AsyncRuntimeQueueEdits
PASS observed implementation trace: MemoryRuntimeStorage
PASS observed implementation trace: ArchiveScopeRuntimeStorage
PASS rejected trace mutation: AsyncRuntimeToolAnswerEarlyContinuation
PASS rejected trace mutation: AsyncRuntimeToolAnswerMissingResult
PASS rejected trace mutation: AsyncRuntimeToolAnswerReplayedPermission
PASS rejected trace mutation: AsyncRuntimeDeniedBatchMissingResult
PASS rejected trace mutation: AsyncRuntimeDeniedBatchLostDenial
PASS rejected trace mutation: AsyncRuntimeDeniedBatchReplayedPermission
PASS rejected trace mutation: AsyncRuntimeDeniedPythonMissingResult
PASS rejected trace mutation: AsyncRuntimeDeniedPythonLostDenial
PASS rejected trace mutation: AsyncRuntimeDeniedPythonReplayedPermission
PASS rejected trace mutation: AsyncRuntimeNestedDenialMissingResult
PASS rejected trace mutation: AsyncRuntimeNestedDenialLostDenial
PASS rejected trace mutation: AsyncRuntimeNestedDenialReplayedPermission
PASS rejected trace mutation: AsyncRuntimeUndeclaredToolEarlyContinuation
PASS rejected trace mutation: AsyncRuntimeUndeclaredToolMissingResult
PASS rejected trace mutation: AsyncRuntimeTruncatedBatchEarlyContinuation
PASS rejected trace mutation: AsyncRuntimeTruncatedBatchMissingResult
PASS rejected trace mutation: AsyncRuntimeCancelTruncatedBatchMissingResult
PASS rejected trace mutation: AsyncRuntimeIterationLimitEarlyContinuation
PASS rejected trace mutation: AsyncRuntimeIterationLimitMissingResult
PASS rejected trace mutation: AsyncRuntimeIterationLimitReplayedPermission
PASS rejected trace mutation: AsyncRuntimeSteeringLostSteeringCommit
PASS rejected trace mutation: AsyncRuntimeCancelledBatchMissingCancellationRepair
PASS rejected trace mutation: AsyncRuntimeQueueEditsLostQueueDeletion
PASS rejected trace mutation: MemoryRuntimeStorageForgottenDisclosure
PASS rejected trace mutation: ArchiveScopeRuntimeStorageWrongScope
PASS model fault: AsyncRuntimePrematureIdle violates SingleTurnOwnership
PASS model fault: MemoryRuntimeUndrainedEpisode violates EpisodeLifecycleIsDisjoint
PASS model fault: ArchiveScopeRuntimeUnscopedHistory violates DisclosureMatchesRequestedScope
```

## Whitepaper artifact

`make whitepaper` completed with exit status 0 using pdfLaTeX and BibTeX.
The final PDF has 8 pages and 8 separately defined bibliography entries.
All pages were rendered with Poppler and visually reviewed; the final pages
have no clipped text or overlapping elements. Citation keys and reported
numbers were checked against the bibliography and gate output.

PDF SHA-256: `46f18908610e3d436d93156a4d7ad7549a807faeab5b32c18110a7e729163a44`.
