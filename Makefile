SHELL := /bin/sh

.PHONY: check lint test fmt format-staged clippy shellcheck traceability memory-research agent-benchmarks memory-evaluation hooks-check tla conformance setup hooks tla-tools doctor

check: lint test

lint: fmt clippy shellcheck traceability memory-research agent-benchmarks hooks-check tla

test:
	cargo test --all-targets --all-features --locked

fmt:
	cargo fmt --all -- --check

# Auto-format fully staged Rust files and re-stage them. Partially staged files
# are rejected so the hook never broadens a commit with unstaged work.
# `make fmt` still checks the whole tree, so out-of-scope files are reported
# rather than silently rewritten.
format-staged:
	@python3 scripts/format_staged_rust.py

clippy:
	cargo clippy --all-targets --all-features --locked -- -D warnings

shellcheck:
	shellcheck .githooks/pre-commit .githooks/pre-push scripts/*.sh

traceability:
	./scripts/check-runtime-traceability.sh

memory-research:
	PYTHONDONTWRITEBYTECODE=1 python3 docs/research/agent-memory/systems/data/check_corpus.py
	PYTHONDONTWRITEBYTECODE=1 python3 docs/research/agent-memory/safety-evaluation/data/validate-branch.py
	PYTHONDONTWRITEBYTECODE=1 python3 docs/research/agent-memory/data/validate_corpus.py

agent-benchmarks:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks/agent_transport -p 'test_*.py'
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks/episodic_memory -p 'test_*.py'

memory-evaluation:
	PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/episodic_memory/run.py

hooks-check:
	test -x .githooks/pre-commit
	test -x .githooks/pre-push
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_format_staged_rust.py

tla:
	./scripts/check-tla.sh
	./scripts/check-model-conformance.sh

conformance:
	./scripts/check-model-conformance.sh

setup: tla-tools hooks

hooks:
	./scripts/install-git-hooks.sh

tla-tools:
	./scripts/install-tla-tools.sh

doctor:
	./scripts/doctor.sh
