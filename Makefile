SHELL := /bin/sh

.PHONY: check lint test fmt format-staged clippy shellcheck traceability memory-research hooks-check tla setup hooks tla-tools doctor

check: lint test

lint: fmt clippy shellcheck traceability memory-research hooks-check tla

test:
	cargo test --all-targets --all-features --locked

fmt:
	cargo fmt --all -- --check

# Auto-format the staged Rust files and re-stage them, so a commit always
# carries rustfmt-clean content and the working tree never diverges from it.
# `make fmt` still checks the whole tree, so anything outside the staged set
# is reported rather than silently rewritten.
format-staged:
	@staged=$$(git diff --cached --name-only --diff-filter=ACM -- '*.rs'); \
	if [ -n "$$staged" ]; then \
		echo "$$staged" | xargs rustfmt --edition 2021; \
		echo "$$staged" | xargs git add; \
	fi

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

hooks-check:
	test -x .githooks/pre-commit
	test -x .githooks/pre-push

tla:
	./scripts/check-tla.sh

setup: tla-tools hooks

hooks:
	./scripts/install-git-hooks.sh

tla-tools:
	./scripts/install-tla-tools.sh

doctor:
	./scripts/doctor.sh
