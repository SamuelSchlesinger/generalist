SHELL := /bin/sh

.PHONY: check lint test fmt clippy shellcheck traceability hooks-check tla setup hooks tla-tools doctor

check: lint test

lint: fmt clippy shellcheck traceability hooks-check tla

test:
	cargo test --all-targets --all-features --locked

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features --locked -- -D warnings

shellcheck:
	shellcheck .githooks/pre-commit .githooks/pre-push scripts/*.sh

traceability:
	./scripts/check-runtime-traceability.sh

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
