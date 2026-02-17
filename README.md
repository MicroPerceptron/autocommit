# autocommit

`autocommit` is a Rust CLI that analyzes local git diffs and drafts high-signal commit and pull request text with an interactive terminal workflow.

## Features

- AI-assisted commit message generation with interactive approve/edit flow.
- AI-assisted PR title/body generation with branch selection and optional push.
- Per-repo cache and runtime settings (`.git/autocommit/kv/`).
- Optional version bump recommendations with apply/skip confirmation.
- Native local inference path powered by `llama.cpp` (`llama-native` feature).

## Install

### Prerequisites

- Rust toolchain (`cargo`).
- Git.
- For `llama-native`: CMake + `libclang` (for `bindgen`), and a C++ toolchain.

### Install from source (local workspace)

- Lightweight build (no native llama runtime):
  - `cargo install --path crates/cli --locked`
- Native local inference build:
  - `cargo install --path crates/cli --locked --features llama-native`

### Run without installing

- `cargo run --release -p autocommit-cli -- <command> [options]`

## Quickstart

1. Initialize repository cache and model/runtime config:
   - `autocommit-cli init`
2. Review or update repository config later:
   - `autocommit-cli config`
3. Generate and review a commit:
   - `autocommit-cli commit`
4. Generate and review a pull request:
   - `autocommit-cli pr`

For non-interactive runs, use `--yes` and explicit flags (for example `--base`, `--push`).

## Commands

- `autocommit-cli analyze`
  - Analyze staged/worktree diff and emit a structured report.
- `autocommit-cli commit`
  - Generate, review, and create commit messages.
- `autocommit-cli pr`
  - Generate PR title/body and create or update PRs.
- `autocommit-cli init`
  - Initialize per-repo cache, model config, and commit policy defaults.
- `autocommit-cli config`
  - View and update per-repo model/profile/cache settings and commit policy.
- `autocommit-cli clean`
  - Remove persisted generation KV cache for the current repo.
- `autocommit-cli explain-dispatch`
  - Print dispatch routing explanation for a representative sample.

Use generated help for the latest options:

- `autocommit-cli --help`
- `autocommit-cli --version`
- `autocommit-cli <command> --help`

## Model Configuration

When built with `llama-native`, runtime model settings can come from:

1. Explicit CLI flags (`--model-path`, `--hf-repo`, `--cache-dir`, `--profile`).
2. Per-repo metadata saved by `autocommit-cli init`.
3. Runtime defaults (including default HF model selection when unset).

Examples:

- `autocommit-cli commit --hf-repo ggml-org/Qwen3-1.7B-GGUF:Q4_K_M`
- `autocommit-cli pr --hf-repo ggml-org/gemma-3-1b-it-GGUF:Q8_0 --cache-dir ~/.cache/llama.cpp`

## Typical Workflows

### Commit (interactive)

- `autocommit-cli commit`

### Commit (non-interactive / CI style)

- `autocommit-cli commit --yes --no-interactive --staged --dry-run`

### PR to a specific base

- `autocommit-cli pr --yes --base origin/dev --push`

## Development

Workspace layout:

- `crates/core`: shared analysis/pipeline logic.
- `crates/cli`: terminal application.
- `crates/llama-runtime`: runtime integration layer.
- `crates/llama-sys`: native `llama.cpp` integration.
- `third_party/llama.cpp`: upstream submodule.

Useful checks:

- `cargo check -p autocommit-cli`
- `cargo check -p autocommit-cli --features llama-native`

## Troubleshooting

- `runtime model path is not configured`
  - Run `autocommit-cli init` or pass `--hf-repo` / `--model-path`.
- GPG signing failures during commit
  - Configure commit policy with `autocommit-cli commit --configure-commit-policy`, or set up `gpg` and `user.signingkey`.
- PR creation errors from `gh`
  - Ensure `gh auth status` succeeds and the selected base/head actually differ.
