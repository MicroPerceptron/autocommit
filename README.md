# autocommit

AI-powered commit and pull request generation that runs entirely on your machine. Analyzes local git diffs using a bundled LLM and drafts structured, high-signal commit messages and PR descriptions through an interactive terminal workflow.

No API keys. No cloud. Fully local inference powered by [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

- **Commit generation** — analyzes staged/worktree diffs, generates conventional commit messages with interactive approve/edit/cancel flow
- **PR generation** — generates PR title and body, creates or updates pull requests via `gh`, with branch selection and optional push
- **Version bump recommendations** — detects manifest files (Cargo.toml, package.json, go.mod, etc.), suggests semver bumps, syncs lockfiles after applying
- **Local inference** — ships with llama.cpp compiled in, supports Metal (macOS), CUDA (NVIDIA), Vulkan (cross-vendor), and SYCL (Intel) GPU acceleration
- **Per-repo config** — caches model state and runtime settings under `.git/autocommit/` for fast subsequent runs

## Installation

### Pre-built binaries

Download from [GitHub Releases](https://github.com/MicroPerceptron/autocommit/releases):

```sh
# macOS (Apple Silicon)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-aarch64-apple-darwin.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/

# macOS (Intel)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-x86_64-apple-darwin.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/

# Linux (CPU)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/

# Linux (CUDA — NVIDIA GPUs)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-x86_64-unknown-linux-gnu-cuda.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/

# Linux (Vulkan — Intel/AMD/NVIDIA GPUs, requires Vulkan drivers)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-x86_64-unknown-linux-gnu-vulkan.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/

# Linux (SYCL — Intel GPUs, requires Intel oneAPI runtime)
curl -L https://github.com/MicroPerceptron/autocommit/releases/latest/download/autocommit-x86_64-unknown-linux-gnu-sycl.tar.gz | tar xz
sudo mv autocommit /usr/local/bin/
```

### cargo binstall

```sh
cargo binstall autocommit-cli
```

### Build from source

Requires: Rust toolchain, Git, CMake, C++17 compiler, libclang

```sh
git clone --recursive https://github.com/MicroPerceptron/autocommit.git
cd autocommit
cargo install --path crates/cli --locked --features llama-native
```

#### GPU backend selection

By default, Metal is used on macOS and CUDA is auto-detected on Linux. To build with a different GPU backend:

```sh
# Vulkan (requires Vulkan SDK + glslc)
GGML_VULKAN=ON cargo install --path crates/cli --locked --features llama-native

# SYCL / Intel oneAPI (requires oneAPI toolkit)
source /opt/intel/oneapi/setvars.sh
GGML_SYCL=ON cargo install --path crates/cli --locked --features llama-native
```

Only one GPU backend can be active at a time.

## Quick start

```sh
# 1. Initialize — select a model and configure commit policy
autocommit init

# 2. Make some changes, then generate a commit
autocommit commit

# 3. Generate a pull request
autocommit pr --push
```

## Usage

### Commands

| Command              | Description                                        |
| -------------------- | -------------------------------------------------- |
| `autocommit commit`  | Generate and create a commit from local changes    |
| `autocommit pr`      | Generate and create/update a pull request          |
| `autocommit init`    | Initialize per-repo model config and commit policy |
| `autocommit config`  | View and update per-repo settings                  |
| `autocommit analyze` | Analyze diff and emit a structured JSON report     |
| `autocommit clean`   | Remove persisted KV generation cache               |

Run `autocommit <command> --help` for detailed options.

### Common workflows

```sh
# Interactive commit (default)
autocommit commit

# Non-interactive / CI mode
autocommit commit --yes --no-interactive --staged

# Dry run — preview without creating a commit
autocommit commit --dry-run

# PR targeting a specific base branch
autocommit pr --base origin/main --push

# Use a specific model
autocommit commit --hf-repo ggml-org/Qwen3-4B-GGUF:Q4_K_M
```

### Model configuration

Model settings are resolved in order:

1. CLI flags (`--model-path`, `--hf-repo`, `--cache-dir`, `--profile`)
2. Per-repo metadata saved by `autocommit init`
3. Runtime defaults

Supported model sources:

- **Hugging Face** — `--hf-repo ggml-org/Qwen3-1.7B-GGUF:Q4_K_M` (auto-downloaded and cached)
- **Local GGUF** — `--model-path /path/to/model.gguf`

List cached models:

```sh
autocommit init --list-cached-models
```

## Architecture

```
crates/
  core/           Pure Rust analysis pipeline, dispatch, and diff processing
  cli/            Terminal application (clap, dialoguer, indicatif)
  llama-runtime/  Runtime integration layer wrapping llama-sys
  llama-sys/      Native llama.cpp bindings (CMake build + bindgen FFI)
third_party/
  llama.cpp/      Upstream submodule (compiled from source, statically linked)
```

## Development

```sh
# Run tests
cargo test --workspace --features llama-native

# Lint
cargo clippy --workspace --features llama-native -- -D warnings

# Format
cargo fmt --all

# Quick check (no native build)
cargo check -p autocommit-cli
```

## Troubleshooting

| Problem                                | Solution                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `runtime model path is not configured` | Run `autocommit init` or pass `--hf-repo` / `--model-path`                   |
| GPG signing failures                   | Run `autocommit commit --configure-commit-policy` to adjust signing settings |
| PR creation errors from `gh`           | Ensure `gh auth status` succeeds and the base/head branches differ           |

## License

MIT
