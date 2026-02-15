# autocommit

Rust workspace scaffold for evolving `tmp/autocommit.sh` from shell script to a modular CLI.

## Workspace layout

- `crates/core`: shared application logic.
- `crates/cli`: binary entrypoint.
- `crates/llama-sys`: native `llama.cpp` build integration.
- `third_party/llama.cpp`: upstream submodule.

## Build modes

- Fast default check (does not build `llama.cpp`):
  - `cargo check`
- Native check (builds `third_party/llama.cpp` via CMake first):
  - `cargo check -p autocommit-cli --features llama-native`

## Native build notes

- `crates/llama-sys/build.rs` configures and builds `llama.cpp` with CMake.
- The same `build.rs` runs `bindgen` on `llama.h` to generate Rust FFI bindings.
- It disables tests/examples/tools/server for faster compile.
- `bindgen` requires a working `libclang` installation on the build machine.
- Runtime inference in `crates/llama-runtime` now uses multi-sequence batching for chunk analysis on a shared `llama_model` + dedicated contexts.
- Runtime embedding extraction uses `llama_tokenize` + `llama_encode`/`llama_decode` + `llama_get_embeddings*`.
- To enable runtime analysis/embedding, set `AUTOCOMMIT_EMBED_MODEL=/absolute/path/to/model.gguf` (or fallback `LLAMA_MODEL_PATH`).
- Per-repo KV persistence:
  - Run `autocommit-cli init --profile auto` once inside a git repo to initialize `.git/autocommit/kv/`.
  - `init` warms the generation context and writes `generation.session` + `metadata.json`.
  - Subsequent `autocommit-cli analyze` / `autocommit-cli commit` runs auto-attempt state load from that repo cache.
- Override source path with:
  - `LLAMA_CPP_DIR=/absolute/path/to/llama.cpp`
