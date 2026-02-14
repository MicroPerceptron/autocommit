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
- Runtime embedding extraction in `crates/llama-runtime` uses `llama_tokenize` + `llama_encode`/`llama_decode` + `llama_get_embeddings*`.
- To enable real embedding extraction at runtime, set `AUTOCOMMIT_EMBED_MODEL=/absolute/path/to/model.gguf` (or fallback `LLAMA_MODEL_PATH`).
- Override source path with:
  - `LLAMA_CPP_DIR=/absolute/path/to/llama.cpp`
