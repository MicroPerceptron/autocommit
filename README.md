# autocommit

Rust workspace scaffold for evolving `autocommit` from shell script to a modular CLI.

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
- It disables tests/examples/tools/server for faster compile.
- Override source path with:
  - `LLAMA_CPP_DIR=/absolute/path/to/llama.cpp`
