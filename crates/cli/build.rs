fn main() {
    // When linking with Intel MKL (SYCL backend), the Rust linker passes
    // -Wl,--as-needed which drops libmkl_core.so from DT_NEEDED because
    // our binary doesn't directly reference its symbols. At runtime,
    // libmkl_sycl_blas.so needs mkl_core symbols (e.g. mkl_serv_strnlen_s)
    // but can't find them. Disable --as-needed so all MKL shared libs are
    // recorded as needed in the final binary.
    #[cfg(target_os = "linux")]
    if std::env::var("GGML_SYCL").is_ok() {
        println!("cargo:rerun-if-env-changed=GGML_SYCL");
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    }
}
