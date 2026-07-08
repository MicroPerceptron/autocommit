use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The llama.cpp release tag to use for prebuilt binary downloads.
const LLAMA_CPP_RELEASE_TAG: &str = "b9837";

/// Returns the macOS major version (e.g. 14 for macOS 14.x Sonoma), or 0
/// if not running on macOS or the version could not be determined.
fn macos_version_major(target: &str) -> u32 {
    if !target.ends_with("-apple-darwin") {
        return 0;
    }
    Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .and_then(|v| {
            v.trim()
                .split('.')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
        })
        .unwrap_or(0)
}

fn run(command: &mut Command, step: &str) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run cmake during {step}: {err}"));
    assert!(status.success(), "cmake {step} failed with status {status}");
}

fn profile_name() -> &'static str {
    match env::var("PROFILE").as_deref() {
        Ok("release") => "Release",
        _ => "Debug",
    }
}

fn cmake_parallel_jobs() -> usize {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    std::cmp::max(1, nproc.saturating_mul(2) / 3)
}

/// Discover the directory containing a static library via the C compiler and add
/// it as a native link search path. Needed for libraries like `libgomp.a` that
/// live in GCC's internal lib directory.
#[cfg(target_os = "linux")]
fn add_lib_dir_from_compiler(cc: &str, lib_file: &str) {
    if let Ok(output) = Command::new(cc)
        .arg(format!("-print-file-name={lib_file}"))
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if let Some(dir) = Path::new(&path).parent()
            && dir.is_dir()
        {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
}

/// Discover the library directory for a package via pkg-config and add it as a
/// native link search path. Needed for libraries like `libopenblas.a` that live
/// in variant-specific subdirectories on Debian/Ubuntu.
#[cfg(target_os = "linux")]
fn add_lib_dir_from_pkg_config(package: &str) {
    if let Ok(output) = Command::new("pkg-config")
        .args(["--variable=libdir", package])
        .output()
        && output.status.success()
    {
        let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !dir.is_empty() && Path::new(&dir).is_dir() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
}

/// Check whether `GGML_<backend>` is explicitly set to a non-empty value.
///
/// CI passes backend flags as `GGML_VULKAN: ${{ matrix.vulkan && 'ON' || '' }}`,
/// which exports the variable as an empty string on targets that don't use that
/// backend. An empty value must count as "not set" — otherwise a plain CPU or
/// macOS build would try to link the Vulkan libraries that aren't present in its
/// (non-Vulkan) prebuilt archive.
fn is_explicitly_set(var: &str) -> bool {
    env::var(var).is_ok_and(|v| !v.trim().is_empty())
}

/// Returns true if *any other* GPU backend was explicitly requested via env var.
/// Used to suppress auto-detection: if the user asked for Vulkan, don't auto-detect CUDA.
fn another_backend_explicitly_requested(this: &str) -> bool {
    ["GGML_CUDA", "GGML_SYCL", "GGML_VULKAN"]
        .iter()
        .any(|&var| var != this && is_explicitly_set(var))
}

fn detect_cuda() -> bool {
    println!("cargo:rerun-if-env-changed=GGML_CUDA");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Explicit override via GGML_CUDA env var (supports ON/OFF)
    if let Ok(val) = env::var("GGML_CUDA") {
        return matches!(val.to_ascii_lowercase().as_str(), "1" | "on" | "true");
    }

    // If the user explicitly requested another backend, skip CUDA auto-detection.
    if another_backend_explicitly_requested("GGML_CUDA") {
        return false;
    }

    // macOS uses Metal, not CUDA
    if cfg!(target_os = "macos") {
        return false;
    }

    // Auto-detect CUDA Toolkit via CUDA_PATH (set by NVIDIA installer on Windows,
    // commonly set on Linux too)
    if let Ok(cuda_path) = env::var("CUDA_PATH")
        && Path::new(&cuda_path).exists()
    {
        println!("cargo:warning=CUDA auto-detected via CUDA_PATH={cuda_path}");
        return true;
    }

    // On Linux, check the conventional /usr/local/cuda path
    #[cfg(target_os = "linux")]
    if Path::new("/usr/local/cuda/bin/nvcc").is_file() {
        println!("cargo:warning=CUDA auto-detected via /usr/local/cuda");
        return true;
    }

    false
}

fn detect_sycl() -> bool {
    println!("cargo:rerun-if-env-changed=GGML_SYCL");
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");

    // SYCL/oneAPI is Linux-only
    if !cfg!(target_os = "linux") {
        return false;
    }

    // Explicit override via GGML_SYCL env var (supports ON/OFF)
    if let Ok(val) = env::var("GGML_SYCL") {
        return matches!(val.to_ascii_lowercase().as_str(), "1" | "on" | "true");
    }

    // If the user explicitly requested another backend, skip SYCL auto-detection.
    if another_backend_explicitly_requested("GGML_SYCL") {
        return false;
    }

    if let Ok(oneapi_root) = env::var("ONEAPI_ROOT")
        && Path::new(&oneapi_root).exists()
    {
        println!("cargo:warning=SYCL auto-detected via ONEAPI_ROOT={oneapi_root}");
        return true;
    }

    false
}

fn detect_vulkan() -> bool {
    println!("cargo:rerun-if-env-changed=GGML_VULKAN");

    if let Ok(val) = env::var("GGML_VULKAN") {
        return matches!(val.to_ascii_lowercase().as_str(), "1" | "on" | "true");
    }

    // No auto-detection — Vulkan SDK is commonly installed on dev machines
    // that may not intend to use it here. Require explicit opt-in.
    false
}

fn find_intel_compiler(name: &str) -> PathBuf {
    // Check if it's on PATH (user sourced setvars.sh)
    if let Ok(output) = Command::new("which").arg(name).output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }

    // Fallback: look under ONEAPI_ROOT
    if let Ok(oneapi_root) = env::var("ONEAPI_ROOT") {
        let candidate = PathBuf::from(&oneapi_root)
            .join("compiler")
            .join("latest")
            .join("bin")
            .join(name);
        if candidate.is_file() {
            return candidate;
        }
    }

    panic!(
        "Intel compiler '{name}' not found. \
         Ensure Intel oneAPI is installed and setvars.sh is sourced."
    );
}

// ---------------------------------------------------------------------------
// Prebuilt binary support
// ---------------------------------------------------------------------------

/// Maps a Rust target triple to the llama.cpp binary release archive filename,
/// or returns `None` if no prebuilt archive is available for this target.
fn prebuilt_archive_name(target: &str) -> Option<&'static str> {
    match target {
        "aarch64-apple-darwin" => Some("llama-b9837-bin-macos-arm64.tar.gz"),
        "x86_64-apple-darwin" => Some("llama-b9837-bin-macos-x64.tar.gz"),
        "x86_64-unknown-linux-gnu" => {
            if is_explicitly_set("GGML_VULKAN") {
                Some("llama-b9837-bin-ubuntu-vulkan-x64.tar.gz")
            } else {
                Some("llama-b9837-bin-ubuntu-x64.tar.gz")
            }
        }
        "aarch64-unknown-linux-gnu" => {
            if is_explicitly_set("GGML_VULKAN") {
                Some("llama-b9837-bin-ubuntu-vulkan-arm64.tar.gz")
            } else {
                Some("llama-b9837-bin-ubuntu-arm64.tar.gz")
            }
        }
        _ => None,
    }
}

/// Maps a Rust target triple to the appropriate `libggml-cpu` library name
/// for Linux. macOS uses a single generic `ggml-cpu`.
fn ggml_cpu_lib_name(target: &str) -> &'static str {
    match target {
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => "ggml-cpu",
        "aarch64-unknown-linux-gnu" => "ggml-cpu",
        "x86_64-unknown-linux-gnu" => "ggml-cpu-x64",
        _ => "ggml-cpu",
    }
}

/// Returns the list of shared library names (without `lib` prefix or extension)
/// that should be linked from the prebuilt archive for the given target.
fn prebuilt_link_libs(target: &str) -> Vec<&'static str> {
    let mut libs = vec![
        "llama",
        "llama-common",
        "ggml",
        "ggml-base",
        ggml_cpu_lib_name(target),
    ];
    if target.ends_with("-apple-darwin") {
        libs.push("ggml-blas");
        // The prebuilt ggml-metal.dylib requires macOS 15+ (references
        // MTLResidencySetDescriptor).  Skip it on older macOS so the
        // CPU / BLAS backend is used instead.
        if macos_version_major(target) >= 15 {
            libs.push("ggml-metal");
        }
    }
    // When Vulkan is enabled, the prebuilt archive includes the Vulkan backend.
    if is_explicitly_set("GGML_VULKAN") {
        libs.push("ggml-vulkan");
    }
    libs
}

/// Returns the GitHub release download URL for a prebuilt archive.
fn prebuilt_download_url(archive: &str) -> String {
    format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{}/{}",
        LLAMA_CPP_RELEASE_TAG, archive
    )
}

/// Download a file via `curl` to the given destination.
fn download_file(url: &str, dest: &Path) {
    let status = Command::new("curl")
        .args(["-sSL", "--fail", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch curl: {e}"));
    assert!(
        status.success(),
        "curl download failed (url={url}, dest={})",
        dest.display()
    );
}

/// Extract a `.tar.gz` archive to the given directory.
fn extract_tar_gz(archive: &Path, dest: &Path) {
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch tar: {e}"));
    assert!(
        status.success(),
        "tar extraction failed (archive={}, dest={})",
        archive.display(),
        dest.display()
    );
}

fn has_llama_source(path: &Path) -> bool {
    path.join("CMakeLists.txt").is_file()
        && path.join("include").join("llama.h").is_file()
        && path.join("common").is_dir()
}

/// Download and extract the source archive matching `LLAMA_CPP_RELEASE_TAG`.
/// The bridge must be compiled against headers from the same release as the
/// prebuilt shared libraries; newer submodule headers can drift across
/// llama.cpp's internal `common` APIs.
fn download_prebuilt_source(out_dir: &Path) -> PathBuf {
    let cache_dir = out_dir.join("llama-prebuilt");
    let src_marker = cache_dir.join(".extracted-src");
    let src_dir = cache_dir.join("source");

    if src_marker.is_file() && has_llama_source(&src_dir) {
        println!(
            "cargo:warning=using cached llama.cpp source from {}",
            src_dir.display()
        );
        return src_dir;
    }

    fs::create_dir_all(&cache_dir).expect("failed to create prebuilt cache dir");

    let src_url = format!(
        "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{}.tar.gz",
        LLAMA_CPP_RELEASE_TAG
    );
    let src_archive = out_dir.join(format!("llama-{}.tar.gz", LLAMA_CPP_RELEASE_TAG));
    println!("cargo:warning=downloading llama.cpp source for headers from {src_url}");
    download_file(&src_url, &src_archive);
    extract_tar_gz(&src_archive, &cache_dir);

    // The source tarball extracts to llama.cpp-<tag>. Normalize to a stable
    // path under OUT_DIR so subsequent Cargo invocations can reuse it.
    let raw_src = cache_dir.join(format!("llama.cpp-{}", LLAMA_CPP_RELEASE_TAG));
    if raw_src.is_dir() && src_dir.exists() && !has_llama_source(&src_dir) {
        fs::remove_dir_all(&src_dir).ok();
    }
    if raw_src.is_dir() && !src_dir.exists() {
        fs::rename(&raw_src, &src_dir).ok();
    }
    fs::remove_file(&src_archive).ok();
    fs::write(&src_marker, "extracted").ok();

    assert!(
        has_llama_source(&src_dir),
        "downloaded llama.cpp source for {} is incomplete at {}",
        LLAMA_CPP_RELEASE_TAG,
        src_dir.display()
    );

    src_dir
}

/// Download and extract prebuilt llama.cpp shared libraries AND the matching
/// source archive for the current target. Returns `(lib_dir, source_dir)`.
fn download_prebuilt(out_dir: &Path, archive_name: &str) -> (PathBuf, PathBuf) {
    let cache_dir = out_dir.join("llama-prebuilt");
    let bin_marker = cache_dir.join(".extracted-bin");

    let lib_dir = cache_dir.join("llama-b9837");

    if bin_marker.is_file() && lib_dir.is_dir() && has_llama_source(&cache_dir.join("source")) {
        println!(
            "cargo:warning=using cached prebuilt llama.cpp from {}",
            cache_dir.display()
        );
        return (lib_dir, cache_dir.join("source"));
    }

    fs::create_dir_all(&cache_dir).expect("failed to create prebuilt cache dir");

    // Download and extract binary archive
    if !bin_marker.is_file() {
        let url = prebuilt_download_url(archive_name);
        let archive_path = out_dir.join(archive_name);
        println!("cargo:warning=downloading prebuilt llama.cpp libraries from {url}");
        download_file(&url, &archive_path);
        extract_tar_gz(&archive_path, &cache_dir);
        fs::remove_file(&archive_path).ok();
        fs::write(&bin_marker, "extracted").ok();
    }

    let src_dir = download_prebuilt_source(out_dir);
    (lib_dir, src_dir)
}

/// Set up link search paths and emit `cargo:rustc-link-lib` directives for
/// the prebuilt shared libraries in `lib_dir`.
fn emit_prebuilt_link_deps(lib_dir: &Path, target: &str) {
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.to_string_lossy()
    );

    // Embed rpath so the built binary/test can find the shared libraries at
    // runtime without needing LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        lib_dir.to_string_lossy()
    );

    for lib in prebuilt_link_libs(target) {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    // When Vulkan is requested via env var, link the Vulkan loader.
    if is_explicitly_set("GGML_VULKAN") {
        emit_vulkan_link_deps();
    }

    // Platform-level system dependencies — these are already linked into the
    // prebuilt shared libs, but we must repeat them here because Rust's linker
    // invocation needs to resolve all symbols at link time.
    if target.ends_with("-apple-darwin") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        if macos_version_major(target) >= 15 {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
        }
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");

        if let Ok(output) = Command::new("clang").arg("--print-resource-dir").output()
            && output.status.success()
        {
            let resource_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let rt_lib_dir = PathBuf::from(&resource_dir).join("lib").join("darwin");
            if rt_lib_dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", rt_lib_dir.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // SSL/crypto for httplib (download support inside libllama-common)
    if target.ends_with("-apple-darwin") {
        for candidate in [
            "/opt/homebrew/opt/openssl@3/lib",
            "/usr/local/opt/openssl@3/lib",
            "/opt/homebrew/opt/openssl/lib",
            "/usr/local/opt/openssl/lib",
        ] {
            if Path::new(candidate).is_dir() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=ssl");
    println!("cargo:rustc-link-lib=dylib=crypto");
}

/// Decide whether we can and should use a prebuilt binary release instead of
/// building llama.cpp from source.
fn should_use_prebuilt(_source_dir: &Path) -> Option<&'static str> {
    // If LLAMA_CPP_PREBUILT_DIR is set, use the prebuilt libs from that directory.
    if env::var_os("LLAMA_CPP_PREBUILT_DIR").is_some() {
        let target = env::var("TARGET").expect("missing TARGET");
        return prebuilt_archive_name(&target);
    }

    // If the user explicitly pointed at custom source, respect that.
    if env::var_os("LLAMA_CPP_DIR").is_some() {
        return None;
    }

    // When CUDA or SYCL is requested, llama.cpp does not publish prebuilt
    // archives for those backends (binary releases only cover CPU, Metal,
    // and Vulkan on Linux). Fall back to building from source.
    if is_explicitly_set("GGML_CUDA") || is_explicitly_set("GGML_SYCL") {
        return None;
    }

    let target = env::var("TARGET").expect("missing TARGET");
    prebuilt_archive_name(&target)
}

/// Locate the prebuilt libraries directory — either from env var override or
/// by downloading the release archive. Returns `(lib_dir, source_dir)`.
fn resolve_prebuilt(out_dir: &Path, archive_name: &str) -> (PathBuf, PathBuf) {
    if let Ok(prebuilt_dir_str) = env::var("LLAMA_CPP_PREBUILT_DIR") {
        let path = PathBuf::from(&prebuilt_dir_str);
        assert!(
            path.is_dir(),
            "LLAMA_CPP_PREBUILT_DIR={} is not a valid directory",
            path.display()
        );
        println!(
            "cargo:warning=using prebuilt llama.cpp from {}",
            path.display()
        );

        let mut candidates = Vec::new();
        if let Some(source_dir) = env::var_os("LLAMA_CPP_PREBUILT_SOURCE_DIR") {
            candidates.push(PathBuf::from(source_dir));
        }
        if let Some(source_dir) = env::var_os("LLAMA_CPP_DIR") {
            candidates.push(PathBuf::from(source_dir));
        }
        if let Some(parent) = path.parent() {
            candidates.push(parent.join("source"));
            candidates.push(parent.join(format!("llama.cpp-{}", LLAMA_CPP_RELEASE_TAG)));
        }

        let source_dir = candidates
            .into_iter()
            .find(|candidate| has_llama_source(candidate))
            .unwrap_or_else(|| download_prebuilt_source(out_dir));

        (path, source_dir)
    } else {
        download_prebuilt(out_dir, archive_name)
    }
}

// ---------------------------------------------------------------------------
// Source build (fallback)
// ---------------------------------------------------------------------------

fn build_from_source(source_dir: &Path, out_dir: &Path, manifest_dir: &Path) {
    generate_bindings(source_dir, out_dir);

    let cmake_lists = source_dir.join("CMakeLists.txt");
    assert!(
        cmake_lists.is_file(),
        "llama.cpp source not found at {} (set LLAMA_CPP_DIR to override)",
        source_dir.display()
    );

    println!("cargo:rerun-if-changed={}", cmake_lists.display());
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include").display()
    );

    let build_dir = out_dir.join("llama-cmake-build");
    let install_dir = out_dir.join("llama-cmake-install");

    fs::create_dir_all(&build_dir).expect("failed to create cmake build dir");
    fs::create_dir_all(&install_dir).expect("failed to create cmake install dir");

    let use_cuda = detect_cuda();
    let use_sycl = detect_sycl();
    let use_vulkan = detect_vulkan();

    let gpu_backends: Vec<&str> = [
        use_cuda.then_some("CUDA"),
        use_sycl.then_some("SYCL"),
        use_vulkan.then_some("Vulkan"),
    ]
    .into_iter()
    .flatten()
    .collect();

    if gpu_backends.len() > 1 {
        panic!(
            "Multiple GPU backends enabled: {}. Only one may be active at a time. \
             Set exactly one of GGML_CUDA, GGML_SYCL, or GGML_VULKAN.",
            gpu_backends.join(", ")
        );
    }

    let profile = profile_name();

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={profile}"))
        .arg(format!(
            "-DCMAKE_INSTALL_PREFIX={}",
            install_dir.to_string_lossy()
        ))
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg("-DLLAMA_BUILD_COMMON=ON")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        // b9837+ introduced these targets, both defaulting ON for a standalone
        // (top-level) build. We only need the libraries for FFI, so disable
        // them. LLAMA_BUILD_APP (the "unified binary") compiles app/llama.cpp,
        // which needs a generated build-info.h and breaks the source build;
        // LLAMA_BUILD_UI would fetch a prebuilt web UI. Both are unknown to
        // older releases, where CMake harmlessly ignores the unused flags.
        .arg("-DLLAMA_BUILD_APP=OFF")
        .arg("-DLLAMA_BUILD_UI=OFF");
    if use_cuda {
        configure.arg("-DGGML_CUDA=ON");
    }
    if use_sycl {
        configure.arg("-DGGML_SYCL=ON");
        configure.arg("-DGGML_SYCL_TARGET=INTEL");
        let icx = find_intel_compiler("icx");
        let icpx = find_intel_compiler("icpx");
        configure.arg(format!("-DCMAKE_C_COMPILER={}", icx.display()));
        configure.arg(format!("-DCMAKE_CXX_COMPILER={}", icpx.display()));
    }
    if use_vulkan {
        configure.arg("-DGGML_VULKAN=ON");
    }
    // Enable OpenBLAS for CPU GEMM on x86 Linux (non-SYCL builds).
    // SYCL already links MKL which provides BLAS. macOS uses Accelerate by default.
    #[cfg(target_os = "linux")]
    if !use_sycl {
        configure.arg("-DGGML_BLAS=ON");
        configure.arg("-DGGML_BLAS_VENDOR=OpenBLAS");
    }
    if cfg!(target_os = "macos") {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "14.0".to_string());
        configure.arg(format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"));
    }
    run(&mut configure, "configure");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile)
        .arg("--parallel")
        .arg(cmake_parallel_jobs().to_string());
    run(&mut build, "build");

    let mut install = Command::new("cmake");
    install
        .arg("--install")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile);
    run(&mut install, "install");

    build_common_bridge(source_dir, manifest_dir, false);
    emit_common_link_deps(&build_dir);
    emit_install_link_search_paths(&install_dir);
    if use_cuda {
        emit_cuda_link_deps();
    }
    if use_sycl {
        emit_sycl_link_deps();
    }
    if use_vulkan {
        emit_vulkan_link_deps();
    }
    // Static-link OpenMP and BLAS on non-SYCL Linux for minimal runtime deps.
    // SYCL uses Intel's iomp5 and MKL instead (linked in emit_sycl_link_deps).
    #[cfg(target_os = "linux")]
    if !use_sycl {
        // libgomp.a lives in GCC's internal lib dir, not the standard search path.
        add_lib_dir_from_compiler("cc", "libgomp.a");
        println!("cargo:rustc-link-lib=static=gomp");

        // libopenblas.a lives in a variant-specific subdir on Ubuntu/Debian
        // (e.g. /usr/lib/x86_64-linux-gnu/openblas-pthread/). Use pkg-config.
        add_lib_dir_from_pkg_config("openblas");
        println!("cargo:rustc-link-lib=static=openblas");
    }

    println!(
        "cargo:rustc-env=LLAMA_CPP_BUILD_DIR={}",
        build_dir.to_string_lossy()
    );
    println!(
        "cargo:rustc-env=LLAMA_CPP_INSTALL_DIR={}",
        install_dir.to_string_lossy()
    );
}

fn generate_bindings(source_dir: &Path, out_dir: &Path) {
    let header = source_dir.join("include").join("llama.h");
    let ggml_include = source_dir.join("ggml").join("include");

    assert!(
        header.is_file(),
        "llama header not found at {}",
        header.display()
    );
    assert!(
        ggml_include.is_dir(),
        "ggml headers not found at {}",
        ggml_include.display()
    );

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy().into_owned())
        .clang_arg(format!(
            "-I{}",
            source_dir.join("include").to_string_lossy()
        ))
        .clang_arg(format!("-I{}", ggml_include.to_string_lossy()))
        .allowlist_function("^llama_.*")
        .allowlist_function("^ggml_backend_load_all$")
        .allowlist_function("^ggml_backend_reg_count$")
        .allowlist_function("^ggml_backend_dev_count$")
        .allowlist_function("^ggml_backend_dev_get$")
        .allowlist_function("^ggml_backend_dev_type$")
        .allowlist_function("^ggml_backend_dev_name$")
        .allowlist_function("^ggml_backend_dev_description$")
        .allowlist_function("^ggml_backend_dev_memory$")
        .allowlist_type("^llama_.*")
        .allowlist_var("^LLAMA_.*")
        .allowlist_type("^ggml_.*")
        .allowlist_var("^GGML_.*")
        .blocklist_type("^FILE$")
        .blocklist_function("^llama_model_load_from_file_ptr$")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .unwrap_or_else(|err| panic!("bindgen failed for {}: {err}", header.display()));

    let out_file = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&out_file)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_file.display()));
}

fn emit_install_link_search_paths(install_dir: &Path) {
    let candidates = [
        install_dir.join("lib"),
        install_dir.join("lib64"),
        install_dir.join("bin"),
    ];

    for path in candidates {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
            emit_link_libs_from_dir(&path);
        }
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");

        // Link clang compiler runtime to provide __isPlatformVersionAtLeast
        // needed by @available() checks in llama.cpp's Objective-C Metal code.
        if let Ok(output) = Command::new("clang").arg("--print-resource-dir").output()
            && output.status.success()
        {
            let resource_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let rt_lib_dir = PathBuf::from(&resource_dir).join("lib").join("darwin");
            if rt_lib_dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", rt_lib_dir.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn emit_cuda_link_deps() {
    // Locate the CUDA toolkit library directory
    let lib_search_dirs: Vec<PathBuf> = if let Ok(cuda_path) = env::var("CUDA_PATH") {
        vec![
            PathBuf::from(&cuda_path).join("lib").join("x64"), // Windows
            PathBuf::from(&cuda_path).join("lib64"),           // Linux
        ]
    } else {
        // Fallback for Linux conventional path
        vec![PathBuf::from("/usr/local/cuda/lib64")]
    };

    for dir in &lib_search_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            // The CUDA toolkit ships a libcuda.so stub under lib64/stubs/ for
            // building on machines without a GPU driver (e.g. CI runners).
            let stubs = dir.join("stubs");
            if stubs.exists() {
                println!("cargo:rustc-link-search=native={}", stubs.display());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=static=cublas_static");
        println!("cargo:rustc-link-lib=static=cublasLt_static");
        // culibos provides the library loader used internally by static cublas/cublasLt
        println!("cargo:rustc-link-lib=static=culibos");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
}

fn emit_sycl_link_deps() {
    // Add search paths for oneAPI libraries
    if let Ok(oneapi_root) = env::var("ONEAPI_ROOT") {
        let compiler_lib = PathBuf::from(&oneapi_root)
            .join("compiler")
            .join("latest")
            .join("lib");
        if compiler_lib.exists() {
            println!("cargo:rustc-link-search=native={}", compiler_lib.display());
        }

        let mkl_lib = PathBuf::from(&oneapi_root)
            .join("mkl")
            .join("latest")
            .join("lib");
        if mkl_lib.exists() {
            println!("cargo:rustc-link-search=native={}", mkl_lib.display());
        }
    }

    println!("cargo:rustc-link-lib=dylib=sycl");
    println!("cargo:rustc-link-lib=dylib=mkl_sycl_blas");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=dylib=mkl_tbb_thread");
    println!("cargo:rustc-link-lib=dylib=mkl_core");
    println!("cargo:rustc-link-lib=dylib=svml");
    println!("cargo:rustc-link-lib=dylib=irc");
    println!("cargo:rustc-link-lib=dylib=imf");
    println!("cargo:rustc-link-lib=dylib=iomp5");
}

fn emit_vulkan_link_deps() {
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    } else {
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }
}

fn emit_link_libs_from_dir(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        // Unix / MinGW: libfoo.a
        if name.starts_with("lib") && name.ends_with(".a") {
            let lib_name = &name[3..name.len() - 2];
            if !lib_name.is_empty() {
                println!("cargo:rustc-link-lib=static={lib_name}");
            }
            continue;
        }

        // MSVC: foo.lib
        if cfg!(target_os = "windows") && name.ends_with(".lib") {
            let lib_name = &name[..name.len() - 4];
            if !lib_name.is_empty() {
                println!("cargo:rustc-link-lib=static={lib_name}");
            }
        }
    }
}

fn build_common_bridge(source_dir: &Path, manifest_dir: &Path, prebuilt: bool) {
    let bridge_cpp = manifest_dir
        .join("src")
        .join("autocommit_common_bridge.cpp");
    let common_dir = source_dir.join("common");
    let include_dir = source_dir.join("include");
    let ggml_include = source_dir.join("ggml").join("include");
    let vendor_dir = source_dir.join("vendor");

    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .file(&bridge_cpp)
        .include(&common_dir)
        .include(&include_dir)
        .include(&ggml_include)
        .include(&vendor_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-function");

    if prebuilt {
        builder.define("LLAMA_CPP_PREBUILT", None);
    }

    // macOS native ld requires BSD ar format; Homebrew's GNU ar (first in
    // PATH on many systems) creates GNU-format archives whose long-name
    // and symbol-table entries cause "archive member '/' not a mach-o file"
    // linker errors.  Explicitly use the system archiver on macOS.
    #[cfg(target_os = "macos")]
    {
        let sys_ar = std::path::PathBuf::from("/usr/bin/ar");
        if sys_ar.exists() {
            builder.archiver(&sys_ar);
        }
    }

    builder.compile("autocommit_common_bridge");
}

fn has_static_lib(dir: &Path, name: &str) -> bool {
    dir.join(format!("lib{name}.a")).exists() || dir.join(format!("{name}.lib")).exists()
}

fn emit_common_link_deps(build_dir: &Path) {
    let common_dir = build_dir.join("common");
    if has_static_lib(&common_dir, "common") {
        println!(
            "cargo:rustc-link-search=native={}",
            common_dir.to_string_lossy()
        );
        println!("cargo:rustc-link-lib=static=common");
    }

    let httplib_dir = build_dir.join("vendor").join("cpp-httplib");
    if has_static_lib(&httplib_dir, "cpp-httplib") {
        println!(
            "cargo:rustc-link-search=native={}",
            httplib_dir.to_string_lossy()
        );
        println!("cargo:rustc-link-lib=static=cpp-httplib");
        #[cfg(target_os = "macos")]
        {
            for candidate in [
                "/opt/homebrew/opt/openssl@3/lib",
                "/usr/local/opt/openssl@3/lib",
                "/opt/homebrew/opt/openssl/lib",
                "/usr/local/opt/openssl/lib",
            ] {
                if Path::new(candidate).is_dir() {
                    println!("cargo:rustc-link-search=native={candidate}");
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=ssl");
        println!("cargo:rustc-link-lib=dylib=crypto");
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=framework=Security");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_PREBUILT_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-changed=src/autocommit_common_bridge.cpp");
    println!("cargo:rerun-if-changed=src/autocommit_common_bridge.h");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let source_dir = env::var_os("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../third_party/llama.cpp"));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));

    if let Some(archive_name) = should_use_prebuilt(&source_dir) {
        let target = env::var("TARGET").expect("missing TARGET");
        println!(
            "cargo:warning=using prebuilt llama.cpp release {} for target {}",
            LLAMA_CPP_RELEASE_TAG, target
        );

        let (lib_dir, release_source_dir) = resolve_prebuilt(&out_dir, archive_name);

        // Use the source from the prebuilt release for headers (ensures ABI
        // compatibility with the prebuilt shared libraries), falling back to
        // the submodule source if the release source couldn't be downloaded.
        let prebuilt_source_dir = if release_source_dir.join("CMakeLists.txt").is_file() {
            release_source_dir
        } else {
            source_dir
        };
        generate_bindings(&prebuilt_source_dir, &out_dir);
        build_common_bridge(&prebuilt_source_dir, &manifest_dir, true);
        emit_prebuilt_link_deps(&lib_dir, &target);
    } else {
        build_from_source(&source_dir, &out_dir, &manifest_dir);
    }
}
