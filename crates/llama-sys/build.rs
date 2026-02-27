use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Check whether `GGML_<backend>` is explicitly set (to any value).
fn is_explicitly_set(var: &str) -> bool {
    env::var(var).is_ok()
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

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-changed=src/autocommit_common_bridge.cpp");
    println!("cargo:rerun-if-changed=src/autocommit_common_bridge.h");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let source_dir = env::var_os("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../third_party/llama.cpp"));

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

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));
    let build_dir = out_dir.join("llama-cmake-build");
    let install_dir = out_dir.join("llama-cmake-install");

    fs::create_dir_all(&build_dir).expect("failed to create cmake build dir");
    fs::create_dir_all(&install_dir).expect("failed to create cmake install dir");

    generate_bindings(&source_dir, &out_dir);

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
        .arg(&source_dir)
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
        .arg("-DLLAMA_BUILD_SERVER=OFF");
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

    build_common_bridge(&source_dir, &manifest_dir);
    emit_common_link_deps(&build_dir);
    emit_link_search_paths(&install_dir);
    if use_cuda {
        emit_cuda_link_deps();
    }
    if use_sycl {
        emit_sycl_link_deps();
    }
    if use_vulkan {
        emit_vulkan_link_deps();
    }
    // Link OpenBLAS on Linux when BLAS is enabled (non-SYCL builds).
    #[cfg(target_os = "linux")]
    if !use_sycl {
        println!("cargo:rustc-link-lib=dylib=openblas");
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

fn emit_link_search_paths(install_dir: &Path) {
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
        // llama.cpp enables OpenMP for the CPU backend on Linux by default.
        // The static archives contain GOMP symbols that need the runtime.
        println!("cargo:rustc-link-lib=dylib=gomp");
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
        // Windows: static cudart, dynamic cublas (no static cublas on Windows)
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

    // SYCL runtime and MKL are dynamically linked
    println!("cargo:rustc-link-lib=dylib=sycl");
    println!("cargo:rustc-link-lib=dylib=mkl_sycl_blas");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=dylib=mkl_tbb_thread");
    println!("cargo:rustc-link-lib=dylib=mkl_core");

    // Intel compiler runtime libraries — icpx injects calls to SVML vectorized
    // math intrinsics (__svml_*), Intel fast memory ops (_intel_fast_memcpy/memset),
    // and Intel math functions (__libm_sse2_sincosf) into compiled objects.
    println!("cargo:rustc-link-lib=dylib=svml");
    println!("cargo:rustc-link-lib=dylib=irc");
    println!("cargo:rustc-link-lib=dylib=imf");

    // Intel OpenMP runtime — icpx uses __kmpc_* symbols (Intel's OpenMP ABI)
    // instead of GOMP_* (GNU's). libiomp5 provides these.
    println!("cargo:rustc-link-lib=dylib=iomp5");
}

fn emit_vulkan_link_deps() {
    // Vulkan links dynamically via the Vulkan loader (provided by GPU drivers)
    if cfg!(target_os = "windows") {
        // On Windows the loader import library is typically vulkan-1.lib
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    } else {
        // On Unix-like systems the loader is usually libvulkan.so
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

fn build_common_bridge(source_dir: &Path, manifest_dir: &Path) {
    let bridge_cpp = manifest_dir
        .join("src")
        .join("autocommit_common_bridge.cpp");
    let common_dir = source_dir.join("common");
    let include_dir = source_dir.join("include");
    let ggml_include = source_dir.join("ggml").join("include");
    let vendor_dir = source_dir.join("vendor");

    cc::Build::new()
        .cpp(true)
        .file(&bridge_cpp)
        .include(&common_dir)
        .include(&include_dir)
        .include(&ggml_include)
        .include(&vendor_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-function")
        .compile("autocommit_common_bridge");
}

fn has_static_lib(dir: &Path, name: &str) -> bool {
    // Check for Unix/MinGW (libfoo.a) and MSVC (foo.lib)
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
