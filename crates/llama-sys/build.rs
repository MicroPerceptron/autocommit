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
        .arg("-DLLAMA_BUILD_COMMON=ON")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF");
    if cfg!(target_os = "macos") {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "11.0".to_string());
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

        if !name.starts_with("lib") || !name.ends_with(".a") {
            continue;
        }

        let lib_name = &name[3..name.len() - 2];
        if !lib_name.is_empty() {
            println!("cargo:rustc-link-lib=static={lib_name}");
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

fn emit_common_link_deps(build_dir: &Path) {
    let common_dir = build_dir.join("common");
    let common_lib = common_dir.join("libcommon.a");
    if common_lib.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            common_dir.to_string_lossy()
        );
        println!("cargo:rustc-link-lib=static=common");
    }

    let httplib_dir = build_dir.join("vendor").join("cpp-httplib");
    let httplib_lib = httplib_dir.join("libcpp-httplib.a");
    if httplib_lib.exists() {
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
