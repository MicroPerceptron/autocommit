use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command, step: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to run cmake during {step}: {err}"));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "cmake {step} failed with status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
    }
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
    println!("cargo:rerun-if-env-changed=GGML_SRC_DIR");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let source_dir = env::var_os("GGML_SRC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../third_party/llama.cpp/ggml"));

    let cmake_lists = source_dir.join("CMakeLists.txt");
    assert!(
        cmake_lists.is_file(),
        "ggml source not found at {} (set GGML_SRC_DIR to override)",
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
    let build_dir = out_dir.join("ggml-cmake-build");
    let install_dir = out_dir.join("ggml-cmake-install");

    fs::create_dir_all(&build_dir).expect("failed to create cmake build dir");
    fs::create_dir_all(&install_dir).expect("failed to create cmake install dir");

    generate_bindings(&source_dir, &out_dir);

    let profile = profile_name();

    // Use a wrapper CMakeLists.txt so ggml is built as a subdirectory
    // (not standalone). Standalone mode requires ggml.pc.in which may not
    // exist in all versions of the submodule.
    let wrapper_dir = manifest_dir.join("cmake");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&wrapper_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DGGML_SOURCE_DIR={}", source_dir.to_string_lossy()))
        .arg(format!("-DCMAKE_BUILD_TYPE={profile}"))
        .arg(format!(
            "-DCMAKE_INSTALL_PREFIX={}",
            install_dir.to_string_lossy()
        ))
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg("-DGGML_OPENMP=OFF");

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

    emit_link_search_paths(&install_dir);

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");

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
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    println!(
        "cargo:rustc-env=GGML_BUILD_DIR={}",
        build_dir.to_string_lossy()
    );
    println!(
        "cargo:rustc-env=GGML_INSTALL_DIR={}",
        install_dir.to_string_lossy()
    );
}

fn generate_bindings(source_dir: &Path, out_dir: &Path) {
    let include_dir = source_dir.join("include");
    let ggml_h = include_dir.join("ggml.h");
    let gguf_h = include_dir.join("gguf.h");

    assert!(
        ggml_h.is_file(),
        "ggml.h not found at {}",
        ggml_h.display()
    );

    let mut builder = bindgen::Builder::default()
        .header(ggml_h.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.to_string_lossy()))
        .allowlist_function("^ggml_.*")
        .allowlist_function("^gguf_.*")
        .allowlist_type("^ggml_.*")
        .allowlist_type("^gguf_.*")
        .allowlist_var("^GGML_.*")
        .allowlist_var("^GGUF_.*")
        .blocklist_type("^FILE$")
        .blocklist_function("ggml_backend_dev_memory")
        .raw_line("pub type FILE = std::os::raw::c_void;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate_comments(false);

    if gguf_h.is_file() {
        builder = builder.header(gguf_h.to_string_lossy());
    }

    let bindings = builder
        .generate()
        .unwrap_or_else(|err| panic!("bindgen failed: {err}"));

    let out_file = out_dir.join("ggml_bindings.rs");
    bindings
        .write_to_file(&out_file)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_file.display()));
}

fn emit_link_search_paths(install_dir: &Path) {
    let candidates = [
        install_dir.join("lib"),
        install_dir.join("lib64"),
    ];

    for path in candidates {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
            emit_link_libs_from_dir(&path);
        }
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

        // Unix: libfoo.a
        if name.starts_with("lib") && name.ends_with(".a") {
            let lib_name = &name[3..name.len() - 2];
            if !lib_name.is_empty() {
                println!("cargo:rustc-link-lib=static={lib_name}");
            }
        }
    }
}
