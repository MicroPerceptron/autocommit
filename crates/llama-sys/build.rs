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

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");

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
        .arg("-DLLAMA_BUILD_COMMON=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF");
    run(&mut configure, "configure");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile)
        .arg("--parallel");
    if let Ok(jobs) = env::var("CARGO_BUILD_JOBS") {
        if !jobs.is_empty() {
            build.arg(jobs);
        }
    }
    run(&mut build, "build");

    let mut install = Command::new("cmake");
    install
        .arg("--install")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile);
    run(&mut install, "install");

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

fn emit_link_search_paths(install_dir: &Path) {
    let candidates = [
        install_dir.join("lib"),
        install_dir.join("lib64"),
        install_dir.join("bin"),
    ];

    for path in candidates {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
        }
    }
}
