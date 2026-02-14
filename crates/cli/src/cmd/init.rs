#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache::{
    RepoKvMetadata, discover_repo_kv_paths, ensure_cache_dir, write_metadata,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut model_path: Option<String> = None;
    let mut runtime_profile = "auto".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                let path = args
                    .get(i + 1)
                    .ok_or_else(|| "--model-path requires a path".to_string())?;
                model_path = Some(path.clone());
                i += 1;
            }
            "--profile" => {
                let profile = args
                    .get(i + 1)
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                runtime_profile = profile.clone();
                i += 1;
            }
            flag => return Err(format!("unknown init option: {flag}")),
        }
        i += 1;
    }

    if let Some(path) = model_path {
        // SAFETY: this CLI is single-threaded for command setup and sets env before runtime init.
        unsafe {
            std::env::set_var("AUTOCOMMIT_EMBED_MODEL", path);
        }
    }

    #[cfg(feature = "llama-native")]
    {
        run_native(&runtime_profile)
    }

    #[cfg(not(feature = "llama-native"))]
    {
        let _ = runtime_profile;
        Err("init requires llama-native feature at build time".to_string())
    }
}

#[cfg(feature = "llama-native")]
fn run_native(runtime_profile: &str) -> Result<String, String> {
    let paths = discover_repo_kv_paths()
        .map_err(|err| format!("failed to resolve repository paths: {err}"))?;

    ensure_cache_dir(&paths).map_err(|err| format!("failed to prepare cache directory: {err}"))?;

    let engine = llama_runtime::Engine::new_with_generation_cache(
        runtime_profile,
        Some(paths.generation_state.clone()),
    )
    .map_err(|err| format!("runtime init failed: {err}"))?;

    engine
        .warmup_generation_cache()
        .map_err(|err| format!("runtime warmup failed: {err}"))?;

    let metadata = RepoKvMetadata::new(runtime_profile, engine.configured_model_path());
    write_metadata(&paths, &metadata)
        .map_err(|err| format!("failed to write cache metadata: {err}"))?;

    let model = engine
        .configured_model_path()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unset>".to_string());

    Ok(format!(
        "initialized per-repo KV cache\nrepo={}\ngit_dir={}\nstate={}\nmetadata={}\nprofile={}\nmodel={}\n",
        paths.repo_root.display(),
        paths.git_dir.display(),
        paths.generation_state.display(),
        paths.metadata.display(),
        runtime_profile,
        model
    ))
}
