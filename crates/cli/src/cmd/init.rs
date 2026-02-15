#[cfg(feature = "llama-native")]
use dialoguer::Input;
#[cfg(feature = "llama-native")]
use dialoguer::Select;
#[cfg(feature = "llama-native")]
use dialoguer::console::Term;
#[cfg(feature = "llama-native")]
use dialoguer::theme::ColorfulTheme;
#[cfg(feature = "llama-native")]
use std::io::IsTerminal;
#[cfg(feature = "llama-native")]
use std::path::PathBuf;

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache::{
    RepoKvMetadata, discover_repo_kv_paths, ensure_cache_dir, write_metadata,
};
#[cfg(feature = "llama-native")]
use crate::cmd::{commit_policy, commit_policy::CommitPolicy};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut model_path: Option<String> = None;
    let mut model_hf_repo: Option<String> = None;
    let mut model_cache_dir: Option<String> = None;
    let mut list_cached_models = false;
    let mut runtime_profile = "auto".to_string();
    let mut assume_yes = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--yes" | "-y" => assume_yes = true,
            "--model-path" => {
                let path = args
                    .get(i + 1)
                    .ok_or_else(|| "--model-path requires a path".to_string())?;
                model_path = Some(path.clone());
                i += 1;
            }
            "--hf-repo" => {
                let repo = args
                    .get(i + 1)
                    .ok_or_else(|| "--hf-repo requires a value".to_string())?;
                model_hf_repo = Some(repo.clone());
                i += 1;
            }
            "--cache-dir" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "--cache-dir requires a value".to_string())?;
                model_cache_dir = Some(value.clone());
                i += 1;
            }
            "--list-cached-models" | "--cache-list" => list_cached_models = true,
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

    if model_path.is_some() && model_hf_repo.is_some() {
        return Err("use either `--model-path` or `--hf-repo`, not both".to_string());
    }
    #[cfg(not(feature = "llama-native"))]
    let _ = (&model_cache_dir, list_cached_models);

    #[cfg(feature = "llama-native")]
    {
        run_native(
            &runtime_profile,
            assume_yes,
            model_path,
            model_hf_repo,
            model_cache_dir,
            list_cached_models,
        )
    }

    #[cfg(not(feature = "llama-native"))]
    {
        let _ = (runtime_profile, assume_yes);
        Err("init requires llama-native feature at build time".to_string())
    }
}

#[cfg(feature = "llama-native")]
fn run_native(
    runtime_profile: &str,
    assume_yes: bool,
    mut model_path: Option<String>,
    mut model_hf_repo: Option<String>,
    mut model_cache_dir: Option<String>,
    list_cached_models: bool,
) -> Result<String, String> {
    let paths = discover_repo_kv_paths()
        .map_err(|err| format!("failed to resolve repository paths: {err}"))?;

    ensure_cache_dir(&paths).map_err(|err| format!("failed to prepare cache directory: {err}"))?;
    let interactive = !assume_yes && std::io::stdin().is_terminal() && Term::stderr().is_term();

    if list_cached_models {
        let cache_dir = model_cache_dir
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from);
        let listing = llama_runtime::list_cached_models(cache_dir)
            .map_err(|err| format!("failed to list cached models: {err}"))?;
        return Ok(format_cached_models(&listing));
    }

    if interactive && model_path.is_none() && model_hf_repo.is_none() {
        let (selected_path, selected_hf_repo, selected_cache_dir) = prompt_model_selection()?;
        model_path = selected_path;
        model_hf_repo = selected_hf_repo;
        model_cache_dir = selected_cache_dir;
    }

    let model_config = llama_runtime::ModelConfig::from_explicit(
        model_path.as_deref().map(expand_tilde).map(PathBuf::from),
        model_hf_repo.clone(),
        model_cache_dir
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from),
    )
    .with_default_hf_if_unset();

    let engine = llama_runtime::Engine::new_with_generation_cache_and_model(
        runtime_profile,
        Some(paths.generation_state.clone()),
        model_config.clone(),
    )
    .map_err(|err| format!("runtime init failed: {err}"))?;

    engine
        .warmup_generation_cache()
        .map_err(|err| format!("runtime warmup failed: {err}"))?;

    let model_path = model_config
        .local_path
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());
    let model_hf_repo = model_config.hf_repo.clone();
    let model_cache_dir = model_config
        .cache_dir
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());

    let mut metadata = RepoKvMetadata::new(
        runtime_profile,
        model_path.clone(),
        model_hf_repo.clone(),
        model_cache_dir.clone(),
    );
    if interactive {
        let outcome = commit_policy::prompt_commit_policy_setup(None, true)?;
        metadata.commit_policy = outcome.policy;
        metadata.commit_policy_configured = true;
    } else {
        metadata.commit_policy = CommitPolicy::default();
        metadata.commit_policy_configured = false;
    }

    write_metadata(&paths, &metadata)
        .map_err(|err| format!("failed to write cache metadata: {err}"))?;

    let model = model_path
        .map(|path| format!("local:{path}"))
        .or_else(|| model_hf_repo.map(|repo| format!("hf:{repo}")))
        .unwrap_or_else(|| "<unset>".to_string());
    let cache_dir = model_cache_dir.unwrap_or_else(|| "<default llama.cpp cache>".to_string());
    let policy_summary = if metadata.commit_policy_configured {
        commit_policy::policy_summary(&metadata.commit_policy)
    } else {
        "not configured (will prompt on first interactive commit)".to_string()
    };

    Ok(format!(
        "initialized per-repo KV cache\nrepo={}\ngit_dir={}\nstate={}\nmetadata={}\nprofile={}\nmodel={}\nmodel_cache={}\ncommit_policy={}\n",
        paths.repo_root.display(),
        paths.git_dir.display(),
        paths.generation_state.display(),
        paths.metadata.display(),
        runtime_profile,
        model,
        cache_dir,
        policy_summary
    ))
}

#[cfg(feature = "llama-native")]
fn format_cached_models(listing: &llama_runtime::CachedModelList) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "model cache directory: {}\n",
        listing.cache_dir.display()
    ));
    out.push_str(&format!(
        "number of models in cache: {}\n",
        listing.models.len()
    ));
    for (idx, model) in listing.models.iter().enumerate() {
        out.push_str(&format!("{:4}. {model}\n", idx + 1));
    }
    out
}

#[cfg(feature = "llama-native")]
fn prompt_model_selection() -> Result<(Option<String>, Option<String>, Option<String>), String> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();

    let source_options = ["Hugging Face model (recommended)", "Local model path"];
    let source_selection = Select::with_theme(&theme)
        .with_prompt("Model source")
        .items(source_options)
        .default(0)
        .interact_on_opt(&term)
        .map_err(|err| format!("failed to read model source selection: {err}"))?;

    match source_selection {
        Some(1) => {
            let path = Input::<String>::with_theme(&theme)
                .with_prompt("Local model path")
                .interact_text_on(&term)
                .map_err(|err| format!("failed to read model path: {err}"))?;
            Ok((Some(path), None, None))
        }
        Some(0) | None => {
            let repo_options = [
                ("Qwen3 1.7B", "ggml-org/Qwen3-1.7B-GGUF"),
                ("Qwen3 4B", "ggml-org/Qwen3-4B-GGUF"),
                ("Qwen3 8B", "ggml-org/Qwen3-8B-GGUF"),
                ("Gemma 3 1B IT", "ggml-org/gemma-3-1b-it-GGUF"),
                ("Gemma 3n E2B IT", "ggml-org/gemma-3n-E2B-it-GGUF"),
                ("Custom HF repo", ""),
            ];
            let repo_labels = repo_options
                .iter()
                .map(|(label, _)| *label)
                .collect::<Vec<_>>();
            let repo_selection = Select::with_theme(&theme)
                .with_prompt("Model family")
                .items(&repo_labels)
                .default(0)
                .interact_on_opt(&term)
                .map_err(|err| format!("failed to read model family selection: {err}"))?
                .unwrap_or(0);

            let base_repo = if repo_selection == repo_options.len() - 1 {
                Input::<String>::with_theme(&theme)
                    .with_prompt("HF repo (<org>/<model> or <org>/<model>:tag)")
                    .interact_text_on(&term)
                    .map_err(|err| format!("failed to read custom HF repo: {err}"))?
            } else {
                repo_options[repo_selection].1.to_string()
            };
            let hf_repo = if base_repo.contains(':') {
                base_repo
            } else {
                let gemma_3n = base_repo.contains("gemma-3n-E2B");
                let quant_options: Vec<&str> = if gemma_3n {
                    vec!["Q8_0 (recommended)", "f16"]
                } else {
                    vec!["Q4_K_M (recommended)", "Q8_0", "f16"]
                };
                let quant_selection = Select::with_theme(&theme)
                    .with_prompt("Quantization")
                    .items(&quant_options)
                    .default(0)
                    .interact_on_opt(&term)
                    .map_err(|err| format!("failed to read quant selection: {err}"))?
                    .unwrap_or(0);
                let quant = if gemma_3n {
                    match quant_selection {
                        1 => "f16",
                        _ => "Q8_0",
                    }
                } else {
                    match quant_selection {
                        1 => "Q8_0",
                        2 => "f16",
                        _ => "Q4_K_M",
                    }
                };
                format!("{base_repo}:{quant}")
            };

            let use_custom_cache = dialoguer::Confirm::with_theme(&theme)
                .with_prompt("Use a custom model cache directory?")
                .default(false)
                .interact_on(&term)
                .map_err(|err| format!("failed to read cache directory preference: {err}"))?;
            let cache_dir = if use_custom_cache {
                let input = Input::<String>::with_theme(&theme)
                    .with_prompt("Model cache directory")
                    .interact_text_on(&term)
                    .map_err(|err| format!("failed to read model cache directory: {err}"))?;
                Some(input)
            } else {
                None
            };

            Ok((None, Some(hf_repo), cache_dir))
        }
        Some(_) => Err("invalid model source selection".to_string()),
    }
}

#[cfg(feature = "llama-native")]
fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}
