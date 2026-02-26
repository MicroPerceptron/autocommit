#[cfg(feature = "llama-native")]
use dialoguer::console::Term;
#[cfg(feature = "llama-native")]
use dialoguer::theme::ColorfulTheme;
#[cfg(feature = "llama-native")]
use dialoguer::Input;
#[cfg(feature = "llama-native")]
use dialoguer::Select;
#[cfg(feature = "llama-native")]
use std::io::IsTerminal;
#[cfg(feature = "llama-native")]
use std::path::PathBuf;

use clap::Parser;

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache::{
    discover_repo_kv_paths, ensure_cache_dir, write_metadata, RepoKvMetadata,
};
#[cfg(feature = "llama-native")]
use crate::cmd::{commit_policy, commit_policy::CommitPolicy};
#[cfg(feature = "llama-native")]
use crate::path_util::expand_tilde;

pub fn run(args: &[String]) -> Result<String, String> {
    let parsed = match InitArgs::parse_from(args)? {
        ParseOutcome::Continue(parsed) => parsed,
        ParseOutcome::EarlyExit(text) => return Ok(text),
    };
    let model_path = parsed.model_path;
    let model_hf_repo = parsed.hf_repo;
    let model_cache_dir = parsed.cache_dir;
    let list_cached_models = parsed.list_cached_models;
    let runtime_profile = parsed.profile.unwrap_or_else(|| "auto".to_string());
    let assume_yes = parsed.yes;

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

enum ParseOutcome<T> {
    Continue(T),
    EarlyExit(String),
}

#[derive(Parser, Debug)]
#[command(
    name = "autocommit init",
    about = "Initialize per-repository runtime cache and policy settings"
)]
struct InitArgs {
    /// Accept defaults and skip interactive prompts
    #[arg(long, short = 'y')]
    yes: bool,
    /// Explicit local model path (`.gguf`)
    #[arg(long = "model-path", value_name = "PATH")]
    model_path: Option<String>,
    /// Hugging Face model repo (`org/model` or `org/model:file`)
    #[arg(long = "hf-repo", value_name = "REPO")]
    hf_repo: Option<String>,
    /// Override llama.cpp model cache directory
    #[arg(long = "cache-dir", value_name = "PATH")]
    cache_dir: Option<String>,
    /// List models in the configured cache and exit
    #[arg(long = "list-cached-models", alias = "cache-list")]
    list_cached_models: bool,
    /// Runtime profile (`auto`, etc.)
    #[arg(long = "profile", value_name = "PROFILE")]
    profile: Option<String>,
}

impl InitArgs {
    fn parse_from(args: &[String]) -> Result<ParseOutcome<Self>, String> {
        let argv = std::iter::once("autocommit init".to_string()).chain(args.iter().cloned());
        match Self::try_parse_from(argv) {
            Ok(parsed) => Ok(ParseOutcome::Continue(parsed)),
            Err(err) => {
                use clap::error::ErrorKind;
                match err.kind() {
                    ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                        Ok(ParseOutcome::EarlyExit(err.to_string()))
                    }
                    _ => Err(err.to_string()),
                }
            }
        }
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
type ModelSelection = (Option<String>, Option<String>, Option<String>);

fn prompt_model_selection() -> Result<ModelSelection, String> {
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
                ("Gemma 3n E2B IT", "ggml-org/gemma-3n-E2B-it-GGUF"),
                ("Qwen3 1.7B", "ggml-org/Qwen3-1.7B-GGUF"),
                ("Qwen3 4B", "ggml-org/Qwen3-4B-GGUF"),
                ("Qwen3 8B", "ggml-org/Qwen3-8B-GGUF"),
                ("Gemma 3 1B IT", "ggml-org/gemma-3-1b-it-GGUF"),
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
