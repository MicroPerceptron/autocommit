#[cfg(feature = "llama-native")]
use std::io::IsTerminal;
#[cfg(feature = "llama-native")]
use std::path::PathBuf;

use clap::Parser;
#[cfg(feature = "llama-native")]
use dialoguer::console::Term;
#[cfg(feature = "llama-native")]
use dialoguer::theme::ColorfulTheme;
#[cfg(feature = "llama-native")]
use dialoguer::{Confirm, Input, Select};

#[cfg(feature = "llama-native")]
use crate::cmd::commit_policy;
#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache::{
    RepoKvMetadata, RepoKvPaths, discover_repo_kv_paths, ensure_cache_dir, read_metadata,
    write_metadata,
};
#[cfg(feature = "llama-native")]
use crate::path_util::expand_tilde;

pub fn run(args: &[String]) -> Result<String, String> {
    let parsed = match ConfigArgs::parse_from(args)? {
        ParseOutcome::Continue(parsed) => parsed,
        ParseOutcome::EarlyExit(text) => return Ok(text),
    };

    if parsed.model_path.is_some() && parsed.hf_repo.is_some() {
        return Err("use either `--model-path` or `--hf-repo`, not both".to_string());
    }
    if parsed.default_model && (parsed.model_path.is_some() || parsed.hf_repo.is_some()) {
        return Err(
            "`--default-model` cannot be combined with `--model-path` or `--hf-repo`".to_string(),
        );
    }
    if parsed.cache_dir.is_some() && parsed.clear_cache_dir {
        return Err("use either `--cache-dir` or `--clear-cache-dir`, not both".to_string());
    }
    if parsed.configure_commit_policy && parsed.reset_commit_policy {
        return Err(
            "use either `--configure-commit-policy` or `--reset-commit-policy`, not both"
                .to_string(),
        );
    }

    #[cfg(not(feature = "llama-native"))]
    let _ = (
        parsed.yes,
        &parsed.profile,
        &parsed.cache_dir,
        parsed.clear_cache_dir,
        parsed.default_model,
        parsed.configure_commit_policy,
        parsed.reset_commit_policy,
        parsed.show,
        parsed.list_cached_models,
    );

    #[cfg(feature = "llama-native")]
    {
        run_native(parsed)
    }

    #[cfg(not(feature = "llama-native"))]
    {
        Err("config requires llama-native feature at build time".to_string())
    }
}

enum ParseOutcome<T> {
    Continue(T),
    EarlyExit(String),
}

#[derive(Parser, Debug)]
#[command(
    name = "autocommit config",
    about = "View and update per-repository runtime and commit policy settings"
)]
struct ConfigArgs {
    /// Accept defaults and skip interactive prompts
    #[arg(long, short = 'y')]
    yes: bool,
    /// Show current per-repo configuration
    #[arg(long)]
    show: bool,
    /// List models in the configured cache and exit
    #[arg(long = "list-cached-models", alias = "cache-list")]
    list_cached_models: bool,
    /// Runtime profile (`auto`, etc.)
    #[arg(long = "profile", value_name = "PROFILE")]
    profile: Option<String>,
    /// Explicit local model path (`.gguf`)
    #[arg(long = "model-path", value_name = "PATH")]
    model_path: Option<String>,
    /// Hugging Face model repo (`org/model` or `org/model:file`)
    #[arg(long = "hf-repo", value_name = "REPO")]
    hf_repo: Option<String>,
    /// Clear explicit model source and use runtime default model selection
    #[arg(long = "default-model")]
    default_model: bool,
    /// Override llama.cpp model cache directory
    #[arg(long = "cache-dir", value_name = "PATH")]
    cache_dir: Option<String>,
    /// Clear explicit model cache override and use llama.cpp default cache path
    #[arg(long = "clear-cache-dir")]
    clear_cache_dir: bool,
    /// Configure commit policy interactively
    #[arg(long = "configure-commit-policy")]
    configure_commit_policy: bool,
    /// Reset commit policy to defaults
    #[arg(long = "reset-commit-policy")]
    reset_commit_policy: bool,
}

impl ConfigArgs {
    fn parse_from(args: &[String]) -> Result<ParseOutcome<Self>, String> {
        let argv = std::iter::once("autocommit config".to_string()).chain(args.iter().cloned());
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
fn run_native(parsed: ConfigArgs) -> Result<String, String> {
    let paths = discover_repo_kv_paths()
        .map_err(|err| format!("failed to resolve repository paths: {err}"))?;
    ensure_cache_dir(&paths).map_err(|err| format!("failed to prepare cache directory: {err}"))?;

    let interactive_tty = std::io::stdin().is_terminal() && Term::stderr().is_term();
    let interactive = interactive_tty && !parsed.yes;
    let mut metadata =
        read_metadata(&paths).unwrap_or_else(|| RepoKvMetadata::new("auto", None, None, None));

    if parsed.list_cached_models {
        let cache_dir = parsed
            .cache_dir
            .as_deref()
            .map(expand_tilde)
            .or_else(|| metadata.model_cache_dir.clone())
            .map(PathBuf::from);
        let listing = llama_runtime::list_cached_models(cache_dir)
            .map_err(|err| format!("failed to list cached models: {err}"))?;
        return Ok(format_cached_models(&listing));
    }

    let mut changed = false;
    changed |= apply_cli_overrides(&parsed, interactive, &mut metadata)?;

    let has_explicit_changes = has_explicit_change_flags(&parsed);
    if !has_explicit_changes && !parsed.show {
        if interactive {
            match prompt_interactive_edits(&mut metadata)? {
                InteractiveConfigResult::Saved { changed: edited } => {
                    changed |= edited;
                }
                InteractiveConfigResult::Canceled => {
                    return Ok("configuration canceled by user\n".to_string());
                }
            }
        } else {
            return Err("no config changes requested; use flags or run interactively".to_string());
        }
    }

    if changed {
        write_metadata(&paths, &metadata)
            .map_err(|err| format!("failed to write cache metadata: {err}"))?;
        return Ok(render_config_summary(&paths, &metadata, "updated"));
    }

    let status = if parsed.show { "current" } else { "unchanged" };
    Ok(render_config_summary(&paths, &metadata, status))
}

#[cfg(feature = "llama-native")]
fn has_explicit_change_flags(parsed: &ConfigArgs) -> bool {
    parsed.profile.is_some()
        || parsed.model_path.is_some()
        || parsed.hf_repo.is_some()
        || parsed.default_model
        || parsed.cache_dir.is_some()
        || parsed.clear_cache_dir
        || parsed.configure_commit_policy
        || parsed.reset_commit_policy
}

#[cfg(feature = "llama-native")]
fn apply_cli_overrides(
    parsed: &ConfigArgs,
    interactive: bool,
    metadata: &mut RepoKvMetadata,
) -> Result<bool, String> {
    let mut changed = false;

    if let Some(profile) = parsed.profile.as_deref() {
        let profile = profile.trim();
        if profile.is_empty() {
            return Err("`--profile` cannot be empty".to_string());
        }
        if metadata.profile != profile {
            metadata.profile = profile.to_string();
            changed = true;
        }
    }

    if parsed.default_model {
        if metadata.model_path.take().is_some() {
            changed = true;
        }
        if metadata.model_hf_repo.take().is_some() {
            changed = true;
        }
    } else if let Some(path) = parsed.model_path.as_deref() {
        let normalized = expand_tilde(path);
        if metadata.model_path.as_deref() != Some(normalized.as_str()) {
            metadata.model_path = Some(normalized);
            changed = true;
        }
        if metadata.model_hf_repo.take().is_some() {
            changed = true;
        }
    } else if let Some(repo) = parsed.hf_repo.as_deref() {
        let normalized = repo.trim();
        if normalized.is_empty() {
            return Err("`--hf-repo` cannot be empty".to_string());
        }
        if metadata.model_hf_repo.as_deref() != Some(normalized) {
            metadata.model_hf_repo = Some(normalized.to_string());
            changed = true;
        }
        if metadata.model_path.take().is_some() {
            changed = true;
        }
    }

    if let Some(cache_dir) = parsed.cache_dir.as_deref() {
        let normalized = expand_tilde(cache_dir);
        if metadata.model_cache_dir.as_deref() != Some(normalized.as_str()) {
            metadata.model_cache_dir = Some(normalized);
            changed = true;
        }
    } else if parsed.clear_cache_dir && metadata.model_cache_dir.take().is_some() {
        changed = true;
    }

    if parsed.reset_commit_policy {
        metadata.commit_policy = commit_policy::CommitPolicy::default();
        metadata.commit_policy_configured = false;
        changed = true;
    }

    if parsed.configure_commit_policy {
        if !interactive {
            return Err(
                "`--configure-commit-policy` requires an interactive terminal without `--yes`"
                    .to_string(),
            );
        }
        let outcome =
            commit_policy::prompt_commit_policy_setup(Some(&metadata.commit_policy), true)?;
        metadata.commit_policy = outcome.policy;
        metadata.commit_policy_configured = true;
        changed = true;
    }

    Ok(changed)
}

#[cfg(feature = "llama-native")]
enum InteractiveConfigResult {
    Saved { changed: bool },
    Canceled,
}

#[cfg(feature = "llama-native")]
fn prompt_interactive_edits(
    metadata: &mut RepoKvMetadata,
) -> Result<InteractiveConfigResult, String> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();
    let mut changed = false;

    loop {
        let model_source = model_source_label(metadata);
        let cache_dir = metadata
            .model_cache_dir
            .clone()
            .unwrap_or_else(|| "<default llama.cpp cache>".to_string());
        let policy = if metadata.commit_policy_configured {
            commit_policy::policy_summary(&metadata.commit_policy)
        } else {
            "not configured".to_string()
        };

        let options = vec![
            format!("Show current settings"),
            format!("Set runtime profile ({})", metadata.profile),
            format!("Set model source ({model_source})"),
            format!("Set model cache directory ({cache_dir})"),
            format!("Configure commit policy ({policy})"),
            "Reset commit policy to defaults".to_string(),
            "Save and exit".to_string(),
            "Cancel".to_string(),
        ];

        let selection = Select::with_theme(&theme)
            .with_prompt("Config action")
            .items(&options)
            .default(0)
            .interact_on_opt(&term)
            .map_err(|err| format!("failed to read config action: {err}"))?;

        match selection {
            Some(0) => {
                println!();
                println!("{}", render_compact_config(metadata));
                println!();
            }
            Some(1) => {
                let next = Input::<String>::with_theme(&theme)
                    .with_prompt("Runtime profile")
                    .with_initial_text(metadata.profile.clone())
                    .interact_text_on(&term)
                    .map_err(|err| format!("failed to read runtime profile: {err}"))?;
                let trimmed = next.trim();
                if trimmed.is_empty() {
                    println!("profile cannot be empty");
                } else if trimmed != metadata.profile {
                    metadata.profile = trimmed.to_string();
                    changed = true;
                }
            }
            Some(2) => {
                let (model_path, model_hf_repo, cache_dir) = prompt_model_selection()?;
                if metadata.model_path != model_path {
                    metadata.model_path = model_path;
                    changed = true;
                }
                if metadata.model_hf_repo != model_hf_repo {
                    metadata.model_hf_repo = model_hf_repo;
                    changed = true;
                }
                if cache_dir.is_some() && metadata.model_cache_dir != cache_dir {
                    metadata.model_cache_dir = cache_dir;
                    changed = true;
                }
            }
            Some(3) => {
                let next = prompt_cache_directory_edit(metadata.model_cache_dir.as_deref())?;
                if metadata.model_cache_dir != next {
                    metadata.model_cache_dir = next;
                    changed = true;
                }
            }
            Some(4) => {
                let outcome =
                    commit_policy::prompt_commit_policy_setup(Some(&metadata.commit_policy), true)?;
                metadata.commit_policy = outcome.policy;
                metadata.commit_policy_configured = true;
                changed = true;
            }
            Some(5) => {
                let confirm = Confirm::with_theme(&theme)
                    .with_prompt("Reset commit policy to defaults?")
                    .default(false)
                    .interact_on(&term)
                    .map_err(|err| format!("failed to read reset confirmation: {err}"))?;
                if confirm {
                    metadata.commit_policy = commit_policy::CommitPolicy::default();
                    metadata.commit_policy_configured = false;
                    changed = true;
                }
            }
            Some(6) | None => return Ok(InteractiveConfigResult::Saved { changed }),
            Some(7) => return Ok(InteractiveConfigResult::Canceled),
            Some(_) => return Err("invalid config action selection".to_string()),
        }
    }
}

#[cfg(feature = "llama-native")]
fn prompt_cache_directory_edit(current: Option<&str>) -> Result<Option<String>, String> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();
    let options = [
        "Keep current value",
        "Use default cache directory",
        "Set custom cache directory",
    ];
    let selection = Select::with_theme(&theme)
        .with_prompt("Model cache directory")
        .items(options)
        .default(0)
        .interact_on_opt(&term)
        .map_err(|err| format!("failed to read model cache directory action: {err}"))?;

    match selection {
        Some(0) | None => Ok(current.map(ToOwned::to_owned)),
        Some(1) => Ok(None),
        Some(2) => {
            let path = Input::<String>::with_theme(&theme)
                .with_prompt("Model cache directory")
                .with_initial_text(current.unwrap_or_default())
                .interact_text_on(&term)
                .map_err(|err| format!("failed to read model cache directory: {err}"))?;
            Ok(Some(expand_tilde(path.trim())))
        }
        Some(_) => Err("invalid model cache action".to_string()),
    }
}

#[cfg(feature = "llama-native")]
fn model_source_label(metadata: &RepoKvMetadata) -> String {
    metadata
        .model_path
        .as_ref()
        .map(|path| format!("local:{path}"))
        .or_else(|| {
            metadata
                .model_hf_repo
                .as_ref()
                .map(|repo| format!("hf:{repo}"))
        })
        .unwrap_or_else(|| "default runtime model".to_string())
}

#[cfg(feature = "llama-native")]
fn render_compact_config(metadata: &RepoKvMetadata) -> String {
    let cache_dir = metadata
        .model_cache_dir
        .clone()
        .unwrap_or_else(|| "<default llama.cpp cache>".to_string());
    let policy = if metadata.commit_policy_configured {
        commit_policy::policy_summary(&metadata.commit_policy)
    } else {
        "not configured".to_string()
    };
    format!(
        "profile={}\nmodel={}\nmodel_cache={}\ncommit_policy={}",
        metadata.profile,
        model_source_label(metadata),
        cache_dir,
        policy
    )
}

#[cfg(feature = "llama-native")]
fn render_config_summary(paths: &RepoKvPaths, metadata: &RepoKvMetadata, status: &str) -> String {
    let policy_summary = if metadata.commit_policy_configured {
        commit_policy::policy_summary(&metadata.commit_policy)
    } else {
        "not configured (run `autocommit config --configure-commit-policy`)".to_string()
    };
    let cache_dir = metadata
        .model_cache_dir
        .clone()
        .unwrap_or_else(|| "<default llama.cpp cache>".to_string());
    format!(
        "{status} per-repo config\nrepo={}\ngit_dir={}\nstate={}\nmetadata={}\nprofile={}\nmodel={}\nmodel_cache={}\ncommit_policy={}\n",
        paths.repo_root.display(),
        paths.git_dir.display(),
        paths.generation_state.display(),
        paths.metadata.display(),
        metadata.profile,
        model_source_label(metadata),
        cache_dir,
        policy_summary,
    )
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

#[cfg(feature = "llama-native")]
fn prompt_model_selection() -> Result<ModelSelection, String> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();

    let source_options = [
        "Hugging Face model (recommended)",
        "Local model path",
        "Use runtime default",
    ];
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
            Ok((Some(expand_tilde(path.trim())), None, None))
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

            let use_custom_cache = Confirm::with_theme(&theme)
                .with_prompt("Use a custom model cache directory?")
                .default(false)
                .interact_on(&term)
                .map_err(|err| format!("failed to read cache directory preference: {err}"))?;
            let cache_dir = if use_custom_cache {
                let input = Input::<String>::with_theme(&theme)
                    .with_prompt("Model cache directory")
                    .interact_text_on(&term)
                    .map_err(|err| format!("failed to read model cache directory: {err}"))?;
                Some(expand_tilde(input.trim()))
            } else {
                None
            };

            Ok((None, Some(hf_repo), cache_dir))
        }
        Some(2) => Ok((None, None, None)),
        Some(_) => Err("invalid model source selection".to_string()),
    }
}
