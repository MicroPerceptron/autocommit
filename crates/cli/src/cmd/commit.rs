use std::io::{IsTerminal, Write};
#[cfg(feature = "llama-native")]
use std::path::PathBuf;
use std::process::Command;

use autocommit_core::AnalysisReport;
use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};
use dialoguer::console::{Term, style};
use dialoguer::{Confirm, Editor, Select, theme::ColorfulTheme};
use indicatif::{ProgressBar, ProgressStyle};

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;
use crate::cmd::{
    analysis_progress::AnalysisProgress, commit_policy, commit_policy::CommitPolicy, git,
    report_cache, version_bump,
};
use crate::output;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::types::{
    ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef, FileStatus,
    PartialReport, RiskReport, TypeTag,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut staged_only = false;
    let mut push = false;
    let mut dry_run = false;
    let mut json = false;
    let mut no_verify = false;
    let mut configure_commit_policy = false;
    let mut interactive_override: Option<bool> = None;
    let mut assume_yes = false;
    let mut model_path: Option<String> = None;
    let mut model_hf_repo: Option<String> = None;
    let mut model_cache_dir: Option<String> = None;
    #[cfg(feature = "llama-native")]
    let mut runtime_profile = "auto".to_string();
    #[cfg(feature = "llama-native")]
    let mut runtime_profile_overridden = false;
    #[cfg(not(feature = "llama-native"))]
    let runtime_profile = "mock".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--staged" | "-s" => staged_only = true,
            "--push" | "-p" => push = true,
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            "--no-verify" => no_verify = true,
            "--configure-commit-policy" => configure_commit_policy = true,
            "--interactive" => interactive_override = Some(true),
            "--no-interactive" => interactive_override = Some(false),
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
            "--profile" => {
                let profile = args
                    .get(i + 1)
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                #[cfg(feature = "llama-native")]
                {
                    runtime_profile = profile.clone();
                    runtime_profile_overridden = true;
                }
                #[cfg(not(feature = "llama-native"))]
                {
                    let _ = profile;
                }
                i += 1;
            }
            flag => return Err(format!("unknown commit option: {flag}")),
        }
        i += 1;
    }

    if model_path.is_some() && model_hf_repo.is_some() {
        return Err("use either `--model-path` or `--hf-repo`, not both".to_string());
    }
    #[cfg(not(feature = "llama-native"))]
    let _ = &model_cache_dir;

    let interactive = resolve_interactive_mode(interactive_override, json)?;
    let rich_interactive = interactive && Term::stderr().is_term();

    #[cfg(feature = "llama-native")]
    let repo_paths = repo_cache::maybe_discover_repo_kv_paths();
    #[cfg(feature = "llama-native")]
    let repo_metadata = repo_paths.as_ref().and_then(repo_cache::read_metadata);

    #[cfg(feature = "llama-native")]
    if model_path.is_none()
        || model_hf_repo.is_none()
        || model_cache_dir.is_none()
        || !runtime_profile_overridden
    {
        if let Some(metadata) = repo_metadata.as_ref() {
            if model_path.is_none() && model_hf_repo.is_none() {
                model_path = metadata.model_path.clone();
                model_hf_repo = metadata.model_hf_repo.clone();
            }
            if model_cache_dir.is_none() {
                model_cache_dir = metadata.model_cache_dir.clone();
            }
            if !runtime_profile_overridden && !metadata.profile.trim().is_empty() {
                runtime_profile = metadata.profile.clone();
            }
        }
    }

    #[cfg(feature = "llama-native")]
    let mut commit_policy_config = repo_metadata
        .as_ref()
        .map(|metadata| metadata.commit_policy.clone())
        .unwrap_or_default();
    #[cfg(feature = "llama-native")]
    let commit_policy_configured = repo_metadata
        .as_ref()
        .map(|metadata| metadata.commit_policy_configured)
        .unwrap_or(false);
    #[cfg(not(feature = "llama-native"))]
    let mut commit_policy_config = CommitPolicy::default();
    #[cfg(not(feature = "llama-native"))]
    let commit_policy_configured = true;

    let repo = run_step(
        rich_interactive,
        "Discovering repository",
        git::Repo::discover,
    )
    .map_err(|err| err.to_string())?;
    // Best-effort partial cache for large diffs.
    if let Some(cache_dir) = repo
        .common_git_dir()
        .join("autocommit/kv/partials")
        .to_str()
    {
        unsafe {
            std::env::set_var("AUTOCOMMIT_PARTIAL_CACHE_DIR", cache_dir);
        }
    }

    if interactive && !assume_yes && (!commit_policy_configured || configure_commit_policy) {
        let outcome = commit_policy::prompt_commit_policy_setup(
            if commit_policy_configured {
                Some(&commit_policy_config)
            } else {
                None
            },
            rich_interactive,
        )?;
        commit_policy_config = outcome.policy;

        #[cfg(feature = "llama-native")]
        if let Some(paths) = repo_paths.as_ref() {
            let effective_model_config = llama_runtime::ModelConfig::from_explicit(
                model_path.as_deref().map(expand_tilde).map(PathBuf::from),
                model_hf_repo.clone(),
                model_cache_dir
                    .as_deref()
                    .map(expand_tilde)
                    .map(PathBuf::from),
            )
            .with_default_hf_if_unset();
            let mut metadata = repo_metadata.clone().unwrap_or_else(|| {
                repo_cache::RepoKvMetadata::new(
                    &runtime_profile,
                    effective_model_config
                        .local_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    effective_model_config.hf_repo.clone(),
                    effective_model_config
                        .cache_dir
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                )
            });
            metadata.profile = runtime_profile.clone();
            metadata.model_path = effective_model_config
                .local_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned());
            metadata.model_hf_repo = effective_model_config.hf_repo.clone();
            metadata.model_cache_dir = effective_model_config
                .cache_dir
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned());
            metadata.commit_policy = commit_policy_config.clone();
            metadata.commit_policy_configured = true;
            repo_cache::write_metadata(paths, &metadata)
                .map_err(|err| format!("failed to persist commit policy: {err}"))?;
        }
    }

    let diff_text = run_step(rich_interactive, "Collecting staged/worktree diff", || {
        prepare_diff(&repo, staged_only, dry_run)
    })
    .map_err(|err| err.to_string())?;

    let diff_hash = report_cache::diff_hash(&diff_text);
    let cache_key = report_cache::cache_key("commit", runtime_profile.as_str(), &diff_hash, "1.0");
    let cache_path = report_cache::cache_path(repo.common_git_dir());

    let cached_report = report_cache::read_cached_report(&cache_path, &cache_key);

    #[cfg(feature = "llama-native")]
    let generation_state = repo_paths.map(|paths| paths.generation_state);

    #[cfg(feature = "llama-native")]
    let model_config = llama_runtime::ModelConfig::from_explicit(
        model_path.as_deref().map(expand_tilde).map(PathBuf::from),
        model_hf_repo.clone(),
        model_cache_dir
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from),
    )
    .with_default_hf_if_unset();

    #[cfg(feature = "llama-native")]
    let engine = run_step(rich_interactive, "Initializing llama runtime", || {
        llama_runtime::Engine::new_with_generation_cache_and_model(
            &runtime_profile,
            generation_state,
            model_config.clone(),
        )
    })
    .map_err(|err| format!("runtime init failed: {err}"))?;

    #[cfg(not(feature = "llama-native"))]
    let engine = run_step(rich_interactive, "Initializing analysis engine", || {
        Ok::<MockEngine, CoreError>(MockEngine)
    })
    .map_err(|err| format!("runtime init failed: {err}"))?;

    let report = if let Some(report) = cached_report {
        if rich_interactive {
            println!("[ok] Using cached analysis");
        }
        report
    } else {
        let progress = if rich_interactive {
            Some(AnalysisProgress::new(
                &diff_text,
                "Generating commit analysis",
            ))
        } else {
            None
        };

        #[cfg(feature = "llama-native")]
        if let Some(progress) = progress.as_ref() {
            engine.set_progress_callback(Some(progress.callback()));
        }

        let report = core_run(&engine, &diff_text, &AnalyzeOptions::default())
            .map_err(|err| format!("analysis failed: {err}"))?;

        #[cfg(feature = "llama-native")]
        {
            engine.set_progress_callback(None);
        }
        if let Some(progress) = progress {
            progress.finish();
        }

        let _ = report_cache::write_cached_report(&cache_path, &cache_key, &report);
        report
    };
    let embedding_bump_level = infer_embedding_bump_level(&engine, &diff_text, &report);
    let version_recommendations =
        version_bump::recommend(&repo, &diff_text, &report, embedding_bump_level);
    let approved_version_recommendations = resolve_version_bump_recommendations(
        &version_recommendations,
        interactive,
        rich_interactive,
        assume_yes,
    )?;

    if !dry_run && !approved_version_recommendations.is_empty() {
        run_step(rich_interactive, "Applying version bumps", || {
            apply_approved_version_bumps(&repo, staged_only, &approved_version_recommendations)
        })?;
    }

    if dry_run {
        if json {
            return output::json::to_pretty_json(&report).map_err(|err| err.to_string());
        }

        let composed_message = compose_commit_message(&report, &approved_version_recommendations);
        let composed_message =
            prepare_message_for_policy(&repo, &composed_message, &commit_policy_config)
                .map_err(|err| err.to_string())?;
        let mut out = String::new();
        out.push_str("dry-run: commit was not created\n");
        out.push_str(&format!("message:\n{}\n", composed_message));
        out.push_str(&output::text::render_report(&report));
        return Ok(out);
    }

    let composed_message = compose_commit_message(&report, &approved_version_recommendations);
    let final_message = if interactive && !assume_yes {
        let mut message = composed_message;
        loop {
            let selected = match prompt_for_commit_message(&message, rich_interactive)? {
                Some(message) => message,
                None => return Ok("commit canceled by user\n".to_string()),
            };
            match prepare_message_for_policy(&repo, &selected, &commit_policy_config) {
                Ok(ready) => break ready,
                Err(err) => {
                    println!(
                        "{}",
                        style(format!("commit policy validation failed: {err}")).red()
                    );
                    message = selected;
                }
            }
        }
    } else {
        prepare_message_for_policy(&repo, &composed_message, &commit_policy_config)
            .map_err(|err| err.to_string())?
    };

    ensure_signing_tool_ready(
        &repo,
        &mut commit_policy_config,
        interactive,
        rich_interactive,
        assume_yes,
    )
    .map_err(|err| err.to_string())?;

    run_step(rich_interactive, "Creating commit", || {
        commit_with_message(
            &repo,
            &final_message,
            staged_only,
            no_verify,
            &commit_policy_config,
        )
    })
    .map_err(|err| err.to_string())?;

    let should_push = if interactive && !assume_yes {
        prompt_should_push(push, rich_interactive)?
    } else {
        push
    };

    let mut push_note: Option<String> = None;
    if should_push {
        push_note = push_with_feedback(rich_interactive, &repo).map_err(|err| err.to_string())?;
    }

    if json {
        output::json::to_pretty_json(&report).map_err(|err| err.to_string())
    } else {
        let mut out = String::new();
        out.push_str("created commit with message:\n");
        out.push_str(&final_message);
        out.push('\n');
        if let Some(note) = push_note {
            out.push_str(&note);
            out.push('\n');
        }
        Ok(out)
    }
}

fn resolve_interactive_mode(
    interactive_override: Option<bool>,
    json: bool,
) -> Result<bool, String> {
    if json {
        if interactive_override == Some(true) {
            return Err("--interactive cannot be combined with --json".to_string());
        }
        return Ok(false);
    }

    let stdin_tty = std::io::stdin().is_terminal();
    let stderr_tty = Term::stderr().is_term();
    match interactive_override {
        Some(true) => {
            if !stdin_tty && !stderr_tty {
                return Err(
                    "interactive mode requested but no terminal is attached to stdin/stderr"
                        .to_string(),
                );
            }
            Ok(true)
        }
        Some(false) => Ok(false),
        None => Ok(stderr_tty),
    }
}

fn run_step<T, E, F>(interactive: bool, label: &str, action: F) -> Result<T, E>
where
    E: std::fmt::Display,
    F: FnOnce() -> Result<T, E>,
{
    let progress = if interactive {
        let progress = ProgressBar::new_spinner();
        progress.set_style(spinner_style());
        progress.set_message(label.to_string());
        progress.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(progress)
    } else {
        None
    };

    let result = action();
    if let Some(progress) = progress {
        match &result {
            Ok(_) => progress.finish_with_message(format!("[ok] {label}")),
            Err(err) => progress.abandon_with_message(format!("[error] {label}: {err}")),
        }
    }

    result
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_strings(&["-", "\\", "|", "/"])
}

fn prompt_for_commit_message(initial_message: &str, rich: bool) -> Result<Option<String>, String> {
    let mut message = initial_message.trim().to_string();
    if message.is_empty() {
        return Err("generated commit subject is empty".to_string());
    }

    if !rich {
        return prompt_for_commit_message_basic(&message);
    }

    let term = Term::stderr();
    let theme = ColorfulTheme::default();
    let options = ["Approve commit", "Edit message", "Cancel"];

    loop {
        print_commit_preview(&message, true);
        let selection = Select::with_theme(&theme)
            .with_prompt("Commit action")
            .items(options)
            .default(0)
            .interact_on_opt(&term)
            .map_err(|err| format!("failed to read commit action: {err}"))?;

        match selection {
            Some(0) => return Ok(Some(message)),
            Some(1) => {
                let edited = Editor::new()
                    .extension(".gitcommit")
                    .edit(&message)
                    .map_err(|err| format!("failed to open commit editor: {err}"))?;

                let Some(next_message) = edited else {
                    continue;
                };

                let trimmed = next_message.trim();
                if trimmed.is_empty() {
                    println!("commit message cannot be empty");
                    continue;
                }

                message = trimmed.to_string();
            }
            Some(2) | None => return Ok(None),
            Some(_) => return Err("invalid commit action selection".to_string()),
        }
    }
}

fn prompt_for_commit_message_basic(initial_message: &str) -> Result<Option<String>, String> {
    let mut message = initial_message.trim().to_string();

    loop {
        print_commit_preview(&message, false);
        println!("Commit action:");
        println!("  [a] Approve commit");
        println!("  [e] Edit message");
        println!("  [c] Cancel");
        print!("Select action [a/e/c]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;

        let choice = read_line_trimmed()?;
        match choice.as_str() {
            "a" | "A" => return Ok(Some(message)),
            "e" | "E" => {
                let edited = Editor::new()
                    .extension(".gitcommit")
                    .edit(&message)
                    .map_err(|err| format!("failed to open commit editor: {err}"))?;

                let Some(next_message) = edited else {
                    continue;
                };

                let trimmed = next_message.trim();
                if trimmed.is_empty() {
                    println!("commit message cannot be empty");
                    continue;
                }
                message = trimmed.to_string();
            }
            "c" | "C" => return Ok(None),
            _ => println!("invalid choice, enter a, e, or c"),
        }
    }
}

fn prompt_should_push(default: bool, rich: bool) -> Result<bool, String> {
    if rich {
        return Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Push commit to remote now?")
            .default(default)
            .interact_on(&Term::stderr())
            .map_err(|err| format!("failed to read push confirmation: {err}"));
    }

    let default_hint = if default { "Y/n" } else { "y/N" };
    loop {
        print!("Push commit to remote now? [{default_hint}]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(default);
        }

        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn push_with_feedback(interactive: bool, repo: &git::Repo) -> Result<Option<String>, CoreError> {
    if !interactive {
        return match repo.push() {
            Ok(()) => Ok(None),
            Err(err) if looks_like_unimplemented_push_error(&err) => {
                Ok(Some(format!("push skipped: {err}")))
            }
            Err(err) => Err(err),
        };
    }

    let progress = ProgressBar::new_spinner();
    progress.set_style(spinner_style());
    progress.set_message("Pushing commit".to_string());
    progress.enable_steady_tick(std::time::Duration::from_millis(80));

    match repo.push() {
        Ok(()) => {
            progress.finish_with_message("[ok] Pushing commit");
            Ok(None)
        }
        Err(err) if looks_like_unimplemented_push_error(&err) => {
            progress.finish_with_message("[skip] Pushing commit (not available in this build)");
            Ok(Some(format!("push skipped: {err}")))
        }
        Err(err) => {
            progress.abandon_with_message(format!("[error] Pushing commit: {err}"));
            Err(err)
        }
    }
}

fn looks_like_unimplemented_push_error(err: &CoreError) -> bool {
    err.to_string()
        .to_ascii_lowercase()
        .contains("not implemented")
}

fn print_commit_preview(message: &str, rich: bool) {
    if !rich {
        println!("\nProposed commit message:\n\n{message}\n");
        return;
    }

    let mut lines = message.lines();
    let subject = lines.next().unwrap_or_default();

    println!();
    println!(
        "{}",
        style("Proposed commit message").bold().underlined().cyan()
    );
    println!(
        "{}",
        style("----------------------------------------").dim()
    );
    if !subject.is_empty() {
        println!("{}", style(subject).bold().green());
    }

    for line in lines {
        if line.is_empty() {
            println!();
            continue;
        }

        if line.starts_with("### ") || line.ends_with(':') {
            println!("{}", style(line).bold().yellow());
            continue;
        }

        if line.starts_with("- [") {
            println!("{}", style(line).cyan());
            continue;
        }

        if line.starts_with("- ") {
            println!("{}", style(line).blue());
            continue;
        }

        println!("{}", style(line).white());
    }
    println!(
        "{}",
        style("----------------------------------------").dim()
    );
    println!();
}

fn read_line_trimmed() -> Result<String, String> {
    let mut buffer = String::new();
    let read = std::io::stdin()
        .read_line(&mut buffer)
        .map_err(|err| format!("failed to read prompt input: {err}"))?;
    if read == 0 {
        return Err("interactive input was closed".to_string());
    }
    Ok(buffer.trim().to_string())
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

fn prepare_diff(repo: &git::Repo, staged_only: bool, dry_run: bool) -> Result<String, CoreError> {
    if staged_only {
        let staged = repo.diff_cached()?;
        if staged.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no staged changes to commit".to_string(),
            ));
        }
        return Ok(staged);
    }

    if dry_run {
        let staged = repo.diff_cached()?;
        let unstaged = repo.diff_worktree()?;
        let combined = concat_diffs(&staged, &unstaged);
        if combined.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no changes in working tree".to_string(),
            ));
        }
        return Ok(combined);
    }

    let staged = repo.diff_cached()?;
    let unstaged = repo.diff_worktree()?;
    let combined = concat_diffs(&staged, &unstaged);
    if combined.trim().is_empty() {
        return Err(CoreError::InvalidDiff(
            "no changes in working tree".to_string(),
        ));
    }

    Ok(combined)
}

fn concat_diffs(staged: &str, unstaged: &str) -> String {
    let mut combined = String::new();
    if !staged.trim().is_empty() {
        combined.push_str(staged);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    if !unstaged.trim().is_empty() {
        combined.push_str(unstaged);
    }
    combined
}

fn commit_with_message(
    repo: &git::Repo,
    message: &str,
    staged_only: bool,
    no_verify: bool,
    policy: &CommitPolicy,
) -> Result<(), CoreError> {
    let mut lines = message.lines();
    let subject = lines.next().unwrap_or_default().trim();
    if subject.is_empty() {
        return Err(CoreError::InvalidDiff(
            "generated commit subject is empty".to_string(),
        ));
    }

    let body = lines.collect::<Vec<_>>().join("\n");
    let body = body.trim();

    let full_message = if body.is_empty() {
        subject.to_string()
    } else {
        format!("{subject}\n\n{body}")
    };

    repo.commit(&full_message, staged_only, no_verify, policy.sign_commits)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct GpgInstallHint {
    manager: &'static str,
    command: &'static str,
    args: &'static [&'static str],
}

fn ensure_signing_tool_ready(
    repo: &git::Repo,
    policy: &mut CommitPolicy,
    interactive: bool,
    rich_interactive: bool,
    assume_yes: bool,
) -> Result<(), CoreError> {
    if !policy.sign_commits {
        return Ok(());
    }

    if !command_exists("gpg") {
        let hint = detect_gpg_install_hint();
        if interactive && !assume_yes {
            prompt_install_gpg(hint, rich_interactive)?;
        } else {
            return Err(CoreError::Io(missing_gpg_message(hint)));
        }
    }

    let keys = list_gpg_secret_keys()?;
    if keys.is_empty() {
        if interactive && !assume_yes {
            return prompt_resolve_missing_secret_key(repo, policy, rich_interactive);
        }
        return Err(CoreError::Io(
            "signed commit policy requires a usable GPG secret key, but none was found. Run `gpg --full-generate-key` (or import an existing key), then configure `git config --global user.signingkey <KEYID>`."
                .to_string(),
        ));
    }

    if let Some(configured) = repo.signing_key()? {
        if keys
            .iter()
            .any(|key| signing_key_matches(key, configured.as_str()))
        {
            return Ok(());
        }
        if interactive && !assume_yes {
            return prompt_resolve_invalid_signing_key(
                repo,
                policy,
                rich_interactive,
                &keys,
                &configured,
            );
        }
        return Err(CoreError::Io(format!(
            "configured signing key `{configured}` was not found in local GPG secret keys. Set a valid key with `git config --global user.signingkey <KEYID>`."
        )));
    }

    if interactive && !assume_yes {
        maybe_prompt_configure_signing_key(repo, rich_interactive, &keys)?;
    }

    Ok(())
}

fn prompt_install_gpg(
    hint: Option<GpgInstallHint>,
    rich_interactive: bool,
) -> Result<(), CoreError> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();
    let prompt = if let Some(hint) = hint.as_ref() {
        format!(
            "Signed commits require `gpg`, but it was not found. Install now using {}?",
            hint.manager
        )
    } else {
        "Signed commits require `gpg`, but it was not found. Install now?".to_string()
    };

    let install = if rich_interactive {
        Confirm::with_theme(&theme)
            .with_prompt(prompt)
            .default(true)
            .interact_on(&term)
            .map_err(|err| {
                CoreError::Io(format!("failed to read gpg install confirmation: {err}"))
            })?
    } else {
        prompt_yes_no_basic_local(&format!("{prompt} [Y/n]: "), true)?
    };

    if !install {
        return Err(CoreError::Io(missing_gpg_message(hint)));
    }

    let Some(hint) = hint else {
        return Err(CoreError::Io(
            "`gpg` installer was not auto-detected; install it with your OS package manager and retry".to_string(),
        ));
    };

    let mut command = Command::new(hint.command);
    command.args(hint.args);
    if hint.command == "brew" {
        command
            .env("HOMEBREW_NO_INSTALL_CLEANUP", "1")
            .env("HOMEBREW_NO_ENV_HINTS", "1");
    }
    let status = command.status().map_err(|err| {
        CoreError::Io(format!(
            "failed to run `{}` for gpg installation: {err}",
            hint.render()
        ))
    })?;
    if status.success() {
        if command_exists("gpg") {
            return Ok(());
        }
        return Err(CoreError::Io(
            "`gpg` install command completed but `gpg` is still not on PATH".to_string(),
        ));
    }

    Err(CoreError::Io(format!(
        "gpg installation command failed (exit: {}): {}",
        status
            .code()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "terminated by signal".to_string()),
        hint.render()
    )))
}

fn missing_gpg_message(hint: Option<GpgInstallHint>) -> String {
    match hint {
        Some(hint) => format!(
            "signed commit policy requires `gpg`, but it was not found. Install it with `{}` or disable signing via `autocommit-cli commit --configure-commit-policy`",
            hint.render()
        ),
        None => "signed commit policy requires `gpg`, but it was not found. Install it with your OS package manager or disable signing via `autocommit-cli commit --configure-commit-policy`".to_string(),
    }
}

#[derive(Clone, Debug)]
struct GpgSecretKey {
    key_id: String,
    fingerprint: Option<String>,
    user_ids: Vec<String>,
}

impl GpgSecretKey {
    fn config_value(&self) -> &str {
        self.fingerprint.as_deref().unwrap_or(self.key_id.as_str())
    }

    fn label(&self) -> String {
        let identity = self
            .user_ids
            .first()
            .map(|value| value.as_str())
            .unwrap_or("<no uid>");
        format!("{identity} ({})", self.config_value())
    }
}

fn list_gpg_secret_keys() -> Result<Vec<GpgSecretKey>, CoreError> {
    let output = Command::new("gpg")
        .args(["--list-secret-keys", "--with-colons", "--fingerprint"])
        .output()
        .map_err(|err| CoreError::Io(format!("failed to run gpg key listing: {err}")))?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut keys: Vec<GpgSecretKey> = Vec::new();
    let mut current: Option<GpgSecretKey> = None;

    for line in stdout.lines() {
        let fields = line.split(':').collect::<Vec<_>>();
        let Some(record) = fields.first().copied() else {
            continue;
        };
        match record {
            "sec" | "sec#" => {
                if let Some(prev) = current.take() {
                    keys.push(prev);
                }
                let key_id = fields
                    .get(4)
                    .copied()
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                current = Some(GpgSecretKey {
                    key_id,
                    fingerprint: None,
                    user_ids: Vec::new(),
                });
            }
            "fpr" => {
                if let Some(key) = current.as_mut() {
                    if key.fingerprint.is_none() {
                        let value = fields.get(9).copied().unwrap_or_default().trim();
                        if !value.is_empty() {
                            key.fingerprint = Some(value.to_string());
                        }
                    }
                }
            }
            "uid" => {
                if let Some(key) = current.as_mut() {
                    let value = fields.get(9).copied().unwrap_or_default().trim();
                    if !value.is_empty() {
                        key.user_ids.push(value.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(last) = current.take() {
        keys.push(last);
    }

    keys.retain(|key| !key.key_id.trim().is_empty() || key.fingerprint.is_some());
    Ok(keys)
}

fn normalize_key_ref(value: &str) -> String {
    value
        .trim()
        .trim_start_matches("0x")
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn signing_key_matches(key: &GpgSecretKey, configured: &str) -> bool {
    let target = normalize_key_ref(configured);
    if target.is_empty() {
        return false;
    }
    let key_id = normalize_key_ref(&key.key_id);
    if !key_id.is_empty()
        && (key_id == target || key_id.ends_with(&target) || target.ends_with(&key_id))
    {
        return true;
    }

    key.fingerprint
        .as_deref()
        .map(normalize_key_ref)
        .map(|fpr| fpr == target || fpr.ends_with(&target) || target.ends_with(&fpr))
        .unwrap_or(false)
}

fn prompt_resolve_missing_secret_key(
    repo: &git::Repo,
    policy: &mut CommitPolicy,
    rich_interactive: bool,
) -> Result<(), CoreError> {
    let prompt = "Signed commits require a GPG secret key, but none was found.";
    if rich_interactive {
        println!("\n{}", style(prompt).yellow().bold());
        let options = [
            "Generate a new key now (`gpg --full-generate-key`) (recommended)",
            "Commit unsigned this time",
            "Abort",
        ];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Signing key action")
            .items(options)
            .default(0)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| CoreError::Io(format!("failed to read signing key action: {err}")))?;
        match selection {
            Some(0) => {
                run_gpg_generate_key()?;
                let keys = list_gpg_secret_keys()?;
                if keys.is_empty() {
                    return Err(CoreError::Io(
                        "no usable GPG secret key found after key generation".to_string(),
                    ));
                }
                prompt_configure_signing_key(repo, rich_interactive, &keys)?;
                Ok(())
            }
            Some(1) => {
                policy.sign_commits = false;
                Ok(())
            }
            _ => Err(CoreError::Io(
                "commit canceled: signing key setup required for signed commits".to_string(),
            )),
        }
    } else {
        println!("{prompt}");
        let generate = prompt_yes_no_basic_local(
            "Generate one now using `gpg --full-generate-key`? [Y/n]: ",
            true,
        )?;
        if generate {
            run_gpg_generate_key()?;
            let keys = list_gpg_secret_keys()?;
            if keys.is_empty() {
                return Err(CoreError::Io(
                    "no usable GPG secret key found after key generation".to_string(),
                ));
            }
            prompt_configure_signing_key(repo, rich_interactive, &keys)?;
            return Ok(());
        }
        let commit_unsigned =
            prompt_yes_no_basic_local("Commit unsigned this time instead? [y/N]: ", false)?;
        if commit_unsigned {
            policy.sign_commits = false;
            return Ok(());
        }
        Err(CoreError::Io(
            "commit canceled: signing key setup required for signed commits".to_string(),
        ))
    }
}

fn prompt_resolve_invalid_signing_key(
    repo: &git::Repo,
    policy: &mut CommitPolicy,
    rich_interactive: bool,
    keys: &[GpgSecretKey],
    configured: &str,
) -> Result<(), CoreError> {
    let message = format!("Configured signing key `{configured}` is not available locally.");
    if rich_interactive {
        println!("\n{}", style(message).yellow().bold());
        let options = [
            "Select available key and update `user.signingkey` (recommended)",
            "Commit unsigned this time",
            "Abort",
        ];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Signing key action")
            .items(options)
            .default(0)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| CoreError::Io(format!("failed to read signing key action: {err}")))?;
        match selection {
            Some(0) => prompt_configure_signing_key(repo, rich_interactive, keys),
            Some(1) => {
                policy.sign_commits = false;
                Ok(())
            }
            _ => Err(CoreError::Io(
                "commit canceled: signing key setup required for signed commits".to_string(),
            )),
        }
    } else {
        println!("{message}");
        let fix = prompt_yes_no_basic_local(
            "Select a valid local key and update `user.signingkey` now? [Y/n]: ",
            true,
        )?;
        if fix {
            return prompt_configure_signing_key(repo, rich_interactive, keys);
        }
        let commit_unsigned =
            prompt_yes_no_basic_local("Commit unsigned this time instead? [y/N]: ", false)?;
        if commit_unsigned {
            policy.sign_commits = false;
            return Ok(());
        }
        Err(CoreError::Io(
            "commit canceled: signing key setup required for signed commits".to_string(),
        ))
    }
}

fn maybe_prompt_configure_signing_key(
    repo: &git::Repo,
    rich_interactive: bool,
    keys: &[GpgSecretKey],
) -> Result<(), CoreError> {
    let configure = if rich_interactive {
        Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("No `user.signingkey` configured. Configure one now?")
            .default(true)
            .interact_on(&Term::stderr())
            .map_err(|err| {
                CoreError::Io(format!(
                    "failed to read signing key configuration choice: {err}"
                ))
            })?
    } else {
        prompt_yes_no_basic_local(
            "No `user.signingkey` configured. Configure one now? [Y/n]: ",
            true,
        )?
    };
    if !configure {
        return Ok(());
    }

    prompt_configure_signing_key(repo, rich_interactive, keys)
}

fn prompt_configure_signing_key(
    repo: &git::Repo,
    rich_interactive: bool,
    keys: &[GpgSecretKey],
) -> Result<(), CoreError> {
    if keys.is_empty() {
        return Err(CoreError::Io(
            "no local GPG secret keys available to configure".to_string(),
        ));
    }

    let key_index = select_signing_key_index(keys, rich_interactive)?;
    let selected = &keys[key_index];
    let global = select_signing_key_scope(rich_interactive)?;
    repo.set_signing_key(selected.config_value(), global)?;

    let scope = if global { "global" } else { "repository-local" };
    println!(
        "{}",
        style(format!(
            "configured {scope} `user.signingkey` = {}",
            selected.config_value()
        ))
        .green()
    );
    Ok(())
}

fn select_signing_key_index(
    keys: &[GpgSecretKey],
    rich_interactive: bool,
) -> Result<usize, CoreError> {
    if keys.len() == 1 {
        let label = keys[0].label();
        let accept = if rich_interactive {
            Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt(format!("Use signing key: {label}?"))
                .default(true)
                .interact_on(&Term::stderr())
                .map_err(|err| {
                    CoreError::Io(format!("failed to read signing key selection: {err}"))
                })?
        } else {
            prompt_yes_no_basic_local(&format!("Use signing key: {label}? [Y/n]: "), true)?
        };
        if accept {
            return Ok(0);
        }
        return Err(CoreError::Io(
            "commit canceled: signing key selection required for signed commits".to_string(),
        ));
    }

    let labels = keys.iter().map(GpgSecretKey::label).collect::<Vec<_>>();
    if rich_interactive {
        return Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select signing key")
            .items(&labels)
            .default(0)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| CoreError::Io(format!("failed to read signing key selection: {err}")))?
            .ok_or_else(|| {
                CoreError::Io(
                    "commit canceled: signing key selection required for signed commits"
                        .to_string(),
                )
            });
    }

    println!("Available signing keys:");
    for (idx, label) in labels.iter().enumerate() {
        println!("{}. {}", idx + 1, label);
    }
    loop {
        print!("Select signing key [1-{}]: ", labels.len());
        std::io::stdout()
            .flush()
            .map_err(|err| CoreError::Io(format!("failed to flush prompt output: {err}")))?;
        let value = read_line_trimmed().map_err(CoreError::Io)?;
        let parsed = value
            .parse::<usize>()
            .ok()
            .filter(|idx| *idx >= 1 && *idx <= labels.len());
        if let Some(index) = parsed {
            return Ok(index - 1);
        }
        println!(
            "invalid choice, enter a number between 1 and {}",
            labels.len()
        );
    }
}

fn select_signing_key_scope(rich_interactive: bool) -> Result<bool, CoreError> {
    if rich_interactive {
        let options = [
            "Global (`git config --global`) (recommended)",
            "Repository-local (`git config --local`)",
        ];
        let selected = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Where should `user.signingkey` be written?")
            .items(options)
            .default(0)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| CoreError::Io(format!("failed to read signing key scope: {err}")))?;
        return Ok(!matches!(selected, Some(1)));
    }

    prompt_yes_no_basic_local(
        "Write `user.signingkey` to global git config? [Y/n]: ",
        true,
    )
}

fn run_gpg_generate_key() -> Result<(), CoreError> {
    let status = Command::new("gpg")
        .arg("--full-generate-key")
        .status()
        .map_err(|err| CoreError::Io(format!("failed to launch gpg key generation: {err}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(CoreError::Io(format!(
            "gpg key generation failed (exit: {})",
            status
                .code()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string())
        )))
    }
}

fn detect_gpg_install_hint() -> Option<GpgInstallHint> {
    let candidates: &[GpgInstallHint] = if cfg!(target_os = "macos") {
        &[
            GpgInstallHint {
                manager: "Homebrew",
                command: "brew",
                args: &["install", "gnupg"],
            },
            GpgInstallHint {
                manager: "MacPorts",
                command: "sudo",
                args: &["port", "install", "gnupg"],
            },
        ]
    } else if cfg!(target_os = "windows") {
        &[
            GpgInstallHint {
                manager: "Winget",
                command: "winget",
                args: &["install", "--id", "GnuPG.GnuPG", "-e"],
            },
            GpgInstallHint {
                manager: "Chocolatey",
                command: "choco",
                args: &["install", "gnupg", "-y"],
            },
        ]
    } else {
        &[
            GpgInstallHint {
                manager: "apt",
                command: "sudo",
                args: &["apt-get", "install", "-y", "gnupg"],
            },
            GpgInstallHint {
                manager: "dnf",
                command: "sudo",
                args: &["dnf", "install", "-y", "gnupg2"],
            },
            GpgInstallHint {
                manager: "yum",
                command: "sudo",
                args: &["yum", "install", "-y", "gnupg2"],
            },
            GpgInstallHint {
                manager: "pacman",
                command: "sudo",
                args: &["pacman", "-S", "--noconfirm", "gnupg"],
            },
            GpgInstallHint {
                manager: "zypper",
                command: "sudo",
                args: &["zypper", "--non-interactive", "install", "gpg2"],
            },
            GpgInstallHint {
                manager: "apk",
                command: "sudo",
                args: &["apk", "add", "gnupg"],
            },
        ]
    };

    candidates
        .iter()
        .find(|hint| command_exists(hint.command))
        .cloned()
}

fn command_exists(command: &str) -> bool {
    Command::new(command).arg("--version").output().is_ok()
}

impl GpgInstallHint {
    fn render(&self) -> String {
        format!("{} {}", self.command, self.args.join(" "))
    }
}

fn prompt_yes_no_basic_local(prompt: &str, default: bool) -> Result<bool, CoreError> {
    loop {
        print!("{prompt}");
        std::io::stdout()
            .flush()
            .map_err(|err| CoreError::Io(format!("failed to flush prompt output: {err}")))?;
        let value = read_line_trimmed().map_err(CoreError::Io)?;
        if value.is_empty() {
            return Ok(default);
        }
        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn prepare_message_for_policy(
    repo: &git::Repo,
    message: &str,
    policy: &CommitPolicy,
) -> Result<String, CoreError> {
    let mut out = message.trim().to_string();
    if policy.add_signoff {
        let identity = repo.signoff_identity()?;
        out = commit_policy::append_signoff_trailer(&out, &identity);
    }
    commit_policy::validate_commit_message(&out, policy).map_err(CoreError::InvalidDiff)?;
    Ok(out)
}

fn apply_approved_version_bumps(
    repo: &git::Repo,
    staged_only: bool,
    recommendations: &[version_bump::VersionRecommendation],
) -> Result<(), String> {
    if recommendations.is_empty() {
        return Ok(());
    }
    if staged_only {
        return Err(
            "approved version bumps cannot be auto-applied with --staged; rerun without --staged"
                .to_string(),
        );
    }

    version_bump::apply(repo, recommendations)?;
    Ok(())
}

fn infer_embedding_bump_level(
    engine: &dyn LlmEngine,
    _diff_text: &str,
    report: &AnalysisReport,
) -> Option<version_bump::BumpLevel> {
    let signal = build_bump_embedding_signal(report);
    let signal_embedding = engine.embed(&signal).ok()?;
    if signal_embedding.is_empty() {
        return None;
    }

    let anchors = [
        (
            version_bump::BumpLevel::Patch,
            "Patch release: backward-compatible bug fixes, documentation updates, refactors that keep public behavior stable.",
        ),
        (
            version_bump::BumpLevel::Minor,
            "Minor release: backward-compatible new functionality, new commands, new APIs, new features without breaking existing usage.",
        ),
        (
            version_bump::BumpLevel::Major,
            "Major release: breaking changes, incompatible API changes, removed behavior, required migration steps for users.",
        ),
    ];

    let mut best: Option<(version_bump::BumpLevel, f32)> = None;
    let mut runner_up: Option<f32> = None;
    for (level, text) in anchors {
        let anchor_embedding = engine.embed(text).ok()?;
        let similarity = cosine_similarity(&signal_embedding, &anchor_embedding)?;
        match best {
            Some((_, current)) if current >= similarity => {
                if runner_up.map_or(true, |value| similarity > value) {
                    runner_up = Some(similarity);
                }
            }
            Some((_, current)) => {
                runner_up = Some(current);
                best = Some((level, similarity));
            }
            None => {
                best = Some((level, similarity));
            }
        }
    }
    let (level, score) = best?;
    let runner = runner_up.unwrap_or(0.0);
    let margin = score - runner;
    // Require a modest confidence gap before overriding heuristic levels.
    if score < 0.55 || margin < 0.05 {
        return None;
    }

    Some(level)
}

fn build_bump_embedding_signal(report: &AnalysisReport) -> String {
    let subject = report.commit_message.trim();
    let summary = report.summary.trim();
    let risk = clamp_text(&report.risk.notes.join(" "), 160);
    let item_text = report
        .items
        .iter()
        .take(8)
        .map(|item| format!("{} {}", item.title.trim(), item.intent.trim()))
        .collect::<Vec<_>>()
        .join(" | ");
    let item_text = clamp_text(&item_text, 420);

    let signal =
        format!("subject: {subject}\nsummary: {summary}\nrisk: {risk}\nitems: {item_text}");
    clamp_text(&signal, 760)
}

fn clamp_text(value: &str, max_chars: usize) -> String {
    value
        .chars()
        .take(max_chars)
        .collect::<String>()
        .replace('\n', " ")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let len = a.len().min(b.len());
    if len == 0 {
        return None;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for idx in 0..len {
        let av = a[idx];
        let bv = b[idx];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return None;
    }

    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

fn compose_commit_message(
    report: &AnalysisReport,
    recommendations: &[version_bump::VersionRecommendation],
) -> String {
    let subject = report.commit_message.trim();
    if subject.is_empty() {
        return String::new();
    }

    let body = compose_commit_body(report, subject, recommendations);
    if body.is_empty() {
        subject.to_string()
    } else {
        format!("{subject}\n\n{body}")
    }
}

fn resolve_version_bump_recommendations(
    recommendations: &[version_bump::VersionRecommendation],
    interactive: bool,
    rich_interactive: bool,
    assume_yes: bool,
) -> Result<Vec<version_bump::VersionRecommendation>, String> {
    if recommendations.is_empty() {
        return Ok(Vec::new());
    }
    if assume_yes {
        return Ok(recommendations.to_vec());
    }
    if !interactive {
        return Ok(Vec::new());
    }

    if rich_interactive {
        return prompt_version_bump_recommendations_rich(recommendations);
    }

    prompt_version_bump_recommendations_basic(recommendations)
}

fn prompt_version_bump_recommendations_rich(
    recommendations: &[version_bump::VersionRecommendation],
) -> Result<Vec<version_bump::VersionRecommendation>, String> {
    let theme = ColorfulTheme::default();
    let term = Term::stderr();
    let options = [
        "Approve, apply, and include in commit message",
        "Skip for this commit",
    ];

    println!();
    println!(
        "{}",
        style("Version bump recommendations")
            .bold()
            .underlined()
            .cyan()
    );
    for rec in recommendations {
        println!(
            "{}",
            style(format!("- {}", format_version_recommendation(rec))).cyan()
        );
    }
    println!();

    let selection = Select::with_theme(&theme)
        .with_prompt("Version bump action")
        .items(options)
        .default(0)
        .interact_on_opt(&term)
        .map_err(|err| format!("failed to read version bump action: {err}"))?;

    match selection {
        Some(0) => Ok(recommendations.to_vec()),
        Some(1) | None => Ok(Vec::new()),
        Some(_) => Err("invalid version bump action selection".to_string()),
    }
}

fn prompt_version_bump_recommendations_basic(
    recommendations: &[version_bump::VersionRecommendation],
) -> Result<Vec<version_bump::VersionRecommendation>, String> {
    println!();
    println!("Version bump recommendations:");
    for rec in recommendations {
        println!("- {}", format_version_recommendation(rec));
    }

    loop {
        print!("Apply these version bumps and include them in commit message? [Y/n]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(recommendations.to_vec());
        }
        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(recommendations.to_vec()),
            "n" | "no" => return Ok(Vec::new()),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn compose_commit_body(
    report: &AnalysisReport,
    subject: &str,
    recommendations: &[version_bump::VersionRecommendation],
) -> String {
    let mut sections = Vec::new();

    let summary = report.summary.trim();
    if should_include_summary(summary, subject) {
        sections.push(summary.to_string());
    }

    let changes = compose_changes_section(report);
    if !changes.is_empty() {
        sections.push(changes);
    }

    let versions = compose_version_recommendations_section(recommendations);
    if !versions.is_empty() {
        sections.push(versions);
    }

    let risk = compose_risk_section(report);
    if !risk.is_empty() {
        sections.push(risk);
    }

    sections.join("\n\n")
}

fn should_include_summary(summary: &str, subject: &str) -> bool {
    if summary.is_empty() {
        return false;
    }

    let summary_norm = normalize_for_compare(summary_subject(summary));
    let subject_norm = normalize_for_compare(subject_description(subject));
    if summary_norm.is_empty() || subject_norm.is_empty() {
        return true;
    }

    if summary_norm == subject_norm {
        return false;
    }

    !(summary_norm.starts_with(&subject_norm) || subject_norm.starts_with(&summary_norm))
}

fn subject_description(subject: &str) -> &str {
    if let Some((_, desc)) = subject.split_once(':') {
        return desc.trim();
    }
    subject.trim()
}

fn summary_subject(summary: &str) -> &str {
    let summary = summary.trim();
    let Some((label, rest)) = summary.split_once(':') else {
        return summary;
    };

    let label = label.trim().to_ascii_lowercase();
    if matches!(
        label.as_str(),
        "feat"
            | "feature"
            | "fix"
            | "refactor"
            | "docs"
            | "doc"
            | "test"
            | "chore"
            | "perf"
            | "style"
    ) {
        rest.trim()
    } else {
        summary
    }
}

fn normalize_for_compare(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn compose_changes_section(report: &AnalysisReport) -> String {
    if report.items.is_empty() {
        return String::new();
    }

    let mut out = String::from("### Changes\n");
    for item in &report.items {
        out.push_str("- ");
        out.push_str(&format_change_item(item));
        out.push('\n');
    }
    out.trim_end().to_string()
}

fn compose_version_recommendations_section(
    recommendations: &[version_bump::VersionRecommendation],
) -> String {
    if recommendations.is_empty() {
        return String::new();
    }

    let mut out = String::from("### Version Bumps\n");
    for rec in recommendations {
        out.push_str("- ");
        out.push_str(&format_version_recommendation(rec));
        out.push('\n');
    }
    out.trim_end().to_string()
}

fn format_version_recommendation(rec: &version_bump::VersionRecommendation) -> String {
    let mut out = format!("[`{}`] {} / {}", rec.manifest_path, rec.ecosystem, rec.tool);
    if let (Some(current), Some(suggested)) = (
        rec.current_version.as_deref(),
        rec.suggested_version.as_deref(),
    ) {
        out.push_str(&format!(
            ": {} -> {} ({})",
            current,
            suggested,
            rec.level.as_str()
        ));
        out.push_str(&format!("; {}", rec.reason));
    } else {
        out.push_str(&format!(": {} ({})", rec.reason, rec.level.as_str()));
    }
    out
}

fn compose_risk_section(report: &AnalysisReport) -> String {
    let level = report.risk.level.trim();
    let notes = report
        .risk
        .notes
        .iter()
        .map(|note| note.trim())
        .filter(|note| !note.is_empty() && !looks_like_internal_risk_tag(note))
        .collect::<Vec<_>>();

    if level.is_empty() && notes.is_empty() {
        return String::new();
    }

    let mut out = String::from("### Risk\n");
    if !level.is_empty() {
        out.push_str(&format!("- Level: {level}\n"));
    }

    if !notes.is_empty() {
        for note in notes {
            out.push_str("- ");
            out.push_str(note);
            out.push('\n');
        }
    }

    out.trim_end().to_string()
}

fn format_change_item(item: &autocommit_core::types::ChangeItem) -> String {
    let path = item
        .files
        .first()
        .map(|file| file.path.as_str())
        .unwrap_or("<unknown>");
    let file_suffix = if item.files.len() > 1 {
        format!(" (+{} more)", item.files.len().saturating_sub(1))
    } else {
        String::new()
    };

    let title = normalize_change_fragment(item.title.trim());
    let title = rewrite_low_signal_suffix(&title);
    let title = wrap_code_like_tokens(&title);
    let title = capitalize_first_alpha(&title);
    let title = clamp_words(&title, 88);
    let title = if title.is_empty() {
        "update file".to_string()
    } else {
        title
    };

    let intent = normalize_change_fragment(item.intent.trim());
    let intent = rewrite_low_signal_suffix(&intent);
    let intent = wrap_code_like_tokens(&intent);
    let intent = clamp_words(&intent, 132);

    let suffix = if should_include_intent_detail(&title, &intent) {
        format!(": {intent}")
    } else {
        String::new()
    };

    format!("[`{path}`{file_suffix}] {title}{suffix}")
}

fn normalize_change_fragment(raw: &str) -> String {
    let mut out = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    out = out
        .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .trim()
        .trim_end_matches(['.', ';', ','])
        .to_string();
    if out.is_empty() {
        return out;
    }

    if let Some((prefix, rest)) = out.split_once(':') {
        let lower = prefix.trim().to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "feat"
                | "feature"
                | "fix"
                | "refactor"
                | "docs"
                | "doc"
                | "test"
                | "chore"
                | "perf"
                | "style"
        ) {
            out = rest.trim().to_string();
        }
    }

    for prefix in [
        "implement a feature that ",
        "implement feature that ",
        "add a feature that ",
        "add feature that ",
        "create a feature that ",
        "create feature that ",
    ] {
        if out.to_ascii_lowercase().starts_with(prefix) {
            out = out[prefix.len()..].trim_start().to_string();
            break;
        }
    }

    for (from, to) in [
        ("composes ", "compose "),
        ("creates ", "create "),
        ("adds ", "add "),
        ("builds ", "build "),
        ("updates ", "update "),
        ("fixes ", "fix "),
        ("refactors ", "refactor "),
        ("improves ", "improve "),
        ("supports ", "support "),
        ("prevents ", "prevent "),
        ("guards ", "guard "),
        ("implements ", "implement "),
    ] {
        if out.to_ascii_lowercase().starts_with(from) {
            out = format!("{to}{}", out[from.len()..].trim_start());
            break;
        }
    }

    for (from, to) in [
        (" and creates ", " and create "),
        (" and adds ", " and add "),
        (" and builds ", " and build "),
        (" and updates ", " and update "),
        (" and fixes ", " and fix "),
        (" and refactors ", " and refactor "),
        (" and improves ", " and improve "),
        (" and supports ", " and support "),
        (" and prevents ", " and prevent "),
        (" and guards ", " and guard "),
    ] {
        out = out.replace(from, to);
    }

    out
}

fn rewrite_low_signal_suffix(raw: &str) -> String {
    let value = raw.trim();
    if value.is_empty() {
        return String::new();
    }

    let Some((head, tail)) = value.rsplit_once(':') else {
        return value.to_string();
    };
    let head = head.trim();
    let tail = normalize_for_compare(tail);

    if head.is_empty() {
        return value.to_string();
    }

    if matches!(tail.as_str(), "diagnostic" | "diagnostics") {
        return format!("add diagnostics for {head}");
    }
    if is_generic_single_word(&tail) {
        return head.to_string();
    }

    value.to_string()
}

fn is_generic_single_word(value: &str) -> bool {
    matches!(
        value,
        "add"
            | "added"
            | "addition"
            | "update"
            | "updated"
            | "change"
            | "changes"
            | "diagnostic"
            | "diagnostics"
            | "implementation"
            | "implement"
            | "refactor"
            | "cleanup"
            | "misc"
            | "other"
    )
}

fn wrap_code_like_tokens(raw: &str) -> String {
    raw.split_whitespace()
        .map(wrap_code_like_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn wrap_code_like_token(token: &str) -> String {
    if token.contains('`') || token.is_empty() {
        return token.to_string();
    }

    let bytes = token.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let ch = bytes[start] as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':') {
            break;
        }
        start += 1;
    }
    let mut end = bytes.len();
    while end > start {
        let ch = bytes[end - 1] as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':') {
            break;
        }
        end -= 1;
    }
    if start >= end {
        return token.to_string();
    }

    let prefix = &token[..start];
    let core = &token[start..end];
    let suffix = &token[end..];
    if !looks_code_like(core) {
        return token.to_string();
    }

    format!("{prefix}`{core}`{suffix}")
}

fn looks_code_like(token: &str) -> bool {
    if token.starts_with("--") && token.len() > 2 {
        return true;
    }
    if token.contains('_') && token.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return true;
    }
    false
}

fn capitalize_first_alpha(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut done = false;
    for ch in raw.chars() {
        if !done && ch.is_ascii_alphabetic() {
            out.push(ch.to_ascii_uppercase());
            done = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn clamp_words(value: &str, max_chars: usize) -> String {
    let value = value.trim();
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut out = String::new();
    for token in value.split_whitespace() {
        let next = if out.is_empty() {
            token.to_string()
        } else {
            format!("{out} {token}")
        };
        if next.chars().count() > max_chars {
            break;
        }
        out = next;
    }

    if out.is_empty() {
        value
            .chars()
            .take(max_chars)
            .collect::<String>()
            .trim()
            .to_string()
    } else {
        out
    }
}

fn should_include_intent_detail(title: &str, intent: &str) -> bool {
    let title = title.trim();
    let intent = intent.trim();
    if title.is_empty() || intent.is_empty() {
        return false;
    }
    if title.eq_ignore_ascii_case(intent) {
        return false;
    }
    if is_low_signal_intent(intent) {
        return false;
    }

    let title_lower = title.to_ascii_lowercase();
    let intent_lower = intent.to_ascii_lowercase();
    if intent_lower.contains(&title_lower) || title_lower.contains(&intent_lower) {
        return false;
    }

    let title_words = title_lower.split_whitespace().collect::<Vec<_>>();
    let intent_words = intent_lower.split_whitespace().collect::<Vec<_>>();
    let shared_prefix = title_words
        .iter()
        .zip(intent_words.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if shared_prefix >= 3 {
        return false;
    }

    true
}

fn is_low_signal_intent(intent: &str) -> bool {
    let normalized = normalize_for_compare(intent);
    if normalized.is_empty() {
        return true;
    }

    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.len() == 1 {
        return is_generic_single_word(words[0]);
    }

    false
}

fn looks_like_internal_risk_tag(note: &str) -> bool {
    let lower = note.to_ascii_lowercase();
    lower.contains(':')
        && lower
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '-' | '/'))
}

#[cfg(not(feature = "llama-native"))]
struct MockEngine;

#[cfg(not(feature = "llama-native"))]
impl LlmEngine for MockEngine {
    fn analyze_chunk(&self, chunk: &DiffChunk) -> Result<PartialReport, CoreError> {
        Ok(PartialReport {
            summary: format!("Analyzed {}", chunk.path),
            items: vec![ChangeItem {
                id: format!("item-{}", chunk.path.replace('/', "_")),
                bucket: ChangeBucket::Patch,
                type_tag: TypeTag::Fix,
                title: format!("Update {}", chunk.path),
                intent: "Apply diff chunk updates".to_string(),
                files: vec![FileRef {
                    path: chunk.path.clone(),
                    status: FileStatus::Modified,
                    ranges: chunk.ranges.clone(),
                }],
                confidence: 0.8,
            }],
        })
    }

    fn reduce_report(
        &self,
        partials: &[PartialReport],
        decision: &DispatchDecision,
        stats: &DiffStats,
    ) -> Result<AnalysisReport, CoreError> {
        let mut items = Vec::new();
        for partial in partials {
            items.extend(partial.items.clone());
        }

        Ok(AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "fix(core): synthesize structured analysis".to_string(),
            summary: format!("{} partial analyses reduced", partials.len()),
            items,
            risk: RiskReport {
                level: "low".to_string(),
                notes: vec!["mock engine".to_string()],
            },
            stats: stats.clone(),
            dispatch: decision.clone(),
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        let len = text.len() as f32;
        Ok(vec![(len % 97.0) / 97.0, (len % 53.0) / 53.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autocommit_core::types::{
        ChangeBucket, ChangeItem, DiffStats, DispatchDecision, DispatchRoute, FileRef, FileStatus,
        RiskReport, TypeTag,
    };

    fn sample_report() -> AnalysisReport {
        AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat(core): add detailed commit composition".to_string(),
            summary: "Compose commit output from chunk-level analyses.".to_string(),
            items: vec![
                ChangeItem {
                    id: "a".to_string(),
                    bucket: ChangeBucket::Feature,
                    type_tag: TypeTag::Feat,
                    title: "Compose final commit body".to_string(),
                    intent: "Include per-file details in commit body".to_string(),
                    files: vec![FileRef {
                        path: "crates/cli/src/cmd/commit.rs".to_string(),
                        status: FileStatus::Modified,
                        ranges: Vec::new(),
                    }],
                    confidence: 0.9,
                },
                ChangeItem {
                    id: "b".to_string(),
                    bucket: ChangeBucket::Other,
                    type_tag: TypeTag::Refactor,
                    title: "Refactor output formatting".to_string(),
                    intent: "Refactor output formatting".to_string(),
                    files: vec![
                        FileRef {
                            path: "crates/cli/src/output/text.rs".to_string(),
                            status: FileStatus::Modified,
                            ranges: Vec::new(),
                        },
                        FileRef {
                            path: "crates/core/src/llm/prompts.rs".to_string(),
                            status: FileStatus::Modified,
                            ranges: Vec::new(),
                        },
                    ],
                    confidence: 0.7,
                },
            ],
            risk: RiskReport {
                level: "medium".to_string(),
                notes: vec![
                    "Generated details were composed from partial analyses.".to_string(),
                    "dispatch:DraftThenReduce".to_string(),
                ],
            },
            stats: DiffStats {
                files_changed: 2,
                lines_changed: 42,
                hunks: 4,
                binary_files: 0,
            },
            dispatch: DispatchDecision {
                route: DispatchRoute::DraftThenReduce,
                reason_codes: vec!["test".to_string()],
                estimated_cost_tokens: 0,
            },
        }
    }

    #[test]
    fn compose_commit_message_includes_summary_changes_and_risk() {
        let message = compose_commit_message(&sample_report(), &[]);
        assert!(message.starts_with("feat(core): add detailed commit composition\n\n"));
        assert!(message.contains("Compose commit output from chunk-level analyses."));
        assert!(message.contains("### Changes\n- [`crates/cli/src/cmd/commit.rs`] Compose final commit body: Include per-file details in commit body"));
        assert!(
            message.contains(
                "- [`crates/cli/src/output/text.rs` (+1 more)] Refactor output formatting"
            )
        );
        assert!(message.contains("### Risk\n- Level: medium"));
        assert!(message.contains("- Generated details were composed from partial analyses."));
        assert!(!message.contains("dispatch:DraftThenReduce"));
    }

    #[test]
    fn compose_commit_message_handles_empty_body_sections() {
        let mut report = sample_report();
        report.summary.clear();
        report.items.clear();
        report.risk.level.clear();
        report.risk.notes.clear();

        let message = compose_commit_message(&report, &[]);
        assert_eq!(message, "feat(core): add detailed commit composition");
    }

    #[test]
    fn compose_commit_message_omits_duplicate_summary_from_body() {
        let mut report = sample_report();
        report.commit_message = "refactor(core): simplify reduction logic".to_string();
        report.summary = "Refactor: simplify reduction logic".to_string();
        report.items.clear();
        report.risk.level.clear();
        report.risk.notes.clear();

        let message = compose_commit_message(&report, &[]);
        assert_eq!(message, "refactor(core): simplify reduction logic");
    }

    #[test]
    fn format_change_item_collapses_boilerplate_and_redundant_intent() {
        let item = ChangeItem {
            id: "c".to_string(),
            bucket: ChangeBucket::Feature,
            type_tag: TypeTag::Feat,
            title: "Commit Message Composition and Creation with Analysis Report".to_string(),
            intent: "Implement a feature that composes commit messages based on an analysis report and creates commits with the composed messages.".to_string(),
            files: vec![FileRef {
                path: "crates/cli/src/cmd/commit.rs".to_string(),
                status: FileStatus::Modified,
                ranges: Vec::new(),
            }],
            confidence: 0.8,
        };

        let formatted = format_change_item(&item);
        assert_eq!(
            formatted,
            "[`crates/cli/src/cmd/commit.rs`] Commit Message Composition and Creation with Analysis Report: compose commit messages based on an analysis report and create commits with the composed messages"
        );
    }

    #[test]
    fn compose_commit_message_includes_version_bump_recommendations_section() {
        let report = sample_report();
        let recommendations = vec![version_bump::test_recommendation(
            "Cargo.toml",
            "Rust",
            "Cargo",
            Some("0.1.0"),
            Some("0.2.0"),
            version_bump::BumpLevel::Minor,
            "code changes detected; suggest a minor project version bump",
        )];

        let message = compose_commit_message(&report, &recommendations);
        assert!(message.contains("### Version Bumps"));
        assert!(message.contains("[`Cargo.toml`] Rust / Cargo: 0.1.0 -> 0.2.0 (minor)"));
    }

    #[test]
    fn format_change_item_rewrites_diagnostics_suffix() {
        let item = ChangeItem {
            id: "diag".to_string(),
            bucket: ChangeBucket::Patch,
            type_tag: TypeTag::Fix,
            title: "analyze chunk: diagnostics".to_string(),
            intent: "Add".to_string(),
            files: vec![FileRef {
                path: "crates/llama-runtime/tests/contract.rs".to_string(),
                status: FileStatus::Modified,
                ranges: Vec::new(),
            }],
            confidence: 0.7,
        };

        let formatted = format_change_item(&item);
        assert_eq!(
            formatted,
            "[`crates/llama-runtime/tests/contract.rs`] Add diagnostics for analyze chunk"
        );
    }

    #[test]
    fn format_change_item_wraps_identifier_and_drops_generic_intent() {
        let item = ChangeItem {
            id: "hash".to_string(),
            bucket: ChangeBucket::Patch,
            type_tag: TypeTag::Refactor,
            title: "Implement hash function for diff_text".to_string(),
            intent: "Addition".to_string(),
            files: vec![FileRef {
                path: "crates/cli/src/cmd/report_cache.rs".to_string(),
                status: FileStatus::Modified,
                ranges: Vec::new(),
            }],
            confidence: 0.8,
        };

        let formatted = format_change_item(&item);
        assert_eq!(
            formatted,
            "[`crates/cli/src/cmd/report_cache.rs`] Implement hash function for `diff_text`"
        );
    }
}
