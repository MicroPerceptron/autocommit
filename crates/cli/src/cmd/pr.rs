use std::io::{IsTerminal, Write};
use std::process::Command;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};
use dialoguer::console::{Term, style};
use dialoguer::{Confirm, Editor, Select, theme::ColorfulTheme};
use serde::Deserialize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cmd::{analysis_progress::AnalysisProgress, git, report_cache};
#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut staged_only = false;
    let mut push = false;
    let mut dry_run = false;
    let mut draft = false;
    let mut base: Option<String> = None;
    let mut head: Option<String> = None;
    let mut title: Option<String> = None;
    let mut body: Option<String> = None;
    let mut interactive_override: Option<bool> = None;
    let mut assume_yes = false;
    let mut model_path: Option<String> = None;
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
            "--draft" => draft = true,
            "--base" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "--base requires a value".to_string())?;
                base = Some(value.clone());
                i += 1;
            }
            "--head" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "--head requires a value".to_string())?;
                head = Some(value.clone());
                i += 1;
            }
            "--title" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "--title requires a value".to_string())?;
                title = Some(value.clone());
                i += 1;
            }
            "--body" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "--body requires a value".to_string())?;
                body = Some(value.clone());
                i += 1;
            }
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
            flag => return Err(format!("unknown pr option: {flag}")),
        }
        i += 1;
    }

    let interactive = resolve_interactive_mode(interactive_override)?;
    let rich_interactive = interactive && Term::stderr().is_term();

    #[cfg(feature = "llama-native")]
    let repo_paths = repo_cache::maybe_discover_repo_kv_paths();

    #[cfg(feature = "llama-native")]
    if model_path.is_none() || !runtime_profile_overridden {
        if let Some(metadata) = repo_paths.as_ref().and_then(repo_cache::read_metadata) {
            if model_path.is_none() {
                model_path = metadata.model_path;
            }
            if !runtime_profile_overridden && !metadata.profile.trim().is_empty() {
                runtime_profile = metadata.profile;
            }
        }
    }

    if let Some(path) = model_path {
        // SAFETY: this CLI is single-threaded for command setup and sets env before runtime init.
        unsafe {
            std::env::set_var("AUTOCOMMIT_EMBED_MODEL", path);
        }
    }

    let repo = run_step(rich_interactive, "Discovering repository", git::Repo::discover)
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

    let remote_names = repo.remote_names().map_err(|err| err.to_string())?;

    let (base, head) = resolve_pr_branches(
        &repo,
        base,
        head,
        interactive,
        rich_interactive,
    )?;

    let diff_text = run_step(rich_interactive, "Collecting PR diff", || {
        prepare_pr_diff(&repo, staged_only, base.as_deref(), head.as_deref())
    })
    .map_err(|err| err.to_string())?;

    let diff_hash = report_cache::diff_hash(&diff_text);
    let cache_key = report_cache::cache_key("pr", runtime_profile.as_str(), &diff_hash, "1.0");
    let cache_path = report_cache::cache_path(repo.common_git_dir());
    let cached_report = report_cache::read_cached_report(&cache_path, &cache_key);

    let report = if let Some(report) = cached_report {
        if rich_interactive {
            println!("[ok] Using cached analysis");
        }
        report
    } else {
        #[cfg(feature = "llama-native")]
        let generation_state = repo_paths.map(|paths| paths.generation_state);

        #[cfg(feature = "llama-native")]
        let engine = run_step(rich_interactive, "Initializing llama runtime", || {
            llama_runtime::Engine::new_with_generation_cache(&runtime_profile, generation_state)
        })
        .map_err(|err| format!("runtime init failed: {err}"))?;

        #[cfg(not(feature = "llama-native"))]
        let engine = run_step(rich_interactive, "Initializing analysis engine", || {
            Ok::<MockEngine, CoreError>(MockEngine)
        })
        .map_err(|err| format!("runtime init failed: {err}"))?;

        let progress = if rich_interactive {
            Some(AnalysisProgress::new(&diff_text))
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

    let generated_title = title
        .clone()
        .unwrap_or_else(|| report.commit_message.trim().to_string());
    let generated_body = body.clone().unwrap_or_else(|| compose_pr_body(&report));

    if generated_title.trim().is_empty() {
        return Err("generated pull request title is empty".to_string());
    }

    let draft = if draft {
        true
    } else if interactive && !assume_yes {
        prompt_should_draft(rich_interactive)?
    } else {
        false
    };

    if dry_run {
        return Ok(render_dry_run(
            &generated_title,
            &generated_body,
            draft,
            base.as_deref(),
            head.as_deref(),
            push,
        ));
    }

    let (final_title, final_body) = if interactive && !assume_yes {
        match prompt_for_pr_content(&generated_title, &generated_body, rich_interactive)? {
            Some(value) => value,
            None => return Ok("pull request canceled by user\n".to_string()),
        }
    } else {
        (generated_title, generated_body)
    };

    let gh_base = base
        .as_deref()
        .map(|value| normalize_branch_for_gh(value, &remote_names));
    let gh_head = head
        .as_deref()
        .map(|value| normalize_branch_for_gh(value, &remote_names));

    let head_requires_push = if let Some(head_value) = head.as_deref() {
        let remote = extract_remote_prefix(head_value, &remote_names);
        if let Some(remote) = remote {
            let branch = normalize_branch_for_gh(head_value, &remote_names);
            !repo
                .remote_branch_exists(&remote, &branch)
                .map_err(|err| err.to_string())?
        } else {
            !repo
                .remote_branch_exists("origin", head_value)
                .map_err(|err| err.to_string())?
        }
    } else {
        false
    };

    let mut should_push = push;
    if head_requires_push && !should_push {
        if interactive && !assume_yes {
            should_push = prompt_should_push(rich_interactive)?;
        } else {
            return Err("head branch is not on remote; rerun with --push".to_string());
        }
    }

    if should_push {
        run_step(rich_interactive, "Pushing branch", || repo.push())
            .map_err(|err| err.to_string())?;
    }

    if let Some(existing) = find_existing_pr(gh_head.as_deref(), gh_base.as_deref())? {
        let policy = if interactive && !assume_yes {
            prompt_existing_pr_policy(&existing, rich_interactive)?
        } else if assume_yes {
            ExistingPrPolicy::Update
        } else {
            return Err(format!(
                "an open pull request already exists (#{}) for this branch; rerun with --interactive or --yes",
                existing.number
            ));
        };

        match policy {
            ExistingPrPolicy::Update => {
                let output = run_step(rich_interactive, "Updating pull request", || {
                    update_pr(existing.number, &final_title, &final_body)
                })
                .map_err(|err| err.to_string())?;
                return Ok(output);
            }
            ExistingPrPolicy::CreateNew => {}
            ExistingPrPolicy::Cancel => {
                return Ok("pull request canceled by user\n".to_string());
            }
        }
    }

    let output = run_step(rich_interactive, "Creating pull request", || {
        create_pr(
            &final_title,
            &final_body,
            draft,
            gh_base.as_deref(),
            gh_head.as_deref(),
        )
    })
    .map_err(|err| err.to_string())?;

    Ok(output)
}

#[derive(Debug, Deserialize)]
struct ExistingPr {
    number: u64,
    title: String,
}

#[derive(Debug, Clone, Copy)]
enum ExistingPrPolicy {
    Update,
    CreateNew,
    Cancel,
}

fn find_existing_pr(head: Option<&str>, base: Option<&str>) -> Result<Option<ExistingPr>, String> {
    let Some(head) = head else {
        return Ok(None);
    };

    let mut cmd = Command::new("gh");
    cmd.args([
        "pr",
        "list",
        "--state",
        "open",
        "--head",
        head,
        "--json",
        "number,title",
    ]);

    if let Some(base) = base {
        cmd.args(["--base", base]);
    }

    let output = cmd
        .output()
        .map_err(|err| format!("failed to run gh pr list: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("gh pr list failed: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut list: Vec<ExistingPr> =
        serde_json::from_str(stdout.trim()).map_err(|err| {
            format!("failed to parse gh pr list output: {err}")
        })?;

    if list.is_empty() {
        return Ok(None);
    }

    list.sort_by_key(|pr| pr.number);
    Ok(list.into_iter().next())
}

fn prompt_existing_pr_policy(pr: &ExistingPr, rich: bool) -> Result<ExistingPrPolicy, String> {
    let prompt = format!(
        "An open pull request already exists (#{}: {})",
        pr.number, pr.title
    );

    if rich {
        let options = ["Update existing PR", "Create new PR", "Cancel"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt(prompt)
            .items(options)
            .default(0)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| format!("failed to read PR policy: {err}"))?;

        return Ok(match selection {
            Some(0) => ExistingPrPolicy::Update,
            Some(1) => ExistingPrPolicy::CreateNew,
            _ => ExistingPrPolicy::Cancel,
        });
    }

    loop {
        println!("{prompt}");
        println!("  [u] Update existing PR");
        println!("  [n] Create new PR");
        println!("  [c] Cancel");
        print!("Select action [u/n/c]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let choice = read_line_trimmed()?;
        match choice.as_str() {
            "u" | "U" => return Ok(ExistingPrPolicy::Update),
            "n" | "N" => return Ok(ExistingPrPolicy::CreateNew),
            "c" | "C" => return Ok(ExistingPrPolicy::Cancel),
            _ => println!("invalid choice, enter u, n, or c"),
        }
    }
}

fn update_pr(number: u64, title: &str, body: &str) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr",
            "edit",
            &number.to_string(),
            "--title",
            title,
            "--body",
            body,
        ])
        .output()
        .map_err(|err| format!("failed to run gh pr edit: {err}"))?;

    if output.status.success() {
        let mut msg = String::new();
        msg.push_str("updated pull request\n");
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            msg.push_str(stdout.trim());
            msg.push('\n');
        }
        return Ok(msg);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let fallback = Command::new("gh")
        .args([
            "api",
            "-X",
            "PATCH",
            &format!("repos/:owner/:repo/pulls/{number}"),
            "-f",
            &format!("title={title}"),
            "-f",
            &format!("body={body}"),
        ])
        .output()
        .map_err(|err| format!("failed to run gh api: {err}"))?;

    if !fallback.status.success() {
        let fallback_err = String::from_utf8_lossy(&fallback.stderr);
        return Err(format!(
            "gh pr edit failed: {stderr}; gh api fallback failed: {}",
            fallback_err.trim()
        ));
    }

    Ok("updated pull request (via gh api)\n".to_string())
}

fn resolve_interactive_mode(interactive_override: Option<bool>) -> Result<bool, String> {
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

fn render_dry_run(
    title: &str,
    body: &str,
    draft: bool,
    base: Option<&str>,
    head: Option<&str>,
    push: bool,
) -> String {
    let mut out = String::new();
    out.push_str("dry-run: pull request was not created\n");
    out.push_str("title:\n");
    out.push_str(title);
    out.push('\n');
    out.push_str("body:\n");
    out.push_str(body);
    out.push('\n');
    out.push_str("command:\n");
    out.push_str("gh pr create");
    if draft {
        out.push_str(" --draft");
    }
    if let Some(base) = base {
        out.push_str(" --base ");
        out.push_str(base);
    }
    if let Some(head) = head {
        out.push_str(" --head ");
        out.push_str(head);
    }
    out.push_str(" --title <...> --body <...>");
    if push {
        out.push_str("\n(push before create: true)");
    }
    out.push('\n');
    out
}

fn prompt_for_pr_content(
    title: &str,
    body: &str,
    rich: bool,
) -> Result<Option<(String, String)>, String> {
    if !rich {
        return prompt_for_pr_content_basic(title, body);
    }

    let term = Term::stderr();
    let theme = ColorfulTheme::default();
    let options = ["Create PR", "Edit title/body", "Cancel"];

    let mut current_title = title.to_string();
    let mut current_body = body.to_string();

    loop {
        print_pr_preview(&current_title, &current_body, true);
        let selection = Select::with_theme(&theme)
            .with_prompt("Pull request action")
            .items(options)
            .default(0)
            .interact_on_opt(&term)
            .map_err(|err| format!("failed to read pull request action: {err}"))?;

        match selection {
            Some(0) => return Ok(Some((current_title, current_body))),
            Some(1) => {
                let edited_title = Editor::new()
                    .extension(".txt")
                    .edit(&current_title)
                    .map_err(|err| format!("failed to open editor: {err}"))?;
                if let Some(next_title) = edited_title {
                    let trimmed = next_title.trim();
                    if !trimmed.is_empty() {
                        current_title = trimmed.to_string();
                    }
                }

                let edited_body = Editor::new()
                    .extension(".md")
                    .edit(&current_body)
                    .map_err(|err| format!("failed to open editor: {err}"))?;
                if let Some(next_body) = edited_body {
                    current_body = next_body.trim().to_string();
                }
            }
            Some(2) | None => return Ok(None),
            Some(_) => return Err("invalid pull request action selection".to_string()),
        }
    }
}

fn prompt_for_pr_content_basic(
    title: &str,
    body: &str,
) -> Result<Option<(String, String)>, String> {
    let mut current_title = title.to_string();
    let mut current_body = body.to_string();

    loop {
        print_pr_preview(&current_title, &current_body, false);
        println!("Pull request action:");
        println!("  [c] Create PR");
        println!("  [e] Edit title/body");
        println!("  [x] Cancel");
        print!("Select action [c/e/x]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;

        let choice = read_line_trimmed()?;
        match choice.as_str() {
            "c" | "C" => return Ok(Some((current_title, current_body))),
            "e" | "E" => {
                let edited_title = Editor::new()
                    .extension(".txt")
                    .edit(&current_title)
                    .map_err(|err| format!("failed to open editor: {err}"))?;
                if let Some(next_title) = edited_title {
                    let trimmed = next_title.trim();
                    if !trimmed.is_empty() {
                        current_title = trimmed.to_string();
                    }
                }

                let edited_body = Editor::new()
                    .extension(".md")
                    .edit(&current_body)
                    .map_err(|err| format!("failed to open editor: {err}"))?;
                if let Some(next_body) = edited_body {
                    current_body = next_body.trim().to_string();
                }
            }
            "x" | "X" => return Ok(None),
            _ => println!("invalid choice, enter c, e, or x"),
        }
    }
}

fn prompt_should_push(rich: bool) -> Result<bool, String> {
    if rich {
        return Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Push branch to remote now?")
            .default(true)
            .interact_on(&Term::stderr())
            .map_err(|err| format!("failed to read push confirmation: {err}"));
    }

    loop {
        print!("Push branch to remote now? [Y/n]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(true);
        }
        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn prompt_should_draft(rich: bool) -> Result<bool, String> {
    if rich {
        return Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Create as draft pull request?")
            .default(false)
            .interact_on(&Term::stderr())
            .map_err(|err| format!("failed to read draft confirmation: {err}"));
    }

    loop {
        print!("Create as draft pull request? [y/N]: ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(false);
        }
        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn print_pr_preview(title: &str, body: &str, rich: bool) {
    if !rich {
        println!("\nProposed pull request:\n\n{title}\n\n{body}\n");
        return;
    }

    println!();
    println!("{}", style("Proposed pull request").bold().underlined().cyan());
    println!("{}", style("----------------------------------------").dim());
    if !title.trim().is_empty() {
        println!("{}", style(title).bold().green());
    }
    if !body.trim().is_empty() {
        println!();
        for line in body.lines() {
            if line.starts_with("### ") || line.ends_with(':') {
                println!("{}", style(line).bold().yellow());
            } else if line.starts_with("- [") {
                println!("{}", style(line).cyan());
            } else if line.starts_with("- ") {
                println!("{}", style(line).blue());
            } else {
                println!("{}", style(line).white());
            }
        }
    }
    println!("{}", style("----------------------------------------").dim());
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

fn resolve_pr_branches(
    repo: &git::Repo,
    base: Option<String>,
    head: Option<String>,
    interactive: bool,
    rich: bool,
) -> Result<(Option<String>, Option<String>), String> {
    if base.is_some() && head.is_some() {
        if base == head {
            return Err("base and head branches cannot be the same".to_string());
        }
        return Ok((base, head));
    }

    let current_branch = repo.current_branch().map_err(|err| err.to_string())?;
    let base_options = collect_base_branch_options(repo)?;
    let head_options = collect_head_branch_options(repo, current_branch.as_deref())?;
    let remote_default = repo
        .remote_default_branch()
        .map_err(|err| err.to_string())?;

    let base = match base {
        Some(value) => Some(value),
        None => {
            let default = pick_default_base(&base_options, remote_default.as_deref());
            if interactive {
                Some(prompt_for_branch(
                    "Select destination branch",
                    &base_options,
                    default.as_deref(),
                    rich,
                )?)
            } else {
                default.or_else(|| current_branch.clone())
            }
        }
    };

    let head = match head {
        Some(value) => Some(value),
        None => {
            if interactive {
                Some(prompt_for_branch(
                    "Select source branch (current)",
                    &head_options,
                    current_branch.as_deref(),
                    rich,
                )?)
            } else {
                current_branch.clone()
            }
        }
    };

    if base.is_none() || head.is_none() {
        return Err("unable to resolve base/head branch; pass --base and --head".to_string());
    }
    if base == head {
        return Err("base and head branches cannot be the same".to_string());
    }

    Ok((base, head))
}

fn pick_default_base(branches: &[String], remote_default: Option<&str>) -> Option<String> {
    if let Some(default) = remote_default {
        if branches.iter().any(|branch| branch == default) {
            return Some(default.to_string());
        }
    }
    for candidate in ["main", "master", "trunk", "develop"] {
        let remote_candidate = format!("origin/{candidate}");
        if branches.iter().any(|branch| branch == &remote_candidate) {
            return Some(remote_candidate);
        }
        if branches.iter().any(|branch| branch == candidate) {
            return Some(candidate.to_string());
        }
    }
    branches.first().cloned()
}

fn prompt_for_branch(
    prompt: &str,
    branches: &[String],
    default: Option<&str>,
    rich: bool,
) -> Result<String, String> {
    if branches.is_empty() {
        return prompt_for_branch_manual(prompt);
    }

    let mut options = branches.to_vec();
    options.push("Enter manually".to_string());
    let default_index = default
        .and_then(|value| options.iter().position(|opt| opt == value))
        .unwrap_or(0);

    if rich {
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt(prompt)
            .items(&options)
            .default(default_index)
            .interact_on_opt(&Term::stderr())
            .map_err(|err| format!("failed to read branch selection: {err}"))?;

        match selection {
            Some(index) if index < branches.len() => Ok(options[index].clone()),
            Some(_) => prompt_for_branch_manual(prompt),
            None => prompt_for_branch_manual(prompt),
        }
    } else {
        println!("{prompt}:");
        for (idx, option) in options.iter().enumerate() {
            println!("  [{}] {}", idx + 1, option);
        }
        print!("Select branch [1-{}]: ", options.len());
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        let index = value.parse::<usize>().unwrap_or(0);
        if index == 0 || index > options.len() {
            return Err("invalid branch selection".to_string());
        }
        if index - 1 < branches.len() {
            Ok(options[index - 1].clone())
        } else {
            prompt_for_branch_manual(prompt)
        }
    }
}

fn collect_base_branch_options(repo: &git::Repo) -> Result<Vec<String>, String> {
    let mut remote_branches = repo.remote_branches().map_err(|err| err.to_string())?;
    let mut locals = repo.local_branches().map_err(|err| err.to_string())?;
    remote_branches.sort();
    locals.sort();

    let mut ordered = Vec::new();
    // Prefer origin/* for base selection, then other remotes, then locals.
    let mut origin_first = Vec::new();
    let mut other_remotes = Vec::new();
    for branch in remote_branches {
        if branch.starts_with("origin/") {
            origin_first.push(branch);
        } else {
            other_remotes.push(branch);
        }
    }
    origin_first.sort();
    other_remotes.sort();
    ordered.extend(origin_first);
    ordered.extend(other_remotes);
    for local in locals {
        if !ordered.contains(&local) {
            ordered.push(local);
        }
    }

    Ok(ordered)
}

fn collect_head_branch_options(
    repo: &git::Repo,
    current_branch: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut locals = repo.local_branches().map_err(|err| err.to_string())?;
    let mut remotes = repo.remote_branches().map_err(|err| err.to_string())?;
    locals.sort();
    remotes.sort();

    let mut ordered = Vec::new();
    if let Some(current) = current_branch {
        ordered.push(current.to_string());
    }
    for local in locals {
        if !ordered.contains(&local) {
            ordered.push(local);
        }
    }
    for remote in remotes {
        if !ordered.contains(&remote) {
            ordered.push(remote);
        }
    }

    Ok(ordered)
}

fn prompt_for_branch_manual(prompt: &str) -> Result<String, String> {
    loop {
        print!("{prompt} (manual entry): ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if !value.is_empty() {
            return Ok(value);
        }
        println!("branch name cannot be empty");
    }
}

fn normalize_branch_for_gh(value: &str, remotes: &[String]) -> String {
    for remote in remotes {
        let prefix = format!("{remote}/");
        if let Some(stripped) = value.strip_prefix(&prefix) {
            return stripped.to_string();
        }
    }
    value.to_string()
}

fn extract_remote_prefix(value: &str, remotes: &[String]) -> Option<String> {
    for remote in remotes {
        let prefix = format!("{remote}/");
        if value.starts_with(&prefix) {
            return Some(remote.to_string());
        }
    }
    None
}

fn prepare_pr_diff(
    repo: &git::Repo,
    staged_only: bool,
    base: Option<&str>,
    head: Option<&str>,
) -> Result<String, CoreError> {
    if staged_only {
        let staged = repo.diff_cached()?;
        if staged.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no staged changes to include".to_string(),
            ));
        }
        return Ok(staged);
    }

    if let (Some(base), Some(head)) = (base, head) {
        let diff = repo.diff_range(base, head)?;
        if diff.trim().is_empty() {
            return Err(CoreError::InvalidDiff(format!(
                "no commits between {base} and {head}"
            )));
        }
        return Ok(diff);
    }

    let staged = repo.diff_cached()?;
    let unstaged = repo.diff_worktree()?;
    let combined = concat_diffs(&staged, &unstaged);
    if combined.trim().is_empty() {
        return Err(CoreError::InvalidDiff(
            "no changes between base/head and no local diffs to include".to_string(),
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

fn compose_pr_body(report: &autocommit_core::AnalysisReport) -> String {
    let mut sections = Vec::new();

    let summary = report.summary.trim();
    if !summary.is_empty() {
        sections.push(format!("### Summary\n{summary}"));
    }

    let changes = compose_changes_section(report);
    if !changes.is_empty() {
        sections.push(changes);
    }

    let risk = compose_risk_section(report);
    if !risk.is_empty() {
        sections.push(risk);
    }

    sections.join("\n\n")
}

fn compose_changes_section(report: &autocommit_core::types::AnalysisReport) -> String {
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

fn compose_risk_section(report: &autocommit_core::types::AnalysisReport) -> String {
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
    let title = clamp_words(&title, 88);
    let title = if title.is_empty() {
        "update file".to_string()
    } else {
        title
    };

    let intent = normalize_change_fragment(item.intent.trim());
    let intent = clamp_words(&intent, 132);

    let suffix = if should_include_intent_detail(&title, &intent) {
        format!(": {intent}")
    } else {
        String::new()
    };

    format!("[{path}{file_suffix}] {title}{suffix}")
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

fn looks_like_internal_risk_tag(note: &str) -> bool {
    let lower = note.to_ascii_lowercase();
    lower.contains(':')
        && lower
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '-' | '/'))
}

fn create_pr(
    title: &str,
    body: &str,
    draft: bool,
    base: Option<&str>,
    head: Option<&str>,
) -> Result<String, CoreError> {
    let mut cmd = Command::new("gh");
    cmd.arg("pr").arg("create").arg("--title").arg(title).arg("--body").arg(body);
    if draft {
        cmd.arg("--draft");
    }
    if let Some(base) = base {
        cmd.arg("--base").arg(base);
    }
    if let Some(head) = head {
        cmd.arg("--head").arg(head);
    }

    let output = cmd
        .output()
        .map_err(|err| CoreError::Io(format!("failed to run gh pr create: {err}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut detail = String::new();
        if !stderr.is_empty() {
            detail.push_str(&stderr);
        }
        if !stdout.is_empty() {
            if !detail.is_empty() {
                detail.push_str("; ");
            }
            detail.push_str(&stdout);
        }
        if detail.is_empty() {
            detail.push_str("unknown error");
        }
        return Err(CoreError::Io(format!("gh pr create failed: {detail}")));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
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
