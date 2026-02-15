use std::io::{IsTerminal, Write};
use std::process::Command;

use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};
use dialoguer::console::{Term, style};
use dialoguer::{Confirm, Editor, Select, theme::ColorfulTheme};
use indicatif::{ProgressBar, ProgressStyle};

use crate::cmd::git;
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

    let diff_text = run_step(rich_interactive, "Collecting staged/worktree diff", || {
        prepare_diff(&repo, staged_only, dry_run)
    })
    .map_err(|err| err.to_string())?;

    #[cfg(feature = "llama-native")]
    let generation_state = repo_paths.map(|paths| paths.generation_state);

    #[cfg(feature = "llama-native")]
    let engine: Box<dyn LlmEngine> =
        run_step(rich_interactive, "Initializing llama runtime", || {
            llama_runtime::Engine::new_with_generation_cache(&runtime_profile, generation_state)
                .map(|engine| Box::new(engine) as Box<dyn LlmEngine>)
        })
        .map_err(|err| format!("runtime init failed: {err}"))?;

    #[cfg(not(feature = "llama-native"))]
    let engine: Box<dyn LlmEngine> =
        run_step(rich_interactive, "Initializing analysis engine", || {
            Ok::<Box<dyn LlmEngine>, CoreError>(Box::new(MockEngine))
        })
        .map_err(|err| format!("runtime init failed: {err}"))?;

    let report = run_step(rich_interactive, "Generating PR analysis", || {
        core_run(engine.as_ref(), &diff_text, &AnalyzeOptions::default())
    })
    .map_err(|err| format!("analysis failed: {err}"))?;

    let generated_title = title
        .clone()
        .unwrap_or_else(|| report.commit_message.trim().to_string());
    let generated_body = body.clone().unwrap_or_else(|| compose_pr_body(&report));

    if generated_title.trim().is_empty() {
        return Err("generated pull request title is empty".to_string());
    }

    let (base, head) = resolve_pr_branches(
        &repo,
        base,
        head,
        interactive,
        rich_interactive,
    )?;

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

    if push {
        run_step(rich_interactive, "Pushing branch", || repo.push())
            .map_err(|err| err.to_string())?;
    } else if interactive && !assume_yes {
        let should_push = prompt_should_push(rich_interactive)?;
        if should_push {
            run_step(rich_interactive, "Pushing branch", || repo.push())
                .map_err(|err| err.to_string())?;
        }
    }

    let output = run_step(rich_interactive, "Creating pull request", || {
        create_pr(&final_title, &final_body, draft, base.as_deref(), head.as_deref())
    })
    .map_err(|err| err.to_string())?;

    Ok(output)
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

    let branches = collect_branch_options(repo)?;
    let current_branch = repo.current_branch().map_err(|err| err.to_string())?;

    let base = match base {
        Some(value) => Some(value),
        None => {
            let default = pick_default_base(&branches);
            if interactive {
                Some(prompt_for_branch(
                    "Select base branch",
                    &branches,
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
                    "Select head branch",
                    &branches,
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

fn pick_default_base(branches: &[String]) -> Option<String> {
    for candidate in ["main", "master", "trunk", "develop"] {
        if branches.iter().any(|branch| branch == candidate) {
            return Some(candidate.to_string());
        }
        let remote_candidate = format!("origin/{candidate}");
        if branches.iter().any(|branch| branch == &remote_candidate) {
            return Some(remote_candidate);
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

fn collect_branch_options(repo: &git::Repo) -> Result<Vec<String>, String> {
    let mut branches = repo.local_branches().map_err(|err| err.to_string())?;
    let remotes = repo.remote_branches().map_err(|err| err.to_string())?;
    for remote in remotes {
        if !branches.contains(&remote) {
            branches.push(remote);
        }
    }
    branches.sort();
    Ok(branches)
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

fn prepare_diff(repo: &git::Repo, staged_only: bool, dry_run: bool) -> Result<String, CoreError> {
    if staged_only {
        let staged = repo.diff_cached()?;
        if staged.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no staged changes to include".to_string(),
            ));
        }
        return Ok(staged);
    }

    let staged = repo.diff_cached()?;
    let unstaged = repo.diff_worktree()?;
    let combined = concat_diffs(&staged, &unstaged);
    if combined.trim().is_empty() {
        return Err(CoreError::InvalidDiff(
            "no changes in working tree".to_string(),
        ));
    }

    if dry_run {
        return Ok(combined);
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
