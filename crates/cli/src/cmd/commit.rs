use std::process::Command;

use autocommit_core::AnalysisReport;
use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;
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
            "--json" => json = true,
            "--no-verify" => no_verify = true,
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
            flag => return Err(format!("unknown commit option: {flag}")),
        }
        i += 1;
    }

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

    let diff_text = prepare_diff(staged_only, dry_run).map_err(|err| err.to_string())?;

    #[cfg(feature = "llama-native")]
    let generation_state = repo_paths.map(|paths| paths.generation_state);

    #[cfg(feature = "llama-native")]
    let engine: Box<dyn LlmEngine> = Box::new(
        llama_runtime::Engine::new_with_generation_cache(&runtime_profile, generation_state)
            .map_err(|err| format!("runtime init failed: {err}"))?,
    );

    #[cfg(not(feature = "llama-native"))]
    let engine: Box<dyn LlmEngine> = Box::new(MockEngine);

    let report = core_run(engine.as_ref(), &diff_text, &AnalyzeOptions::default())
        .map_err(|err| format!("analysis failed: {err}"))?;

    if dry_run {
        if json {
            return output::json::to_pretty_json(&report).map_err(|err| err.to_string());
        }

        let composed_message = compose_commit_message(&report);
        let mut out = String::new();
        out.push_str("dry-run: commit was not created\n");
        out.push_str(&format!("message:\n{}\n", composed_message));
        out.push_str(&output::text::render_report(&report));
        return Ok(out);
    }

    let composed_message = compose_commit_message(&report);
    commit_with_message(&composed_message, no_verify).map_err(|err| err.to_string())?;

    if push {
        run_git(&["push"]).map_err(|err| err.to_string())?;
    }

    if json {
        output::json::to_pretty_json(&report).map_err(|err| err.to_string())
    } else {
        Ok(format!(
            "created commit with message:\n{}\n",
            composed_message
        ))
    }
}

fn prepare_diff(staged_only: bool, dry_run: bool) -> Result<String, CoreError> {
    if staged_only {
        let staged = run_git(&["diff", "--cached"])?;
        if staged.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no staged changes to commit".to_string(),
            ));
        }
        return Ok(staged);
    }

    if dry_run {
        let staged = run_git(&["diff", "--cached"])?;
        let unstaged = run_git(&["diff"])?;
        let combined = concat_diffs(&staged, &unstaged);
        if combined.trim().is_empty() {
            return Err(CoreError::InvalidDiff(
                "no changes in working tree".to_string(),
            ));
        }
        return Ok(combined);
    }

    run_git(&["add", "-A"])?;
    let staged = run_git(&["diff", "--cached"])?;
    if staged.trim().is_empty() {
        return Err(CoreError::InvalidDiff(
            "no changes staged after git add -A".to_string(),
        ));
    }

    Ok(staged)
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

fn commit_with_message(message: &str, no_verify: bool) -> Result<(), CoreError> {
    let mut lines = message.lines();
    let subject = lines.next().unwrap_or_default().trim();
    if subject.is_empty() {
        return Err(CoreError::InvalidDiff(
            "generated commit subject is empty".to_string(),
        ));
    }

    let body = lines.collect::<Vec<_>>().join("\n");
    let body = body.trim();

    let mut args = vec!["commit", "-m", subject];
    if !body.is_empty() {
        args.push("-m");
        args.push(body);
    }
    if no_verify {
        args.push("--no-verify");
    }
    run_git(&args)?;
    Ok(())
}

fn compose_commit_message(report: &AnalysisReport) -> String {
    let subject = report.commit_message.trim();
    if subject.is_empty() {
        return String::new();
    }

    let body = compose_commit_body(report);
    if body.is_empty() {
        subject.to_string()
    } else {
        format!("{subject}\n\n{body}")
    }
}

fn compose_commit_body(report: &AnalysisReport) -> String {
    let mut sections = Vec::new();

    let summary = report.summary.trim();
    if !summary.is_empty() {
        sections.push(summary.to_string());
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

fn compose_changes_section(report: &AnalysisReport) -> String {
    if report.items.is_empty() {
        return String::new();
    }

    let mut out = String::from("Changes:\n");
    for item in &report.items {
        out.push_str("- ");
        out.push_str(&format_change_item(item));
        out.push('\n');
    }
    out.trim_end().to_string()
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

    let mut out = String::new();
    if !level.is_empty() {
        out.push_str(&format!("Risk: {level}\n"));
    }

    if !notes.is_empty() {
        out.push_str("Risk Notes:\n");
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

    let title = item.title.trim();
    let intent = item.intent.trim();
    let suffix = if intent.is_empty() || intent.eq_ignore_ascii_case(title) {
        String::new()
    } else {
        format!(": {intent}")
    };

    format!("[{path}{file_suffix}] {title}{suffix}")
}

fn looks_like_internal_risk_tag(note: &str) -> bool {
    let lower = note.to_ascii_lowercase();
    lower.contains(':')
        && lower
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '-' | '/'))
}

fn run_git(args: &[&str]) -> Result<String, CoreError> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|err| CoreError::Io(format!("failed to run git {}: {err}", args.join(" "))))?;

    if !output.status.success() {
        return Err(CoreError::Io(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
        let message = compose_commit_message(&sample_report());
        assert!(message.starts_with("feat(core): add detailed commit composition\n\n"));
        assert!(message.contains("Compose commit output from chunk-level analyses."));
        assert!(message.contains("Changes:\n- [crates/cli/src/cmd/commit.rs] Compose final commit body: Include per-file details in commit body"));
        assert!(
            message
                .contains("- [crates/cli/src/output/text.rs (+1 more)] Refactor output formatting")
        );
        assert!(message.contains("Risk: medium"));
        assert!(
            message
                .contains("Risk Notes:\n- Generated details were composed from partial analyses.")
        );
        assert!(!message.contains("dispatch:DraftThenReduce"));
    }

    #[test]
    fn compose_commit_message_handles_empty_body_sections() {
        let mut report = sample_report();
        report.summary.clear();
        report.items.clear();
        report.risk.level.clear();
        report.risk.notes.clear();

        let message = compose_commit_message(&report);
        assert_eq!(message, "feat(core): add detailed commit composition");
    }
}
