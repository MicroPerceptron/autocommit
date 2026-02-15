use std::fs;
use std::path::Path;

use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};

use crate::cmd::{git, report_cache};
#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;
use crate::output;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut json = false;
    let mut diff_file: Option<String> = None;
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
            "--json" => json = true,
            "--diff-file" => {
                let path = args
                    .get(i + 1)
                    .ok_or_else(|| "--diff-file requires a path".to_string())?;
                diff_file = Some(path.clone());
                i += 1;
            }
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
            flag => return Err(format!("unknown analyze option: {flag}")),
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

    let diff_text = load_diff(diff_file.as_deref()).map_err(|err| err.to_string())?;

    let diff_hash = report_cache::diff_hash(&diff_text);
    let cache_key = report_cache::cache_key("analyze", runtime_profile.as_str(), &diff_hash, "1.0");
    let cache_path = git::Repo::discover()
        .ok()
        .map(|repo| report_cache::cache_path(repo.common_git_dir()));
    if let Some(cache_path) = cache_path.as_ref() {
        if let Some(report) = report_cache::read_cached_report(cache_path, &cache_key) {
            return if json {
                output::json::to_pretty_json(&report).map_err(|err| err.to_string())
            } else {
                Ok(output::text::render_report(&report))
            };
        }
    }

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
    if let Some(cache_path) = cache_path.as_ref() {
        let _ = report_cache::write_cached_report(cache_path, &cache_key, &report);
    }

    if json {
        output::json::to_pretty_json(&report).map_err(|err| err.to_string())
    } else {
        Ok(output::text::render_report(&report))
    }
}

fn load_diff(diff_file: Option<&str>) -> Result<String, CoreError> {
    if let Some(path) = diff_file {
        return Ok(fs::read_to_string(Path::new(path))?);
    }

    read_git_diff()
}

fn read_git_diff() -> Result<String, CoreError> {
    let repo = git::Repo::discover()?;
    let staged = repo.diff_cached()?;
    let unstaged = repo.diff_worktree()?;
    let mut combined = String::new();

    if !staged.trim().is_empty() {
        combined.push_str(&staged);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }

    if !unstaged.trim().is_empty() {
        combined.push_str(&unstaged);
    }

    if combined.trim().is_empty() {
        return Err(CoreError::InvalidDiff(
            "no working tree or staged diff to analyze".to_string(),
        ));
    }

    Ok(combined)
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
