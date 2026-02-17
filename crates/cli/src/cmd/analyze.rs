use std::fs;
use std::path::Path;
#[cfg(feature = "llama-native")]
use std::path::PathBuf;

use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{run as core_run, AnalyzeOptions, CoreError};
use clap::Parser;

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;
use crate::cmd::{git, report_cache};
use crate::output;
#[cfg(feature = "llama-native")]
use crate::path_util::expand_tilde;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let parsed = match AnalyzeArgs::parse_from(args)? {
        ParseOutcome::Continue(parsed) => parsed,
        ParseOutcome::EarlyExit(text) => return Ok(text),
    };

    let json = parsed.json;
    let diff_file = parsed.diff_file;
    #[allow(unused_mut)]
    let mut model_path = parsed.model_path;
    #[allow(unused_mut)]
    let mut model_hf_repo = parsed.hf_repo;
    #[allow(unused_mut)]
    let mut model_cache_dir = parsed.cache_dir;
    #[cfg(feature = "llama-native")]
    let mut runtime_profile = "auto".to_string();
    #[cfg(feature = "llama-native")]
    let mut runtime_profile_overridden = false;
    #[cfg(not(feature = "llama-native"))]
    let runtime_profile = "mock".to_string();

    if let Some(profile) = parsed.profile {
        #[cfg(feature = "llama-native")]
        {
            runtime_profile = profile;
            runtime_profile_overridden = true;
        }
        #[cfg(not(feature = "llama-native"))]
        {
            let _ = profile;
        }
    }

    if model_path.is_some() && model_hf_repo.is_some() {
        return Err("use either `--model-path` or `--hf-repo`, not both".to_string());
    }
    #[cfg(not(feature = "llama-native"))]
    let _ = &model_cache_dir;

    #[cfg(feature = "llama-native")]
    let repo_paths = repo_cache::maybe_discover_repo_kv_paths();

    #[cfg(feature = "llama-native")]
    if model_path.is_none()
        || model_hf_repo.is_none()
        || model_cache_dir.is_none()
        || !runtime_profile_overridden
    {
        if let Some(metadata) = repo_paths.as_ref().and_then(repo_cache::read_metadata) {
            if model_path.is_none() && model_hf_repo.is_none() {
                model_path = metadata.model_path.clone();
                model_hf_repo = metadata.model_hf_repo.clone();
            }
            if model_cache_dir.is_none() {
                model_cache_dir = metadata.model_cache_dir.clone();
            }
            if !runtime_profile_overridden && !metadata.profile.trim().is_empty() {
                runtime_profile = metadata.profile;
            }
        }
    }

    let diff_text = load_diff(diff_file.as_deref()).map_err(|err| err.to_string())?;
    if let Ok(repo) = git::Repo::discover() {
        if let Some(cache_dir) = repo
            .common_git_dir()
            .join("autocommit/kv/partials")
            .to_str()
        {
            unsafe {
                std::env::set_var("AUTOCOMMIT_PARTIAL_CACHE_DIR", cache_dir);
            }
        }
    }

    let diff_hash = report_cache::diff_hash(&diff_text);
    let cache_key = report_cache::cache_key("analyze", runtime_profile.as_str(), &diff_hash);
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
    let engine: Box<dyn LlmEngine> = Box::new(
        llama_runtime::Engine::new_with_generation_cache_and_model(
            &runtime_profile,
            generation_state,
            model_config,
        )
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

enum ParseOutcome<T> {
    Continue(T),
    EarlyExit(String),
}

#[derive(Parser, Debug)]
#[command(
    name = "autocommit-cli analyze",
    about = "Analyze staged/worktree changes and emit a structured report"
)]
struct AnalyzeArgs {
    /// Render output as JSON
    #[arg(long)]
    json: bool,
    /// Read unified diff text from a file instead of git
    #[arg(long = "diff-file", value_name = "PATH")]
    diff_file: Option<String>,
    /// Explicit local model path (`.gguf`)
    #[arg(long = "model-path", value_name = "PATH")]
    model_path: Option<String>,
    /// Hugging Face model repo (`org/model` or `org/model:file`)
    #[arg(long = "hf-repo", value_name = "REPO")]
    hf_repo: Option<String>,
    /// Override llama.cpp model cache directory
    #[arg(long = "cache-dir", value_name = "PATH")]
    cache_dir: Option<String>,
    /// Runtime profile (`auto`, etc.)
    #[arg(long = "profile", value_name = "PROFILE")]
    profile: Option<String>,
}

impl AnalyzeArgs {
    fn parse_from(args: &[String]) -> Result<ParseOutcome<Self>, String> {
        let argv =
            std::iter::once("autocommit-cli analyze".to_string()).chain(args.iter().cloned());
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
