use std::fs;
use std::path::Path;

use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::{AnalyzeOptions, CoreError, run as core_run};

use crate::output;

#[cfg(not(feature = "llama-native"))]
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};

pub fn run(args: &[String]) -> Result<String, String> {
    let mut json = false;
    let mut diff_file: Option<String> = None;

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
            flag => return Err(format!("unknown analyze option: {flag}")),
        }
        i += 1;
    }

    let diff_text = load_diff(diff_file.as_deref()).map_err(|err| err.to_string())?;

    #[cfg(feature = "llama-native")]
    let engine: Box<dyn LlmEngine> = Box::new(
        llama_runtime::Engine::new("default")
            .map_err(|err| format!("runtime init failed: {err}"))?,
    );

    #[cfg(not(feature = "llama-native"))]
    let engine: Box<dyn LlmEngine> = Box::new(MockEngine);

    let report = core_run(engine.as_ref(), &diff_text, &AnalyzeOptions::default())
        .map_err(|err| format!("analysis failed: {err}"))?;

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

    Ok("diff --git a/src/lib.rs b/src/lib.rs\n@@ -1,1 +1,2 @@\n-pub fn old() {}\n+pub fn new() {}\n+pub fn newer() {}\n".to_string())
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
