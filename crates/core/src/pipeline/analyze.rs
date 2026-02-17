use crate::CoreError;
use crate::diff::collect;
use crate::diff::features::{self, DiffFeatures};
use crate::dispatch::{embedding_gate, heuristics, policy};
use crate::llm::prompts;
use crate::llm::traits::LlmEngine;
use crate::pipeline::{fanout, reduce, validate};
use crate::types::{AnalysisReport, DiffStats, DispatchDecision, DispatchRoute};

#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    pub embedding_threshold: f32,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            embedding_threshold: 0.72,
        }
    }
}

pub fn run(
    engine: &dyn LlmEngine,
    diff_text: &str,
    _options: &AnalyzeOptions,
) -> Result<AnalysisReport, CoreError> {
    let chunks = collect::collect(diff_text);

    if chunks.is_empty() {
        return Ok(AnalysisReport::empty(DispatchDecision {
            route: DispatchRoute::DraftOnly,
            reason_codes: vec!["empty_diff".to_string()],
            estimated_cost_tokens: 0,
        }));
    }

    let features = features::extract(&chunks);
    let stats = to_stats(&features);

    let heuristic = heuristics::score(&features);
    let embedding_hint = if embedding_gate::should_run_embedding(heuristic) {
        let prompt = prompts::build_embedding_prompt(&chunks, 8_000);
        let signal_embedding = engine.embed(&prompt)?;
        let draft_anchor_embedding = engine.embed(prompts::DISPATCH_DRAFT_ANCHOR)?;
        let full_anchor_embedding = engine.embed(prompts::DISPATCH_FULL_ANCHOR)?;
        embedding_gate::classify_embedding(
            &signal_embedding,
            &draft_anchor_embedding,
            &full_anchor_embedding,
        )
    } else {
        None
    };

    let decision = policy::decide(&features, embedding_hint);
    let partials = fanout::analyze_chunks(engine, &chunks)?;
    let report = reduce::reduce(engine, &partials, &decision, &stats)?;
    validate::validate(&report)?;

    Ok(report)
}

fn to_stats(features: &DiffFeatures) -> DiffStats {
    DiffStats {
        files_changed: features.files_changed,
        lines_changed: features.lines_changed,
        hunks: features.hunks,
        binary_files: features.binary_files,
    }
}
