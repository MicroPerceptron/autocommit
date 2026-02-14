use crate::CoreError;
use crate::llm::traits::LlmEngine;
use crate::types::{AnalysisReport, DiffStats, DispatchDecision, PartialReport};

pub fn reduce(
    engine: &dyn LlmEngine,
    partials: &[PartialReport],
    decision: &DispatchDecision,
    stats: &DiffStats,
) -> Result<AnalysisReport, CoreError> {
    engine.reduce_report(partials, decision, stats)
}
