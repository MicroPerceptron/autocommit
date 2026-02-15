use crate::CoreError;
use crate::types::{AnalysisReport, DiffChunk, DiffStats, DispatchDecision, PartialReport};

pub type EmbeddingVector = Vec<f32>;

pub trait LlmEngine: Send + Sync {
    fn analyze_chunk(&self, chunk: &DiffChunk) -> Result<PartialReport, CoreError>;

    fn analyze_chunks_batched(
        &self,
        _chunks: &[DiffChunk],
    ) -> Option<Result<Vec<PartialReport>, CoreError>> {
        None
    }

    fn reduce_report(
        &self,
        partials: &[PartialReport],
        decision: &DispatchDecision,
        stats: &DiffStats,
    ) -> Result<AnalysisReport, CoreError>;

    fn embed(&self, text: &str) -> Result<EmbeddingVector, CoreError>;
}
