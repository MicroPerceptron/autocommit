use crate::CoreError;
use crate::progress::ProgressCallback;
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

    /// Stable identifier for the loaded model (used as cache key for anchor embeddings).
    /// Returns `None` if the engine does not support fingerprinting.
    fn model_fingerprint(&self) -> Option<String> {
        None
    }

    /// Hardware-optimized cosine similarity. Falls back to pure-Rust when unavailable.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Option<f32> {
        crate::dispatch::embedding_gate::cosine_similarity_fallback(a, b)
    }

    /// Set a progress callback for engine-level events (Embedding, Analyze, Reduce).
    fn set_progress_callback(&self, _callback: Option<ProgressCallback>) {}
}
