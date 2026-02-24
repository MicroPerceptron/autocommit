use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum ProgressStage {
    /// Heuristic scoring and embedding gate routing decision completed.
    Dispatch,
    /// An embedding forward pass completed.
    Embedding,
    /// Chunks merged by directory scope before fanout.
    Merging { from: usize, to: usize },
    /// Per-chunk fanout analysis progress.
    Analyze { completed: usize, total: usize },
    /// Reduce inference completed.
    Reduce,
    /// DraftOnly path: heuristic synthesis without inference.
    DraftSynthesis,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressEvent {
    pub stage: ProgressStage,
}

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

/// Emit a progress event if a callback is present.
pub fn emit(callback: Option<&ProgressCallback>, stage: ProgressStage) {
    if let Some(cb) = callback {
        cb(ProgressEvent { stage });
    }
}
