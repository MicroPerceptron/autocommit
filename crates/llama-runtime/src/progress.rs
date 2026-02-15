use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum ProgressStage {
    Embedding,
    Analyze { completed: usize, total: usize },
    Reduce,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressEvent {
    pub stage: ProgressStage,
}

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;
