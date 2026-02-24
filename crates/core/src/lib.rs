pub mod cache;
pub mod diff;
pub mod dispatch;
pub mod llm;
pub mod pipeline;
pub mod progress;
pub mod types;

pub use pipeline::analyze::{AnalyzeOptions, run};
pub use progress::{ProgressCallback, ProgressEvent, ProgressStage};
pub use types::dispatch::{DispatchDecision, DispatchRoute};
pub use types::errors::CoreError;
pub use types::report::{AnalysisReport, DiffStats, PartialReport};
