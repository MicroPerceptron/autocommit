pub mod cache;
pub mod diff;
pub mod dispatch;
pub mod llm;
pub mod pipeline;
pub mod types;

pub use pipeline::analyze::{AnalyzeOptions, run};
pub use types::dispatch::{DispatchDecision, DispatchRoute};
pub use types::errors::CoreError;
pub use types::report::{AnalysisReport, DiffStats, PartialReport};
