pub mod diff;
pub mod dispatch;
pub mod errors;
pub mod report;

pub use diff::{DiffChunk, FileRef, FileStatus, LineRange};
pub use dispatch::{DispatchDecision, DispatchRoute};
pub use errors::CoreError;
pub use report::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffStats, PartialReport, RiskReport, TypeTag,
};
