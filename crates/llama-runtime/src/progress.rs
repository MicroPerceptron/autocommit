// Re-export progress types from core. The canonical definitions live in
// autocommit_core::progress; this module keeps existing `use crate::progress::*`
// imports working throughout the crate.

pub use autocommit_core::progress::{ProgressCallback, ProgressEvent, ProgressStage};
