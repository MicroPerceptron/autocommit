use serde::{Deserialize, Serialize};

use crate::types::diff::FileRef;
use crate::types::dispatch::DispatchDecision;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChangeBucket {
    Feature,
    Patch,
    Addition,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TypeTag {
    Feat,
    Fix,
    Refactor,
    Docs,
    Test,
    Chore,
    Perf,
    Style,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChangeItem {
    pub id: String,
    pub bucket: ChangeBucket,
    pub type_tag: TypeTag,
    pub title: String,
    pub intent: String,
    pub files: Vec<FileRef>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RiskReport {
    pub level: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DiffStats {
    pub files_changed: usize,
    pub lines_changed: usize,
    pub hunks: usize,
    pub binary_files: usize,
    pub whitespace_only_lines: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PartialReport {
    pub summary: String,
    pub items: Vec<ChangeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalysisReport {
    pub schema_version: String,
    pub commit_message: String,
    pub summary: String,
    pub items: Vec<ChangeItem>,
    pub risk: RiskReport,
    pub stats: DiffStats,
    pub dispatch: DispatchDecision,
}

impl AnalysisReport {
    pub fn empty(dispatch: DispatchDecision) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            commit_message: "chore: no-op diff".to_string(),
            summary: "No diff chunks were provided.".to_string(),
            items: Vec::new(),
            risk: RiskReport {
                level: "low".to_string(),
                notes: vec!["No changes detected".to_string()],
            },
            stats: DiffStats::default(),
            dispatch,
        }
    }
}
