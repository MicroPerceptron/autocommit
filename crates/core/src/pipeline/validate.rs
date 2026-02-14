use std::collections::HashSet;

use crate::CoreError;
use crate::types::AnalysisReport;

pub fn validate(report: &AnalysisReport) -> Result<(), CoreError> {
    if report.schema_version != "1.0" {
        return Err(CoreError::Validation(
            "unsupported schema version".to_string(),
        ));
    }

    if report.commit_message.trim().is_empty() {
        return Err(CoreError::Validation("empty commit message".to_string()));
    }

    let mut ids = HashSet::new();
    for item in &report.items {
        if item.id.trim().is_empty() {
            return Err(CoreError::Validation("item id cannot be empty".to_string()));
        }
        if !ids.insert(item.id.clone()) {
            return Err(CoreError::Validation(format!(
                "duplicate change item id: {}",
                item.id
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::types::{
        AnalysisReport, ChangeBucket, ChangeItem, DiffStats, DispatchDecision, DispatchRoute,
        FileRef, FileStatus, RiskReport, TypeTag,
    };

    use super::validate;

    #[test]
    fn rejects_duplicate_item_ids() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat: test".to_string(),
            summary: "x".to_string(),
            items: vec![item("id-1"), item("id-1")],
            risk: RiskReport {
                level: "low".to_string(),
                notes: vec![],
            },
            stats: DiffStats::default(),
            dispatch: DispatchDecision {
                route: DispatchRoute::DraftOnly,
                reason_codes: vec!["test".to_string()],
                estimated_cost_tokens: 0,
            },
        };

        assert!(validate(&report).is_err());
    }

    fn item(id: &str) -> ChangeItem {
        ChangeItem {
            id: id.to_string(),
            bucket: ChangeBucket::Patch,
            type_tag: TypeTag::Fix,
            title: "title".to_string(),
            intent: "intent".to_string(),
            files: vec![FileRef {
                path: "src/main.rs".to_string(),
                status: FileStatus::Modified,
                ranges: vec![],
            }],
            confidence: 0.8,
        }
    }
}
