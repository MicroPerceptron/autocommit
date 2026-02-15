use autocommit_core::AnalysisReport;

pub fn render_report(report: &AnalysisReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("commit: {}\n", report.commit_message));
    out.push_str(&format!("summary: {}\n", report.summary));
    out.push_str(&format!("dispatch: {:?}\n", report.dispatch.route));
    out.push_str(&format!("risk: {}\n", report.risk.level));
    if !report.risk.notes.is_empty() {
        out.push_str("risk-notes:\n");
        for note in &report.risk.notes {
            out.push_str(&format!("- {note}\n"));
        }
    }
    out.push_str("items:\n");

    for item in &report.items {
        let item_ref = item
            .files
            .first()
            .map(|file| file.path.as_str())
            .unwrap_or(item.id.as_str());
        out.push_str(&format!(
            "- [{}] {} ({:?})\n",
            item_ref, item.title, item.bucket
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use autocommit_core::types::{
        AnalysisReport, ChangeBucket, ChangeItem, DiffStats, DispatchDecision, DispatchRoute,
        FileRef, FileStatus, RiskReport, TypeTag,
    };

    use super::render_report;

    #[test]
    fn render_report_uses_file_path_for_item_reference() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat(core): build reduce prompt".to_string(),
            summary: "Build reduce prompt across 1 file.".to_string(),
            items: vec![ChangeItem {
                id: "internal-id".to_string(),
                bucket: ChangeBucket::Feature,
                type_tag: TypeTag::Feat,
                title: "Build Reduce Prompt".to_string(),
                intent: "Generate reduce metadata prompt".to_string(),
                files: vec![FileRef {
                    path: "crates/core/src/llm/prompts.rs".to_string(),
                    status: FileStatus::Modified,
                    ranges: Vec::new(),
                }],
                confidence: 0.8,
            }],
            risk: RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: DiffStats::default(),
            dispatch: DispatchDecision {
                route: DispatchRoute::DraftThenReduce,
                reason_codes: vec!["test".to_string()],
                estimated_cost_tokens: 0,
            },
        };

        let rendered = render_report(&report);
        assert!(rendered.contains("[crates/core/src/llm/prompts.rs]"));
        assert!(!rendered.contains("[internal-id]"));
    }
}
