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
        out.push_str(&format!(
            "- [{}] {} ({:?})\n",
            item.id, item.title, item.bucket
        ));
    }

    out
}
