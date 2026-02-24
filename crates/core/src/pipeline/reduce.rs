use crate::CoreError;
use crate::llm::traits::LlmEngine;
use crate::types::{
    AnalysisReport, ChangeItem, DiffStats, DispatchDecision, PartialReport, RiskReport, TypeTag,
};

pub fn reduce(
    engine: &dyn LlmEngine,
    partials: &[PartialReport],
    decision: &DispatchDecision,
    stats: &DiffStats,
) -> Result<AnalysisReport, CoreError> {
    engine.reduce_report(partials, decision, stats)
}

/// Build a report from partials without running the reduce inference call.
/// Used for the DraftOnly fast path where simple diffs don't warrant an extra model pass.
pub fn synthesize_draft_report(
    partials: &[PartialReport],
    decision: &DispatchDecision,
    stats: &DiffStats,
) -> AnalysisReport {
    let mut items: Vec<ChangeItem> = Vec::new();
    for partial in partials {
        items.extend(partial.items.clone());
    }

    let commit_message = synthesize_commit_message(&items);
    let summary = synthesize_summary(&items, stats);

    AnalysisReport {
        schema_version: "1.0".to_string(),
        commit_message,
        summary,
        items,
        risk: RiskReport {
            level: "low".to_string(),
            notes: vec![
                "commit_source:draft_synthesis".to_string(),
                format!("dispatch:{:?}", decision.route),
            ],
        },
        stats: stats.clone(),
        dispatch: decision.clone(),
    }
}

fn synthesize_commit_message(items: &[ChangeItem]) -> String {
    let best = items
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let Some(best) = best else {
        return "chore: update project code".to_string();
    };

    let type_prefix = type_tag_prefix(best.type_tag.clone());
    let scope = best
        .files
        .first()
        .and_then(|f| scope_from_path(&f.path));
    let description = decapitalize_first(best.title.trim());
    let description = trim_type_prefix(type_prefix, &description);

    match scope {
        Some(scope) => format!("{type_prefix}({scope}): {description}"),
        None => format!("{type_prefix}: {description}"),
    }
}

fn synthesize_summary(items: &[ChangeItem], stats: &DiffStats) -> String {
    let file_count = stats.files_changed.max(1);
    let noun = if file_count == 1 { "file" } else { "files" };

    let best = items
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal));

    match best {
        Some(item) => format!(
            "{} across {file_count} {noun}.",
            capitalize_first(item.title.trim())
        ),
        None => format!("Update project code across {file_count} {noun}."),
    }
}

fn type_tag_prefix(tag: TypeTag) -> &'static str {
    match tag {
        TypeTag::Feat => "feat",
        TypeTag::Fix => "fix",
        TypeTag::Refactor => "refactor",
        TypeTag::Docs => "docs",
        TypeTag::Test => "test",
        TypeTag::Chore => "chore",
        TypeTag::Perf => "chore",
        TypeTag::Style => "chore",
        TypeTag::Mixed => "chore",
    }
}

fn scope_from_path(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 2 {
        return None;
    }
    let scope = match parts[0] {
        "crates" | "src" | "packages" | "libs" => parts.get(1).copied(),
        "tests" | "test" => Some("test"),
        "docs" => Some("docs"),
        _ => parts.first().copied(),
    }?;
    let sanitized = scope
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect::<String>()
        .to_ascii_lowercase();
    if sanitized.is_empty() {
        None
    } else {
        Some(sanitized)
    }
}

fn trim_type_prefix<'a>(type_prefix: &str, description: &'a str) -> String {
    let lower = description.to_ascii_lowercase();
    let prefix_with_space = format!("{type_prefix} ");
    if lower.starts_with(&prefix_with_space) {
        let trimmed = description[prefix_with_space.len()..].trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    description.to_string()
}

fn decapitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChangeBucket, DispatchRoute, FileRef, FileStatus};

    fn sample_partial(title: &str, type_tag: TypeTag, path: &str) -> PartialReport {
        PartialReport {
            summary: format!("Analyzed {path}"),
            items: vec![ChangeItem {
                id: "item-1".to_string(),
                bucket: ChangeBucket::Patch,
                type_tag,
                title: title.to_string(),
                intent: "test intent".to_string(),
                files: vec![FileRef {
                    path: path.to_string(),
                    status: FileStatus::Modified,
                    ranges: vec![],
                }],
                confidence: 0.8,
            }],
        }
    }

    #[test]
    fn draft_report_produces_conventional_commit() {
        let partials = vec![sample_partial(
            "Add validation logic",
            TypeTag::Feat,
            "crates/core/src/validate.rs",
        )];
        let decision = DispatchDecision {
            route: DispatchRoute::DraftOnly,
            reason_codes: vec!["small_diff".to_string()],
            estimated_cost_tokens: 100,
        };
        let stats = DiffStats {
            files_changed: 1,
            lines_changed: 20,
            hunks: 1,
            binary_files: 0,
        };

        let report = synthesize_draft_report(&partials, &decision, &stats);
        assert!(report.commit_message.starts_with("feat(core):"));
        assert_eq!(report.risk.level, "low");
    }
}
