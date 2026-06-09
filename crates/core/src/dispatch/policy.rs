use crate::diff::features::DiffFeatures;
use crate::dispatch::embedding_gate::EmbeddingHint;
use crate::types::{DispatchDecision, DispatchRoute};

pub fn decide(features: &DiffFeatures, embedding_hint: Option<EmbeddingHint>) -> DispatchDecision {
    let mut reasons = Vec::new();

    let ws_ratio = if features.lines_changed > 0 {
        features.whitespace_only_lines as f32 / features.lines_changed as f32
    } else {
        0.0
    };

    let route = if features.files_changed == 0 {
        reasons.push("empty_diff".to_string());
        DispatchRoute::DraftOnly
    } else if ws_ratio >= 0.95 {
        reasons.push("format_only".to_string());
        DispatchRoute::FormatOnly
    } else if features.lines_changed <= 50 && features.risky_paths == 0 {
        reasons.push("small_diff".to_string());
        DispatchRoute::DraftOnly
    } else if features.lines_changed > 900 || features.risky_paths > 1 {
        reasons.push("high_complexity".to_string());
        DispatchRoute::FullPipeline
    } else if let Some(hint) = embedding_hint {
        match hint.preferred_route() {
            DispatchRoute::FullPipeline => {
                reasons.push(format!(
                    "embedding_full:{:.3}>{:.3}",
                    hint.full_similarity, hint.draft_similarity
                ));
                DispatchRoute::FullPipeline
            }
            DispatchRoute::DraftOnly => {
                reasons.push(format!(
                    "embedding_draft:{:.3}>={:.3}",
                    hint.draft_similarity, hint.full_similarity
                ));
                DispatchRoute::DraftOnly
            }
            _ => {
                reasons.push(format!("embedding_margin:{:.3}", hint.margin()));
                DispatchRoute::DraftThenReduce
            }
        }
    } else {
        reasons.push("mid_complexity".to_string());
        DispatchRoute::DraftThenReduce
    };

    let estimated_cost_tokens =
        (features.lines_changed as u32).saturating_mul(3) + (features.files_changed as u32 * 64);

    DispatchDecision {
        route,
        reason_codes: reasons,
        estimated_cost_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn borderline_features() -> DiffFeatures {
        DiffFeatures {
            files_changed: 4,
            lines_changed: 220,
            hunks: 6,
            binary_files: 0,
            risky_paths: 0,
            whitespace_only_lines: 0,
        }
    }

    #[test]
    fn borderline_embedding_can_route_to_draft_only() {
        let decision = decide(
            &borderline_features(),
            Some(EmbeddingHint {
                draft_similarity: 0.82,
                full_similarity: 0.41,
            }),
        );
        assert_eq!(decision.route, DispatchRoute::DraftOnly);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|v| v.contains("embedding_draft"))
        );
    }

    #[test]
    fn borderline_embedding_can_route_to_full_pipeline() {
        let decision = decide(
            &borderline_features(),
            Some(EmbeddingHint {
                draft_similarity: 0.32,
                full_similarity: 0.79,
            }),
        );
        assert_eq!(decision.route, DispatchRoute::FullPipeline);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|v| v.contains("embedding_full"))
        );
    }

    #[test]
    fn format_only_route_for_whitespace_dominated_diff() {
        let features = DiffFeatures {
            files_changed: 10,
            lines_changed: 200,
            hunks: 15,
            binary_files: 0,
            risky_paths: 0,
            whitespace_only_lines: 195, // 97.5% whitespace
        };
        let decision = decide(&features, None);
        assert_eq!(decision.route, DispatchRoute::FormatOnly);
        assert!(decision.reason_codes.contains(&"format_only".to_string()));
    }

    #[test]
    fn no_format_only_below_threshold() {
        let features = DiffFeatures {
            files_changed: 5,
            lines_changed: 100,
            hunks: 8,
            binary_files: 0,
            risky_paths: 0,
            whitespace_only_lines: 90, // 90% — below 95% threshold
        };
        let decision = decide(&features, None);
        assert_ne!(decision.route, DispatchRoute::FormatOnly);
    }

    #[test]
    fn draft_only_threshold_lowered_to_50() {
        let features = DiffFeatures {
            files_changed: 3,
            lines_changed: 80,
            hunks: 4,
            binary_files: 0,
            risky_paths: 0,
            whitespace_only_lines: 0,
        };
        // 80 lines > 50: should NOT be DraftOnly
        let decision = decide(&features, None);
        assert_ne!(decision.route, DispatchRoute::DraftOnly);
    }

    #[test]
    fn draft_only_for_tiny_diffs() {
        let features = DiffFeatures {
            files_changed: 1,
            lines_changed: 30,
            hunks: 2,
            binary_files: 0,
            risky_paths: 0,
            whitespace_only_lines: 0,
        };
        let decision = decide(&features, None);
        assert_eq!(decision.route, DispatchRoute::DraftOnly);
    }
}
