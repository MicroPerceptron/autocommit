use crate::diff::features::DiffFeatures;
use crate::dispatch::embedding_gate::EmbeddingHint;
use crate::types::{DispatchDecision, DispatchRoute};

pub fn decide(features: &DiffFeatures, embedding_hint: Option<EmbeddingHint>) -> DispatchDecision {
    let mut reasons = Vec::new();

    let route = if features.files_changed == 0 {
        reasons.push("empty_diff".to_string());
        DispatchRoute::DraftOnly
    } else if features.lines_changed <= 150 && features.risky_paths == 0 {
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
            DispatchRoute::DraftThenReduce => {
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
}
