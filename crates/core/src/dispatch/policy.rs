use crate::diff::features::DiffFeatures;
use crate::types::{DispatchDecision, DispatchRoute};

pub fn decide(features: &DiffFeatures, embedding_hint: Option<f32>) -> DispatchDecision {
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
    } else if let Some(score) = embedding_hint {
        reasons.push(format!("embedding_score:{score:.2}"));
        DispatchRoute::DraftThenReduce
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
