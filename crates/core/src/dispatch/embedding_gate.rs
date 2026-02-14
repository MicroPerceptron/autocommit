use crate::dispatch::heuristics::HeuristicScore;

pub fn should_run_embedding(score: HeuristicScore) -> bool {
    score.borderline
}

pub fn classify_embedding(vector: &[f32], threshold: f32) -> Option<f32> {
    if vector.is_empty() {
        return None;
    }

    let sum: f32 = vector.iter().copied().sum();
    let normalized = (sum / vector.len() as f32).clamp(0.0, 1.0);

    if normalized >= threshold {
        Some(normalized)
    } else {
        None
    }
}
