use crate::dispatch::heuristics::HeuristicScore;
use crate::types::DispatchRoute;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingHint {
    pub draft_similarity: f32,
    pub full_similarity: f32,
}

impl EmbeddingHint {
    pub fn preferred_route(self) -> DispatchRoute {
        if self.full_similarity > self.draft_similarity {
            DispatchRoute::FullPipeline
        } else {
            DispatchRoute::DraftOnly
        }
    }

    pub fn margin(self) -> f32 {
        (self.full_similarity - self.draft_similarity).abs()
    }
}

pub fn should_run_embedding(score: HeuristicScore) -> bool {
    score.borderline
}

pub fn classify_embedding(
    signal_embedding: &[f32],
    draft_anchor_embedding: &[f32],
    full_anchor_embedding: &[f32],
) -> Option<EmbeddingHint> {
    let draft_similarity = cosine_similarity(signal_embedding, draft_anchor_embedding)?;
    let full_similarity = cosine_similarity(signal_embedding, full_anchor_embedding)?;
    Some(EmbeddingHint {
        draft_similarity,
        full_similarity,
    })
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let len = a.len().min(b.len());
    if len == 0 {
        return None;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for idx in 0..len {
        let av = a[idx];
        let bv = b[idx];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        None
    } else {
        Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_embedding_prefers_closest_anchor() {
        let signal = [0.9f32, 0.1, 0.0];
        let draft = [1.0f32, 0.0, 0.0];
        let full = [0.0f32, 1.0, 0.0];

        let hint = classify_embedding(&signal, &draft, &full).expect("hint");
        assert_eq!(hint.preferred_route(), DispatchRoute::DraftOnly);
        assert!(hint.draft_similarity > hint.full_similarity);
    }
}
