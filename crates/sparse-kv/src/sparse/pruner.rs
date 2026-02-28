use std::collections::HashMap;

use crate::arena::slot::SlotId;

/// Accumulates attention-weight-based importance scores per slot.
/// Uses exponential moving average for score stability.
pub struct AttentionPruner {
    scores: HashMap<SlotId, f32>,
    decay: f32,
}

impl AttentionPruner {
    pub fn new(decay: f32) -> Self {
        Self {
            scores: HashMap::new(),
            decay,
        }
    }

    /// Update scores from attention weights for a set of slots.
    /// `weights` maps slot -> attention weight sum received in the last step.
    pub fn observe(&mut self, weights: &[(SlotId, f32)]) {
        for &(slot, weight) in weights {
            let entry = self.scores.entry(slot).or_insert(0.0);
            *entry = self.decay * *entry + (1.0 - self.decay) * weight;
        }
    }

    /// Return slots that are candidates for eviction:
    /// score below `threshold` AND not in the recency window.
    pub fn candidates(
        &self,
        all_slots: &[SlotId],
        recency_window: usize,
        threshold: f32,
    ) -> Vec<SlotId> {
        if all_slots.len() <= recency_window {
            return Vec::new();
        }

        let evictable = &all_slots[..all_slots.len() - recency_window];
        evictable
            .iter()
            .filter(|slot| {
                self.scores
                    .get(slot)
                    .map(|&s| s < threshold)
                    .unwrap_or(true) // never-observed slots are eviction candidates
            })
            .copied()
            .collect()
    }

    /// Remove tracking for a slot (after eviction or agent removal).
    pub fn remove(&mut self, slot: &SlotId) {
        self.scores.remove(slot);
    }

    pub fn score(&self, slot: &SlotId) -> Option<f32> {
        self.scores.get(slot).copied()
    }
}
