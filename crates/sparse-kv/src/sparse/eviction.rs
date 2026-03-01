use crate::arena::slot::{LogicalPos, SlotId};

/// Policy for selecting which KV cache slots to evict when the arena is full.
pub trait EvictionPolicy {
    /// Select slots to evict from `candidates` to free at least `need` slots.
    /// `candidates` contains (SlotId, LogicalPos) for all non-pinned occupied slots.
    fn select(&self, candidates: &[(SlotId, LogicalPos)], need: usize) -> Vec<SlotId>;
}

/// Evict the oldest (lowest logical_pos) slots first.
///
/// Zero-cost policy — requires no attention weight tracking. Works well for
/// streaming inference where recent tokens are more important.
pub struct RecencyEviction;

impl EvictionPolicy for RecencyEviction {
    fn select(&self, candidates: &[(SlotId, LogicalPos)], need: usize) -> Vec<SlotId> {
        if candidates.is_empty() || need == 0 {
            return Vec::new();
        }

        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by_key(|(_, pos)| *pos);
        sorted
            .iter()
            .take(need)
            .map(|(slot, _)| *slot)
            .collect()
    }
}

/// Configuration for automatic KV cache eviction.
pub struct EvictionConfig {
    /// The eviction policy to use.
    pub policy: Box<dyn EvictionPolicy>,
    /// Evict when `free_slots / capacity` drops below this threshold (e.g. 0.1).
    pub trigger_threshold: f32,
    /// Fraction of occupied slots to evict when triggered (e.g. 0.2).
    pub evict_fraction: f32,
}

impl EvictionConfig {
    /// Default config: recency-based, trigger at 10% free, evict 20% of occupied.
    pub fn default_recency() -> Self {
        Self {
            policy: Box::new(RecencyEviction),
            trigger_threshold: 0.1,
            evict_fraction: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recency_selects_oldest() {
        let policy = RecencyEviction;
        let candidates = vec![
            (SlotId(3), 10),
            (SlotId(0), 0),
            (SlotId(1), 5),
            (SlotId(2), 7),
        ];

        let evicted = policy.select(&candidates, 2);
        assert_eq!(evicted, vec![SlotId(0), SlotId(1)]); // pos 0 and 5
    }

    #[test]
    fn recency_need_zero() {
        let policy = RecencyEviction;
        let candidates = vec![(SlotId(0), 0), (SlotId(1), 1)];
        assert!(policy.select(&candidates, 0).is_empty());
    }

    #[test]
    fn recency_empty_candidates() {
        let policy = RecencyEviction;
        assert!(policy.select(&[], 5).is_empty());
    }

    #[test]
    fn recency_need_exceeds_candidates() {
        let policy = RecencyEviction;
        let candidates = vec![(SlotId(0), 0), (SlotId(1), 1)];
        let evicted = policy.select(&candidates, 10);
        assert_eq!(evicted.len(), 2); // can only evict what's available
    }

    #[test]
    fn eviction_config_default() {
        let config = EvictionConfig::default_recency();
        assert!((config.trigger_threshold - 0.1).abs() < f32::EPSILON);
        assert!((config.evict_fraction - 0.2).abs() < f32::EPSILON);

        // Verify the policy works
        let candidates = vec![(SlotId(1), 5), (SlotId(0), 0)];
        let evicted = config.policy.select(&candidates, 1);
        assert_eq!(evicted, vec![SlotId(0)]);
    }
}
