use crate::arena::position_map::PositionMap;
use crate::arena::slot::{AgentId, LogicalPos, SlotId, SlotState};
use crate::error::InferenceError;
use crate::quant::QuantType;

#[derive(Debug, Clone)]
pub struct ArenaConfig {
    pub capacity: usize,
    pub n_layer: usize,
    pub head_dim: usize,
    pub n_kv_head: usize,
    pub dtype: QuantType,
}

struct LayerBuf {
    data: Vec<u8>,
    slot_stride: usize,
}

impl LayerBuf {
    fn new(capacity: usize, slot_stride: usize) -> Self {
        Self {
            data: vec![0u8; capacity * slot_stride],
            slot_stride,
        }
    }

    fn slot_range(&self, slot: usize) -> std::ops::Range<usize> {
        let start = slot * self.slot_stride;
        start..start + self.slot_stride
    }

    fn key_offset(&self, slot: usize) -> usize {
        slot * self.slot_stride
    }

    fn val_offset(&self, slot: usize) -> usize {
        slot * self.slot_stride + self.slot_stride / 2
    }

    fn half_stride(&self) -> usize {
        self.slot_stride / 2
    }
}

/// Pre-allocated KV cache arena. One contiguous buffer per layer.
///
/// Memory layout per slot: [key_data | val_data]
/// where key_data and val_data are each `n_kv_head * head_dim` elements
/// in the configured dtype.
pub struct KvArena {
    layer_bufs: Vec<LayerBuf>,
    slots: Vec<SlotState>,
    capacity: usize,
    n_layer: usize,
    head_dim: usize,
    n_kv_head: usize,
    dtype: QuantType,
    free_count: usize,
}

impl KvArena {
    pub fn new(config: &ArenaConfig) -> Self {
        assert!(config.n_layer > 0, "KvArena requires at least 1 layer");
        assert!(config.capacity > 0, "KvArena requires capacity > 0");
        let kv_elements = config.n_kv_head * config.head_dim;
        // Each slot stores both K and V, hence *2
        let slot_stride = config.dtype.row_size(kv_elements) * 2;

        let layer_bufs = (0..config.n_layer)
            .map(|_| LayerBuf::new(config.capacity, slot_stride))
            .collect();

        let slots = vec![SlotState::Free; config.capacity];

        Self {
            layer_bufs,
            slots,
            capacity: config.capacity,
            n_layer: config.n_layer,
            head_dim: config.head_dim,
            n_kv_head: config.n_kv_head,
            dtype: config.dtype,
            free_count: config.capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn free_slots(&self) -> usize {
        self.free_count
    }

    pub fn occupied_count(&self) -> usize {
        self.capacity - self.free_count
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    pub fn dtype(&self) -> QuantType {
        self.dtype
    }

    /// Allocate the next free slot for an agent at a given logical position.
    pub fn alloc_slot(
        &mut self,
        agent: AgentId,
        logical_pos: LogicalPos,
    ) -> Result<SlotId, InferenceError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.is_free())
            .ok_or_else(|| {
                InferenceError::Arena(format!(
                    "no free slots (capacity={}, all occupied)",
                    self.capacity
                ))
            })?;

        self.slots[idx] = SlotState::Occupied {
            agent,
            logical_pos,
        };
        self.free_count -= 1;
        Ok(SlotId(idx as u32))
    }

    /// Pin a slot so it cannot be evicted.
    pub fn pin_slot(&mut self, slot: SlotId) {
        let idx = slot.0 as usize;
        if let SlotState::Occupied {
            agent,
            logical_pos,
        } = self.slots[idx]
        {
            self.slots[idx] = SlotState::Pinned {
                agent,
                logical_pos,
            };
        }
    }

    /// Release all slots belonging to an agent.
    pub fn free_agent(&mut self, agent: AgentId) {
        for slot in &mut self.slots {
            if slot.agent() == Some(agent) {
                *slot = SlotState::Free;
                self.free_count += 1;
            }
        }
    }

    /// Mark specific slots for eviction (set them to Free).
    /// Does NOT compact — gaps remain until `compact()` is called.
    pub fn mark_evict(&mut self, slots: &[SlotId]) {
        for &slot in slots {
            let idx = slot.0 as usize;
            if idx < self.capacity && !self.slots[idx].is_pinned() && !self.slots[idx].is_free() {
                self.slots[idx] = SlotState::Free;
                self.free_count += 1;
            }
        }
    }

    /// Get the slot IDs and logical positions for all occupied slots of an agent,
    /// sorted by logical position.
    pub fn agent_slots(&self, agent: AgentId) -> Vec<(SlotId, LogicalPos)> {
        let mut result: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.agent() == Some(agent) {
                    Some((SlotId(i as u32), s.logical_pos().unwrap()))
                } else {
                    None
                }
            })
            .collect();
        result.sort_by_key(|(_, pos)| *pos);
        result
    }

    /// Build a position map for an agent's current slots.
    pub fn agent_position_map(&self, agent: AgentId) -> PositionMap {
        let slots = self.agent_slots(agent);
        PositionMap::new(slots.iter().map(|(_, pos)| *pos).collect())
    }

    /// Compact: move all occupied slots to a contiguous prefix.
    /// Returns a PositionMap reflecting the new dense layout.
    ///
    /// After compaction, occupied slots are at indices 0..n_occupied
    /// and free slots are at n_occupied..capacity.
    pub fn compact(&mut self) -> PositionMap {
        let mut write_idx = 0;
        let mut positions = Vec::new();

        for read_idx in 0..self.capacity {
            if self.slots[read_idx].is_free() {
                continue;
            }

            positions.push(self.slots[read_idx].logical_pos().unwrap());

            if write_idx != read_idx {
                // Move slot data in every layer
                for layer_buf in &mut self.layer_bufs {
                    let src = layer_buf.slot_range(read_idx);
                    let dst_start = layer_buf.slot_range(write_idx).start;
                    // SAFETY: src and dst ranges do not overlap because
                    // write_idx < read_idx (write_idx only advances when
                    // we find an occupied slot, and we skip free ones).
                    layer_buf.data.copy_within(src, dst_start);
                }

                self.slots[write_idx] = self.slots[read_idx];
                self.slots[read_idx] = SlotState::Free;
            }

            write_idx += 1;
        }

        // Recompute free_count from the final write position
        self.free_count = self.capacity - write_idx;
        PositionMap::new(positions)
    }

    /// Get raw pointers to a layer's K and V sub-buffers at a specific slot.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `layer` < `n_layer`
    /// - `slot` < `capacity`
    /// - The returned pointers are not used after `compact()` or `mark_evict()`.
    pub unsafe fn slot_kv_ptrs(&mut self, layer: usize, slot: SlotId) -> (*mut u8, *mut u8) {
        let buf = &mut self.layer_bufs[layer];
        let idx = slot.0 as usize;
        // SAFETY: caller guarantees layer/slot bounds and pointer lifetime.
        unsafe {
            let key_ptr = buf.data.as_mut_ptr().add(buf.key_offset(idx));
            let val_ptr = buf.data.as_mut_ptr().add(buf.val_offset(idx));
            (key_ptr, val_ptr)
        }
    }

    /// Get raw pointer to the entire K buffer for a layer, covering all slots.
    /// The K data for slot `i` starts at offset `i * kv_element_size`.
    ///
    /// # Safety
    /// Same as `slot_kv_ptrs`.
    pub unsafe fn layer_key_ptr(&mut self, layer: usize) -> *mut u8 {
        self.layer_bufs[layer].data.as_mut_ptr()
    }

    /// Byte stride between consecutive slots' key (or value) data.
    pub fn slot_stride(&self) -> usize {
        self.layer_bufs[0].slot_stride
    }

    /// Byte size of one K or V entry per slot.
    pub fn kv_entry_size(&self) -> usize {
        self.layer_bufs[0].half_stride()
    }

    pub fn slot_state(&self, slot: SlotId) -> SlotState {
        self.slots[slot.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(capacity: usize) -> ArenaConfig {
        ArenaConfig {
            capacity,
            n_layer: 2,
            head_dim: 4,
            n_kv_head: 1,
            dtype: QuantType::F16,
        }
    }

    #[test]
    fn alloc_and_free() {
        let mut arena = KvArena::new(&test_config(4));
        let agent = AgentId(0);

        assert_eq!(arena.free_slots(), 4);

        let s0 = arena.alloc_slot(agent, 0).unwrap();
        let s1 = arena.alloc_slot(agent, 1).unwrap();
        assert_eq!(arena.free_slots(), 2);

        arena.free_agent(agent);
        assert_eq!(arena.free_slots(), 4);

        assert!(arena.slot_state(s0).is_free());
        assert!(arena.slot_state(s1).is_free());
    }

    #[test]
    fn alloc_exhaustion() {
        let mut arena = KvArena::new(&test_config(2));
        let agent = AgentId(0);

        arena.alloc_slot(agent, 0).unwrap();
        arena.alloc_slot(agent, 1).unwrap();

        assert!(arena.alloc_slot(agent, 2).is_err());
    }

    #[test]
    fn compact_preserves_logical_positions() {
        let mut arena = KvArena::new(&test_config(8));
        let agent = AgentId(0);

        let slots: Vec<SlotId> = (0u32..8)
            .map(|pos| arena.alloc_slot(agent, pos).unwrap())
            .collect();

        // Evict even positions (0, 2, 4, 6)
        let evict: Vec<SlotId> = slots.iter().copied().step_by(2).collect();
        arena.mark_evict(&evict);

        let pos_map = arena.compact();
        assert_eq!(pos_map.positions, vec![1, 3, 5, 7]);
        assert_eq!(arena.occupied_count(), 4);
        assert_eq!(arena.free_slots(), 4);
    }

    #[test]
    fn pinned_slots_resist_eviction() {
        let mut arena = KvArena::new(&test_config(4));
        let agent = AgentId(0);

        let s0 = arena.alloc_slot(agent, 0).unwrap();
        let _s1 = arena.alloc_slot(agent, 1).unwrap();

        arena.pin_slot(s0);
        arena.mark_evict(&[s0]); // should not evict pinned slot

        assert!(arena.slot_state(s0).is_pinned());
        assert_eq!(arena.occupied_count(), 2);
    }

    #[test]
    fn agent_slots_sorted_by_position() {
        let mut arena = KvArena::new(&test_config(8));
        let agent = AgentId(0);

        // Allocate in reverse order
        for pos in (0u32..4).rev() {
            arena.alloc_slot(agent, pos).unwrap();
        }

        let slots = arena.agent_slots(agent);
        let positions: Vec<LogicalPos> = slots.iter().map(|(_, p)| *p).collect();
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn multi_agent_isolation() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a1, 1).unwrap();

        assert_eq!(arena.agent_slots(a0).len(), 2);
        assert_eq!(arena.agent_slots(a1).len(), 2);

        arena.free_agent(a0);
        assert_eq!(arena.agent_slots(a0).len(), 0);
        assert_eq!(arena.agent_slots(a1).len(), 2);
    }

    #[test]
    fn compact_allows_reallocation() {
        let mut arena = KvArena::new(&test_config(4));
        let agent = AgentId(0);

        // Fill all slots
        for pos in 0..4u32 {
            arena.alloc_slot(agent, pos).unwrap();
        }
        assert_eq!(arena.free_slots(), 0);

        // Evict 2 slots, then compact
        let slots = arena.agent_slots(agent);
        arena.mark_evict(&[slots[0].0, slots[1].0]);
        arena.compact();

        // free_count should be correct, allowing new allocations
        assert_eq!(arena.free_slots(), 2);
        assert_eq!(arena.occupied_count(), 2);

        // Should be able to allocate 2 more slots
        arena.alloc_slot(agent, 10).unwrap();
        arena.alloc_slot(agent, 11).unwrap();
        assert_eq!(arena.free_slots(), 0);

        // And now it should be full again
        assert!(arena.alloc_slot(agent, 12).is_err());
    }
}
