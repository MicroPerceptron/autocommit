use crate::arena::position_map::PositionMap;
use crate::arena::slot::{AgentId, LogicalPos, SlotId, SlotState};
use crate::error::InferenceError;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::tensor::buffer::BackendBuffer;

#[derive(Debug, Clone)]
pub struct ArenaConfig {
    pub capacity: usize,
    pub n_layer: usize,
    pub head_dim: usize,
    pub n_kv_head: usize,
    pub dtype: QuantType,
}

/// Backing storage for one layer's K or V data across all slots.
enum KvBacking {
    /// CPU-side Vec<u8> for tests and CPU-only operation.
    Cpu(Vec<u8>),
    /// Backend-allocated buffer (Metal, CUDA, etc.).
    Device(BackendBuffer),
}

impl KvBacking {
    fn base_ptr(&mut self) -> *mut u8 {
        match self {
            KvBacking::Cpu(v) => v.as_mut_ptr(),
            KvBacking::Device(buf) => buf.base_ptr(),
        }
    }

    fn copy_within(&mut self, src_offset: usize, dst_offset: usize, len: usize) {
        match self {
            KvBacking::Cpu(v) => {
                v.copy_within(src_offset..src_offset + len, dst_offset);
            }
            KvBacking::Device(buf) => {
                // For host-accessible device buffers (e.g. Metal shared memory),
                // we can copy directly through the base pointer.
                if buf.is_host() {
                    let ptr = buf.base_ptr();
                    // SAFETY: caller guarantees non-overlapping or forward copy.
                    unsafe {
                        std::ptr::copy(ptr.add(src_offset), ptr.add(dst_offset), len);
                    }
                }
                // For non-host buffers, compaction requires a staging copy via
                // ggml_backend_tensor_get/set. Not yet implemented — we'll hit
                // this path only on discrete GPUs.
            }
        }
    }
}

/// Per-layer storage: separate K and V buffers.
struct LayerKvBuf {
    key: KvBacking,
    val: KvBacking,
    entry_size: usize, // bytes per slot per K or V
}

/// Pre-allocated KV cache arena. Separate K and V buffers per layer.
///
/// Memory layout per layer:
///   key_buf: [slot_0_key | slot_1_key | ... | slot_N_key]
///   val_buf: [slot_0_val | slot_1_val | ... | slot_N_val]
///
/// Each entry is `n_kv_head * head_dim` elements in the configured dtype.
pub struct KvArena {
    layers: Vec<LayerKvBuf>,
    slots: Vec<SlotState>,
    capacity: usize,
    n_layer: usize,
    head_dim: usize,
    n_kv_head: usize,
    dtype: QuantType,
    free_count: usize,
}

impl KvArena {
    /// Create a KV arena with CPU-backed storage (for tests and CPU inference).
    pub fn new(config: &ArenaConfig) -> Self {
        assert!(config.n_layer > 0, "KvArena requires at least 1 layer");
        assert!(config.capacity > 0, "KvArena requires capacity > 0");
        let entry_size = config.dtype.row_size(config.n_kv_head * config.head_dim);

        let layers = (0..config.n_layer)
            .map(|_| LayerKvBuf {
                key: KvBacking::Cpu(vec![0u8; config.capacity * entry_size]),
                val: KvBacking::Cpu(vec![0u8; config.capacity * entry_size]),
                entry_size,
            })
            .collect();

        let slots = vec![SlotState::Free; config.capacity];

        Self {
            layers,
            slots,
            capacity: config.capacity,
            n_layer: config.n_layer,
            head_dim: config.head_dim,
            n_kv_head: config.n_kv_head,
            dtype: config.dtype,
            free_count: config.capacity,
        }
    }

    /// Create a KV arena with backend-allocated storage (for GPU inference).
    pub fn with_backend(
        config: &ArenaConfig,
        backend: &Backend,
    ) -> Result<Self, InferenceError> {
        assert!(config.n_layer > 0, "KvArena requires at least 1 layer");
        assert!(config.capacity > 0, "KvArena requires capacity > 0");
        let entry_size = config.dtype.row_size(config.n_kv_head * config.head_dim);
        let buf_size = config.capacity * entry_size;

        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            let mut key_buf = backend.alloc_buffer(buf_size)?;
            let mut val_buf = backend.alloc_buffer(buf_size)?;
            key_buf.clear(0);
            val_buf.clear(0);
            layers.push(LayerKvBuf {
                key: KvBacking::Device(key_buf),
                val: KvBacking::Device(val_buf),
                entry_size,
            });
        }

        let slots = vec![SlotState::Free; config.capacity];

        Ok(Self {
            layers,
            slots,
            capacity: config.capacity,
            n_layer: config.n_layer,
            head_dim: config.head_dim,
            n_kv_head: config.n_kv_head,
            dtype: config.dtype,
            free_count: config.capacity,
        })
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

    /// Byte size of one K or V entry per slot.
    pub fn entry_size(&self) -> usize {
        self.layers[0].entry_size
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
                let entry_size = self.layers[0].entry_size;
                // Move slot data in every layer — K and V separately
                for layer in &mut self.layers {
                    let src_off = read_idx * entry_size;
                    let dst_off = write_idx * entry_size;
                    layer.key.copy_within(src_off, dst_off, entry_size);
                    layer.val.copy_within(src_off, dst_off, entry_size);
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

    /// Get raw pointer to a layer's K buffer for a specific slot.
    ///
    /// # Safety
    /// Caller must ensure `layer` < `n_layer` and `slot` < `capacity`.
    /// Pointer is invalidated by `compact()` or `mark_evict()`.
    pub unsafe fn slot_key_ptr(&mut self, layer: usize, slot: SlotId) -> *mut u8 {
        let idx = slot.0 as usize;
        let entry_size = self.layers[layer].entry_size;
        // SAFETY: caller guarantees bounds.
        unsafe { self.layers[layer].key.base_ptr().add(idx * entry_size) }
    }

    /// Get raw pointer to a layer's V buffer for a specific slot.
    ///
    /// # Safety
    /// Same as `slot_key_ptr`.
    pub unsafe fn slot_val_ptr(&mut self, layer: usize, slot: SlotId) -> *mut u8 {
        let idx = slot.0 as usize;
        let entry_size = self.layers[layer].entry_size;
        // SAFETY: caller guarantees bounds.
        unsafe { self.layers[layer].val.base_ptr().add(idx * entry_size) }
    }

    /// Get raw pointer to the entire K buffer for a layer (all slots contiguous).
    ///
    /// # Safety
    /// Same as `slot_key_ptr`.
    pub unsafe fn layer_key_ptr(&mut self, layer: usize) -> *mut u8 {
        self.layers[layer].key.base_ptr()
    }

    /// Get raw pointer to the entire V buffer for a layer (all slots contiguous).
    ///
    /// # Safety
    /// Same as `slot_key_ptr`.
    pub unsafe fn layer_val_ptr(&mut self, layer: usize) -> *mut u8 {
        self.layers[layer].val.base_ptr()
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

    #[test]
    fn entry_size_consistent() {
        let config = test_config(4);
        let arena = KvArena::new(&config);
        // F16, head_dim=4, n_kv_head=1 → 4 elements × 2 bytes = 8 bytes per entry
        assert_eq!(arena.entry_size(), 8);
    }
}
