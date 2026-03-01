use crate::arena::position_map::PositionMap;
use crate::arena::slot::{AgentId, LogicalPos, SharedRegionId, SlotId, SlotState};
use crate::error::InferenceError;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::tensor::buffer::BackendBuffer;

/// Metadata for a shared KV region (immutable prefix shared across agents).
struct SharedRegion {
    id: SharedRegionId,
    /// Arena slot indices belonging to this region, in logical order.
    slot_indices: Vec<u32>,
    /// Number of agents currently referencing this region.
    ref_count: usize,
}

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
    shared_regions: Vec<SharedRegion>,
    next_shared_id: u32,
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
            shared_regions: Vec::new(),
            next_shared_id: 0,
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
            shared_regions: Vec::new(),
            next_shared_id: 0,
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
            if idx < self.capacity
                && !self.slots[idx].is_pinned()
                && !self.slots[idx].is_free()
                && !self.slots[idx].is_shared()
            {
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

    /// Get all evictable slots: occupied (not pinned, not free) with logical positions.
    pub fn evictable_slots(&self) -> Vec<(SlotId, LogicalPos)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if let SlotState::Occupied { logical_pos, .. } = s {
                    Some((SlotId(i as u32), *logical_pos))
                } else {
                    None // Free and Pinned are not evictable
                }
            })
            .collect()
    }

    /// Build a position map for an agent's current slots.
    pub fn agent_position_map(&self, agent: AgentId) -> PositionMap {
        let slots = self.agent_slots(agent);
        PositionMap::new(slots.iter().map(|(_, pos)| *pos).collect())
    }

    /// Compact the arena so that `target` agent's slots occupy positions 0..n_target,
    /// followed by other agents' occupied slots. Free slots fill the rest.
    ///
    /// Returns `(0, n_target)` — the target agent's slots are at indices 0..n_target.
    /// This ensures the target agent's KV data is contiguous for attention.
    pub fn compact_for_agent(&mut self, target: AgentId) -> (usize, usize) {
        let ranges = self.compact_for_batch(&[target]);
        if ranges.is_empty() {
            return (0, 0);
        }
        (ranges[0].1, ranges[0].2)
    }

    /// Compact arena so multiple agents' slots are contiguous and grouped.
    ///
    /// After compaction, the arena layout is:
    ///   [A0_slots | A1_slots | ... | AN_slots | other_occupied | FREE...]
    ///
    /// Each agent's slots are sorted by logical_pos within their block.
    /// The ordering of agents follows the input `agents` slice order
    /// (scheduler priority order). Any occupied slots not belonging to
    /// a listed agent are placed after the listed agents' blocks.
    ///
    /// Returns `Vec<(AgentId, offset, count)>` for each agent in the input.
    pub fn compact_for_batch(&mut self, agents: &[AgentId]) -> Vec<(AgentId, usize, usize)> {
        // Collect slot indices per agent, sorted by logical_pos
        let mut per_agent: Vec<Vec<usize>> = Vec::with_capacity(agents.len());
        let listed_set: Vec<AgentId> = agents.to_vec();

        for &agent in agents {
            let mut indices: Vec<usize> = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.is_free() && s.agent() == Some(agent))
                .map(|(i, _)| i)
                .collect();
            indices.sort_by_key(|&i| self.slots[i].logical_pos().unwrap());
            per_agent.push(indices);
        }

        // Collect any occupied slots NOT in the listed agents
        let mut other_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                !s.is_free() && s.agent().map_or(true, |a| !listed_set.contains(&a))
            })
            .map(|(i, _)| i)
            .collect();
        other_indices.sort_by_key(|&i| self.slots[i].logical_pos().unwrap());

        // Build desired permutation and result ranges
        let mut desired: Vec<usize> = Vec::new();
        let mut result = Vec::with_capacity(agents.len());

        for (agent_idx, indices) in per_agent.iter().enumerate() {
            let offset = desired.len();
            desired.extend_from_slice(indices);
            result.push((agents[agent_idx], offset, indices.len()));
        }
        desired.extend_from_slice(&other_indices);
        let n_total = desired.len();

        // Check if already in the correct order (common fast path)
        let already_ordered = desired.iter().enumerate().all(|(i, &src)| i == src);
        if already_ordered {
            self.free_count = self.capacity - n_total;
            return result;
        }

        // Apply permutation using temporary buffers
        let entry_size = self.layers[0].entry_size;
        let new_states: Vec<SlotState> = desired.iter().map(|&i| self.slots[i]).collect();

        for layer in &mut self.layers {
            let mut temp_k = vec![0u8; n_total * entry_size];
            let mut temp_v = vec![0u8; n_total * entry_size];

            let k_ptr = layer.key.base_ptr();
            let v_ptr = layer.val.base_ptr();

            for (dst_idx, &src_idx) in desired.iter().enumerate() {
                let src_off = src_idx * entry_size;
                let dst_off = dst_idx * entry_size;
                // SAFETY: src_idx < capacity, dst_idx < n_total, both within allocated bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_ptr.add(src_off),
                        temp_k.as_mut_ptr().add(dst_off),
                        entry_size,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_ptr.add(src_off),
                        temp_v.as_mut_ptr().add(dst_off),
                        entry_size,
                    );
                }
            }

            // Write back to positions 0..n_total
            // SAFETY: n_total <= capacity, temp buffers are correctly sized.
            unsafe {
                std::ptr::copy_nonoverlapping(temp_k.as_ptr(), k_ptr, n_total * entry_size);
                std::ptr::copy_nonoverlapping(temp_v.as_ptr(), v_ptr, n_total * entry_size);
            }
        }

        // Update slot states
        for (i, state) in new_states.into_iter().enumerate() {
            self.slots[i] = state;
        }
        for i in n_total..self.capacity {
            self.slots[i] = SlotState::Free;
        }

        self.free_count = self.capacity - n_total;
        result
    }

    /// Compact arena with shared regions placed at offset 0.
    ///
    /// Layout after compaction:
    ///   [SHARED_SLOTS | A0_private | A1_private | ... | other_occupied | FREE...]
    ///
    /// Returns `(shared_len, Vec<(AgentId, private_offset, private_count)>)`.
    /// `shared_len` is the total number of shared region slots at the front.
    pub fn compact_for_batch_with_shared(
        &mut self,
        agents: &[AgentId],
        shared_regions: &[SharedRegionId],
    ) -> (usize, Vec<(AgentId, usize, usize)>) {
        // 1. Collect shared region slot indices (sorted by region order, then logical_pos)
        let mut shared_indices: Vec<usize> = Vec::new();
        for &region_id in shared_regions {
            if let Some(indices) = self.shared_region_slot_indices(region_id) {
                shared_indices.extend(indices.iter().map(|&i| i as usize));
            }
        }

        let shared_set: Vec<usize> = shared_indices.clone();
        let listed_set: Vec<AgentId> = agents.to_vec();

        // 2. Collect private slot indices per agent (excluding shared slots)
        let mut per_agent: Vec<Vec<usize>> = Vec::with_capacity(agents.len());
        for &agent in agents {
            let mut indices: Vec<usize> = self
                .slots
                .iter()
                .enumerate()
                .filter(|(i, s)| {
                    !s.is_free()
                        && s.agent() == Some(agent)
                        && !shared_set.contains(i)
                })
                .map(|(i, _)| i)
                .collect();
            indices.sort_by_key(|&i| self.slots[i].logical_pos().unwrap());
            per_agent.push(indices);
        }

        // 3. Collect other occupied slots (not shared, not listed agents)
        let mut other_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                !s.is_free()
                    && !shared_set.contains(i)
                    && s.agent().map_or(true, |a| !listed_set.contains(&a))
            })
            .map(|(i, _)| i)
            .collect();
        other_indices.sort_by_key(|&i| self.slots[i].logical_pos().unwrap());

        // 4. Build desired permutation and result
        let shared_len = shared_indices.len();
        let mut desired: Vec<usize> = Vec::new();
        desired.extend_from_slice(&shared_indices);

        let mut result = Vec::with_capacity(agents.len());
        for (agent_idx, indices) in per_agent.iter().enumerate() {
            let offset = desired.len();
            desired.extend_from_slice(indices);
            result.push((agents[agent_idx], offset, indices.len()));
        }
        desired.extend_from_slice(&other_indices);
        let n_total = desired.len();

        // 5. Fast path: already ordered
        let already_ordered = desired.iter().enumerate().all(|(i, &src)| i == src);
        if already_ordered {
            self.free_count = self.capacity - n_total;
            return (shared_len, result);
        }

        // 6. Apply permutation using temporary buffers
        let entry_size = self.layers[0].entry_size;
        let new_states: Vec<SlotState> = desired.iter().map(|&i| self.slots[i]).collect();

        for layer in &mut self.layers {
            let mut temp_k = vec![0u8; n_total * entry_size];
            let mut temp_v = vec![0u8; n_total * entry_size];

            let k_ptr = layer.key.base_ptr();
            let v_ptr = layer.val.base_ptr();

            for (dst_idx, &src_idx) in desired.iter().enumerate() {
                let src_off = src_idx * entry_size;
                let dst_off = dst_idx * entry_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_ptr.add(src_off),
                        temp_k.as_mut_ptr().add(dst_off),
                        entry_size,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_ptr.add(src_off),
                        temp_v.as_mut_ptr().add(dst_off),
                        entry_size,
                    );
                }
            }

            unsafe {
                std::ptr::copy_nonoverlapping(temp_k.as_ptr(), k_ptr, n_total * entry_size);
                std::ptr::copy_nonoverlapping(temp_v.as_ptr(), v_ptr, n_total * entry_size);
            }
        }

        // 7. Update slot states
        for (i, state) in new_states.into_iter().enumerate() {
            self.slots[i] = state;
        }
        for i in n_total..self.capacity {
            self.slots[i] = SlotState::Free;
        }

        // 8. Update shared region slot indices to reflect new positions
        for &region_id in shared_regions {
            if let Some(region) = self.shared_regions.iter_mut().find(|r| r.id == region_id) {
                // Find new positions of this region's slots
                region.slot_indices = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.region() == Some(region_id))
                    .map(|(i, _)| i as u32)
                    .collect();
            }
        }

        self.free_count = self.capacity - n_total;
        (shared_len, result)
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

    // ── Shared region management ─────────────────────────────────

    /// Create a new shared region by allocating `n_slots` slots.
    ///
    /// Returns the region ID and slot IDs for the caller to prefill K/V data.
    /// The region starts with `ref_count = 0`; attach it to agents with
    /// `ref_shared_region()`.
    pub fn create_shared_region(
        &mut self,
        n_slots: usize,
    ) -> Result<(SharedRegionId, Vec<SlotId>), InferenceError> {
        if n_slots == 0 {
            return Err(InferenceError::Arena("shared region must have at least 1 slot".into()));
        }
        if self.free_count < n_slots {
            return Err(InferenceError::Arena(format!(
                "not enough free slots for shared region: need {n_slots}, have {}",
                self.free_count
            )));
        }

        let region_id = SharedRegionId(self.next_shared_id);
        self.next_shared_id += 1;

        let mut slot_ids = Vec::with_capacity(n_slots);
        let mut slot_indices = Vec::with_capacity(n_slots);

        for pos in 0..n_slots {
            let idx = self
                .slots
                .iter()
                .position(|s| s.is_free())
                .expect("free_count check guarantees a free slot exists");

            self.slots[idx] = SlotState::Shared {
                region: region_id,
                logical_pos: pos as u32,
            };
            self.free_count -= 1;
            slot_ids.push(SlotId(idx as u32));
            slot_indices.push(idx as u32);
        }

        self.shared_regions.push(SharedRegion {
            id: region_id,
            slot_indices,
            ref_count: 0,
        });

        Ok((region_id, slot_ids))
    }

    /// Increment reference count for a shared region.
    pub fn ref_shared_region(&mut self, id: SharedRegionId) -> Result<(), InferenceError> {
        let region = self
            .shared_regions
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| InferenceError::Arena(format!("shared region {:?} not found", id)))?;
        region.ref_count += 1;
        Ok(())
    }

    /// Decrement reference count. If it reaches 0, free all slots in the region.
    pub fn unref_shared_region(&mut self, id: SharedRegionId) -> Result<(), InferenceError> {
        let region = self
            .shared_regions
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| InferenceError::Arena(format!("shared region {:?} not found", id)))?;

        region.ref_count = region.ref_count.saturating_sub(1);

        if region.ref_count == 0 {
            // Free all slots belonging to this region
            for &idx in &region.slot_indices {
                self.slots[idx as usize] = SlotState::Free;
                self.free_count += 1;
            }
            self.shared_regions.retain(|r| r.id != id);
        }

        Ok(())
    }

    /// Get the sequence length (number of slots) of a shared region.
    pub fn shared_region_len(&self, id: SharedRegionId) -> Option<usize> {
        self.shared_regions
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.slot_indices.len())
    }

    /// Get the slot indices for a shared region (in logical order).
    pub fn shared_region_slot_indices(&self, id: SharedRegionId) -> Option<&[u32]> {
        self.shared_regions
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.slot_indices.as_slice())
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

    #[test]
    fn compact_for_agent_orders_target_first() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Interleaved allocation: A0, A1, A0, A1
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 1).unwrap();

        let (start, count) = arena.compact_for_agent(a0);
        assert_eq!(start, 0);
        assert_eq!(count, 2);

        // Verify a0's slots are at positions 0..2
        let a0_slots = arena.agent_slots(a0);
        assert_eq!(a0_slots.len(), 2);
        assert_eq!(a0_slots[0].0, SlotId(0));
        assert_eq!(a0_slots[1].0, SlotId(1));

        // Verify a1's slots follow at positions 2..4
        let a1_slots = arena.agent_slots(a1);
        assert_eq!(a1_slots.len(), 2);
        assert_eq!(a1_slots[0].0, SlotId(2));
        assert_eq!(a1_slots[1].0, SlotId(3));

        assert_eq!(arena.free_slots(), 4);
    }

    #[test]
    fn compact_for_agent_single_agent_noop() {
        let mut arena = KvArena::new(&test_config(8));
        let agent = AgentId(0);

        // Single agent, contiguous allocation
        for pos in 0u32..4 {
            arena.alloc_slot(agent, pos).unwrap();
        }

        let (start, count) = arena.compact_for_agent(agent);
        assert_eq!(start, 0);
        assert_eq!(count, 4);

        let slots = arena.agent_slots(agent);
        let positions: Vec<LogicalPos> = slots.iter().map(|(_, p)| *p).collect();
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn compact_for_agent_with_gaps() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Allocate interleaved with gaps
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        let evict_slot = arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 1).unwrap();
        arena.alloc_slot(a0, 2).unwrap();

        // Evict one of a0's slots to create a gap
        arena.mark_evict(&[evict_slot]);

        let (start, count) = arena.compact_for_agent(a1);
        assert_eq!(start, 0);
        assert_eq!(count, 2); // a1 has 2 slots

        // a1 at front, a0 (remaining 2 slots) after
        let a1_slots = arena.agent_slots(a1);
        assert_eq!(a1_slots.len(), 2);
        assert_eq!(a1_slots[0].0, SlotId(0));
        assert_eq!(a1_slots[1].0, SlotId(1));
    }

    #[test]
    fn compact_for_batch_two_agents() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Interleaved: A0, A1, A0, A1
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 1).unwrap();

        let ranges = arena.compact_for_batch(&[a1, a0]); // a1 first (scheduler priority)
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], (a1, 0, 2)); // a1 at offset 0, count 2
        assert_eq!(ranges[1], (a0, 2, 2)); // a0 at offset 2, count 2

        // Verify physical positions
        let a1_slots = arena.agent_slots(a1);
        assert_eq!(a1_slots[0].0, SlotId(0));
        assert_eq!(a1_slots[1].0, SlotId(1));

        let a0_slots = arena.agent_slots(a0);
        assert_eq!(a0_slots[0].0, SlotId(2));
        assert_eq!(a0_slots[1].0, SlotId(3));
    }

    #[test]
    fn compact_for_batch_three_agents_priority_order() {
        let mut arena = KvArena::new(&test_config(12));
        let a0 = AgentId(0);
        let a1 = AgentId(1);
        let a2 = AgentId(2);

        // Allocate interleaved
        for pos in 0u32..3 {
            arena.alloc_slot(a0, pos).unwrap();
            arena.alloc_slot(a1, pos).unwrap();
            arena.alloc_slot(a2, pos).unwrap();
        }

        // Priority order: a2, a0, a1
        let ranges = arena.compact_for_batch(&[a2, a0, a1]);
        assert_eq!(ranges[0], (a2, 0, 3));
        assert_eq!(ranges[1], (a0, 3, 3));
        assert_eq!(ranges[2], (a1, 6, 3));
    }

    #[test]
    fn compact_for_batch_single_degenerates() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();

        let ranges = arena.compact_for_batch(&[a0]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (a0, 0, 2));

        // a0 at front, a1 follows
        let a0_slots = arena.agent_slots(a0);
        assert_eq!(a0_slots[0].0, SlotId(0));
        assert_eq!(a0_slots[1].0, SlotId(1));
    }

    #[test]
    fn compact_for_batch_already_ordered() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Allocate in order: a0 then a1
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a1, 1).unwrap();

        // Already in the right order — should hit fast path
        let ranges = arena.compact_for_batch(&[a0, a1]);
        assert_eq!(ranges[0], (a0, 0, 2));
        assert_eq!(ranges[1], (a1, 2, 2));
    }

    // ── Shared region tests ────────────────────────────────────

    #[test]
    fn create_shared_region_allocates_slots() {
        let mut arena = KvArena::new(&test_config(8));
        let (region, slots) = arena.create_shared_region(3).unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(arena.free_slots(), 5);
        assert_eq!(arena.shared_region_len(region), Some(3));

        // Verify slots are marked Shared
        for &sid in &slots {
            assert!(arena.slot_state(sid).is_shared());
            assert_eq!(arena.slot_state(sid).region(), Some(region));
        }
    }

    #[test]
    fn shared_slots_not_evictable() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);

        let (region, shared_slots) = arena.create_shared_region(2).unwrap();
        let _ = region;
        let agent_slot = arena.alloc_slot(a0, 0).unwrap();

        // Only agent slot should be evictable
        let candidates = arena.evictable_slots();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, agent_slot);

        // mark_evict should skip shared slots
        arena.mark_evict(&[shared_slots[0], shared_slots[1], agent_slot]);
        assert!(arena.slot_state(shared_slots[0]).is_shared()); // still shared
        assert!(arena.slot_state(shared_slots[1]).is_shared()); // still shared
        assert!(arena.slot_state(agent_slot).is_free()); // evicted
    }

    #[test]
    fn shared_region_ref_counting() {
        let mut arena = KvArena::new(&test_config(8));
        let (region, _slots) = arena.create_shared_region(3).unwrap();

        arena.ref_shared_region(region).unwrap();
        arena.ref_shared_region(region).unwrap();
        assert_eq!(arena.free_slots(), 5);

        // First unref: count goes to 1, slots stay
        arena.unref_shared_region(region).unwrap();
        assert_eq!(arena.free_slots(), 5);
        assert_eq!(arena.shared_region_len(region), Some(3));

        // Second unref: count goes to 0, slots freed
        arena.unref_shared_region(region).unwrap();
        assert_eq!(arena.free_slots(), 8);
        assert_eq!(arena.shared_region_len(region), None); // region gone
    }

    #[test]
    fn free_agent_ignores_shared_slots() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);

        let (_region, _shared_slots) = arena.create_shared_region(2).unwrap();
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        assert_eq!(arena.free_slots(), 4);

        arena.free_agent(a0);
        assert_eq!(arena.free_slots(), 6); // only agent slots freed, shared remain
        assert_eq!(arena.occupied_count(), 2); // shared slots still occupied
    }

    #[test]
    fn compact_with_shared_at_front() {
        let mut arena = KvArena::new(&test_config(12));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Create shared region first, then agent slots interleaved
        let (region, _shared_slots) = arena.create_shared_region(3).unwrap();
        arena.alloc_slot(a0, 0).unwrap();
        arena.alloc_slot(a1, 0).unwrap();
        arena.alloc_slot(a0, 1).unwrap();
        arena.alloc_slot(a1, 1).unwrap();

        let (shared_len, ranges) =
            arena.compact_for_batch_with_shared(&[a0, a1], &[region]);

        assert_eq!(shared_len, 3);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], (a0, 3, 2)); // a0 private at offset 3
        assert_eq!(ranges[1], (a1, 5, 2)); // a1 private at offset 5

        // Verify shared slots at front (indices 0, 1, 2)
        for i in 0..3 {
            assert!(arena.slots[i].is_shared());
        }
        // Verify agent slots follow
        assert_eq!(arena.slots[3].agent(), Some(a0));
        assert_eq!(arena.slots[4].agent(), Some(a0));
        assert_eq!(arena.slots[5].agent(), Some(a1));
        assert_eq!(arena.slots[6].agent(), Some(a1));
    }

    #[test]
    fn compact_with_shared_preserves_data() {
        let mut arena = KvArena::new(&test_config(8));
        let a0 = AgentId(0);

        // Create shared region and agent slot interleaved
        let (region, shared_slots) = arena.create_shared_region(2).unwrap();
        let agent_slot = arena.alloc_slot(a0, 0).unwrap();

        // Write patterns
        let entry_size = arena.entry_size();
        unsafe {
            std::ptr::write_bytes(arena.slot_key_ptr(0, shared_slots[0]), 0xCC, entry_size);
            std::ptr::write_bytes(arena.slot_key_ptr(0, shared_slots[1]), 0xDD, entry_size);
            std::ptr::write_bytes(arena.slot_key_ptr(0, agent_slot), 0xEE, entry_size);
        }

        let (shared_len, ranges) =
            arena.compact_for_batch_with_shared(&[a0], &[region]);

        assert_eq!(shared_len, 2);
        assert_eq!(ranges[0], (a0, 2, 1));

        // Verify data integrity after compaction
        unsafe {
            let mut buf = vec![0u8; entry_size];

            // Shared slot 0 should be at index 0
            std::ptr::copy_nonoverlapping(
                arena.slot_key_ptr(0, SlotId(0)),
                buf.as_mut_ptr(),
                entry_size,
            );
            assert!(buf.iter().all(|&b| b == 0xCC));

            // Shared slot 1 should be at index 1
            std::ptr::copy_nonoverlapping(
                arena.slot_key_ptr(0, SlotId(1)),
                buf.as_mut_ptr(),
                entry_size,
            );
            assert!(buf.iter().all(|&b| b == 0xDD));

            // Agent slot should be at index 2
            std::ptr::copy_nonoverlapping(
                arena.slot_key_ptr(0, SlotId(2)),
                buf.as_mut_ptr(),
                entry_size,
            );
            assert!(buf.iter().all(|&b| b == 0xEE));
        }
    }

    #[test]
    fn compact_for_agent_preserves_kv_data() {
        let mut arena = KvArena::new(&test_config(4));
        let a0 = AgentId(0);
        let a1 = AgentId(1);

        // Allocate interleaved
        let s_a0_0 = arena.alloc_slot(a0, 0).unwrap();
        let s_a1_0 = arena.alloc_slot(a1, 0).unwrap();

        // Write recognizable patterns to each slot's K buffer (layer 0)
        let entry_size = arena.entry_size();
        unsafe {
            let ptr_a0 = arena.slot_key_ptr(0, s_a0_0);
            std::ptr::write_bytes(ptr_a0, 0xAA, entry_size);

            let ptr_a1 = arena.slot_key_ptr(0, s_a1_0);
            std::ptr::write_bytes(ptr_a1, 0xBB, entry_size);
        }

        // Compact for a1 — a1 should move to front
        arena.compact_for_agent(a1);

        // After compaction: a1 at slot 0, a0 at slot 1
        let a1_slots = arena.agent_slots(a1);
        let a0_slots = arena.agent_slots(a0);

        // Verify data moved correctly
        unsafe {
            let ptr_a1_new = arena.slot_key_ptr(0, a1_slots[0].0);
            let mut buf = vec![0u8; entry_size];
            std::ptr::copy_nonoverlapping(ptr_a1_new, buf.as_mut_ptr(), entry_size);
            assert!(buf.iter().all(|&b| b == 0xBB), "a1 data should be 0xBB");

            let ptr_a0_new = arena.slot_key_ptr(0, a0_slots[0].0);
            std::ptr::copy_nonoverlapping(ptr_a0_new, buf.as_mut_ptr(), entry_size);
            assert!(buf.iter().all(|&b| b == 0xAA), "a0 data should be 0xAA");
        }
    }
}
