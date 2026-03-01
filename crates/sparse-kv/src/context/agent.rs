use crate::arena::kv_arena::KvArena;
use crate::arena::slot::{AgentId, SharedRegionId, SlotId};
use crate::grammar::GrammarEngine;

pub struct AgentContext {
    pub id: AgentId,
    /// Private slots owned by this agent (not shared).
    pub slots: Vec<SlotId>,
    /// Private sequence length (number of private KV entries).
    pub seq_len: usize,
    /// Shared regions attached to this agent.
    pub shared_regions: Vec<SharedRegionId>,
    /// Total shared prefix length across all attached regions.
    pub shared_seq_len: usize,
    pub last_logits: Vec<f32>,
    /// Optional grammar constraint for this agent's generation.
    pub grammar: Option<GrammarEngine>,
    /// Scheduling priority. Higher = more important. Default 0.
    pub priority: i32,
    /// Next token to decode, set by the caller or sampler.
    /// Consumed by `InferenceEngine::step()`.
    pub pending_token: Option<i32>,
}

impl AgentContext {
    pub fn new(id: AgentId) -> Self {
        Self {
            id,
            slots: Vec::new(),
            seq_len: 0,
            shared_regions: Vec::new(),
            shared_seq_len: 0,
            last_logits: Vec::new(),
            grammar: None,
            priority: 0,
            pending_token: None,
        }
    }

    /// Total effective sequence length (shared prefix + private tokens).
    /// Used for RoPE position computation.
    pub fn effective_seq_len(&self) -> usize {
        self.shared_seq_len + self.seq_len
    }

    /// Attach a shared region to this agent.
    pub fn attach_shared(&mut self, region: SharedRegionId, region_len: usize) {
        self.shared_regions.push(region);
        self.shared_seq_len += region_len;
    }

    /// Detach a shared region from this agent.
    pub fn detach_shared(&mut self, region: SharedRegionId, region_len: usize) {
        self.shared_regions.retain(|&r| r != region);
        self.shared_seq_len = self.shared_seq_len.saturating_sub(region_len);
    }

    pub fn push_slot(&mut self, slot: SlotId) {
        self.slots.push(slot);
        self.seq_len += 1;
    }

    /// Rebuild slot list from the arena after compaction.
    /// Compaction moves slots physically, so cached SlotIds become stale.
    pub fn rebuild_slots(&mut self, arena: &KvArena) {
        let live = arena.agent_slots(self.id);
        self.slots = live.iter().map(|(sid, _)| *sid).collect();
        self.seq_len = self.slots.len();
    }
}
