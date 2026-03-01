use crate::arena::kv_arena::KvArena;
use crate::arena::slot::{AgentId, SlotId};
use crate::grammar::GrammarEngine;

pub struct AgentContext {
    pub id: AgentId,
    pub slots: Vec<SlotId>,
    pub seq_len: usize,
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
            last_logits: Vec::new(),
            grammar: None,
            priority: 0,
            pending_token: None,
        }
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
