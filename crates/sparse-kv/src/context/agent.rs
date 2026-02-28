use crate::arena::slot::{AgentId, SlotId};
use crate::grammar::GrammarEngine;

pub struct AgentContext {
    pub id: AgentId,
    pub slots: Vec<SlotId>,
    pub seq_len: usize,
    pub last_logits: Vec<f32>,
    /// Optional grammar constraint for this agent's generation.
    pub grammar: Option<GrammarEngine>,
}

impl AgentContext {
    pub fn new(id: AgentId) -> Self {
        Self {
            id,
            slots: Vec::new(),
            seq_len: 0,
            last_logits: Vec::new(),
            grammar: None,
        }
    }

    pub fn push_slot(&mut self, slot: SlotId) {
        self.slots.push(slot);
        self.seq_len += 1;
    }
}
