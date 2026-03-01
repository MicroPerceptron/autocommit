#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedRegionId(pub u32);

pub type LogicalPos = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Free,
    Occupied {
        agent: AgentId,
        logical_pos: LogicalPos,
    },
    Pinned {
        agent: AgentId,
        logical_pos: LogicalPos,
    },
    /// Immutable shared KV entry. Belongs to a shared region, not an agent.
    /// Protected by ref counting — cannot be evicted while any agent references it.
    Shared {
        region: SharedRegionId,
        logical_pos: LogicalPos,
    },
}

impl SlotState {
    pub fn is_free(self) -> bool {
        matches!(self, Self::Free)
    }

    pub fn is_occupied(self) -> bool {
        !self.is_free()
    }

    pub fn agent(self) -> Option<AgentId> {
        match self {
            Self::Free | Self::Shared { .. } => None,
            Self::Occupied { agent, .. } | Self::Pinned { agent, .. } => Some(agent),
        }
    }

    pub fn logical_pos(self) -> Option<LogicalPos> {
        match self {
            Self::Free => None,
            Self::Occupied { logical_pos, .. }
            | Self::Pinned { logical_pos, .. }
            | Self::Shared { logical_pos, .. } => Some(logical_pos),
        }
    }

    pub fn is_pinned(self) -> bool {
        matches!(self, Self::Pinned { .. })
    }

    pub fn is_shared(self) -> bool {
        matches!(self, Self::Shared { .. })
    }

    pub fn region(self) -> Option<SharedRegionId> {
        match self {
            Self::Shared { region, .. } => Some(region),
            _ => None,
        }
    }
}
