use crate::arena::slot::AgentId;

/// Snapshot of an agent's state visible to the scheduler.
///
/// The scheduler sees only what it needs for scheduling decisions:
/// priority, sequence length, and whether the agent has pending work.
/// It never sees KV tensors, arena layout, or attention scores.
pub struct SchedulerEntry {
    pub id: AgentId,
    pub priority: i32,
    pub seq_len: usize,
    pub has_pending: bool,
}

/// Selects which agent gets the next forward pass.
///
/// From IDEA.md Principle 6: "No agent has an inherent right to compute.
/// The scheduler assembles each batch to maximize useful throughput,
/// not to be fair."
///
/// The scheduler sees a snapshot of all agents' scheduling-relevant state
/// and returns the next agent to decode. Returns None if no agents are ready.
pub trait Scheduler {
    fn next(&mut self, entries: &[SchedulerEntry]) -> Option<AgentId>;
}

/// Round-robin scheduler: cycles through ready agents in agent ID order.
///
/// Fair but not optimal. Useful as a baseline or when all agents
/// have equal priority.
pub struct RoundRobinScheduler {
    last_served: Option<AgentId>,
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self { last_served: None }
    }
}

impl Scheduler for RoundRobinScheduler {
    fn next(&mut self, entries: &[SchedulerEntry]) -> Option<AgentId> {
        let mut ready: Vec<_> = entries.iter().filter(|e| e.has_pending).collect();
        if ready.is_empty() {
            return None;
        }
        // Stable order by agent ID
        ready.sort_by_key(|e| e.id.0);

        let start_idx = match self.last_served {
            Some(last) => {
                // Find the first ready agent with ID strictly greater than last served
                ready
                    .iter()
                    .position(|e| e.id.0 > last.0)
                    .unwrap_or(0) // wrap around
            }
            None => 0,
        };

        let picked = ready[start_idx];
        self.last_served = Some(picked.id);
        Some(picked.id)
    }
}

/// Priority scheduler: always picks the highest-priority ready agent.
///
/// Ties are broken by lowest agent ID (stable, deterministic).
/// Starves low-priority agents when high-priority agents always have work.
pub struct PriorityScheduler;

impl PriorityScheduler {
    pub fn new() -> Self {
        Self
    }
}

impl Scheduler for PriorityScheduler {
    fn next(&mut self, entries: &[SchedulerEntry]) -> Option<AgentId> {
        entries
            .iter()
            .filter(|e| e.has_pending)
            .max_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then(b.id.0.cmp(&a.id.0)) // lower ID wins ties
            })
            .map(|e| e.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u32, priority: i32, has_pending: bool) -> SchedulerEntry {
        SchedulerEntry {
            id: AgentId(id),
            priority,
            seq_len: 0,
            has_pending,
        }
    }

    #[test]
    fn round_robin_cycles() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, true), entry(2, 0, true)];

        assert_eq!(sched.next(&entries), Some(AgentId(0)));
        assert_eq!(sched.next(&entries), Some(AgentId(1)));
        assert_eq!(sched.next(&entries), Some(AgentId(2)));
        assert_eq!(sched.next(&entries), Some(AgentId(0))); // wraps
    }

    #[test]
    fn round_robin_skips_not_ready() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, false), entry(2, 0, true)];

        assert_eq!(sched.next(&entries), Some(AgentId(0)));
        assert_eq!(sched.next(&entries), Some(AgentId(2)));
        assert_eq!(sched.next(&entries), Some(AgentId(0)));
    }

    #[test]
    fn round_robin_none_ready() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, false), entry(1, 0, false)];
        assert_eq!(sched.next(&entries), None);
    }

    #[test]
    fn round_robin_single_agent() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(5, 0, true)];
        assert_eq!(sched.next(&entries), Some(AgentId(5)));
        assert_eq!(sched.next(&entries), Some(AgentId(5)));
    }

    #[test]
    fn priority_picks_highest() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(0, 1, true), entry(1, 10, true), entry(2, 5, true)];
        assert_eq!(sched.next(&entries), Some(AgentId(1)));
    }

    #[test]
    fn priority_breaks_ties_by_lowest_id() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(2, 5, true), entry(0, 5, true), entry(1, 5, true)];
        assert_eq!(sched.next(&entries), Some(AgentId(0)));
    }

    #[test]
    fn priority_skips_not_ready() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(0, 10, false), entry(1, 1, true)];
        assert_eq!(sched.next(&entries), Some(AgentId(1)));
    }

    #[test]
    fn priority_none_ready() {
        let mut sched = PriorityScheduler::new();
        let entries: Vec<SchedulerEntry> = vec![];
        assert_eq!(sched.next(&entries), None);
    }
}
