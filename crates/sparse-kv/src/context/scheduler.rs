use crate::arena::slot::AgentId;

/// Snapshot of an agent's state visible to the scheduler.
///
/// Exposes task-level (priority, pending work) and resource-level
/// (KV slot count, residency) information without leaking arena
/// internals like tensor pointers or physical layout.
pub struct SchedulerEntry {
    pub id: AgentId,
    pub priority: i32,
    pub seq_len: usize,
    pub has_pending: bool,
    /// Number of KV slots this agent currently occupies.
    pub kv_slots: usize,
    /// Whether the agent's KV state is resident in device memory.
    /// Always true until device-host swapping is implemented.
    pub resident: bool,
}

/// Resource constraints the scheduler must respect when assembling a batch.
///
/// Invariant 7 (capacity honesty): the scheduler must never commit to a
/// batch that would require more arena capacity than is physically available.
pub struct BatchConstraints {
    /// Maximum number of agents in this batch. Bounded by arena free
    /// slots (each agent needs 1 new slot per decode step).
    pub max_agents: usize,
    /// Free KV slots available in the arena.
    pub arena_free: usize,
    /// Total arena capacity.
    pub arena_capacity: usize,
}

/// Assembles batches of agents for each forward pass.
///
/// From IDEA.md Principle 6: "No agent has an inherent right to compute.
/// The scheduler assembles each batch to maximize useful throughput,
/// not to be fair."
///
/// Returns an ordered list of agent IDs. The engine processes them
/// sequentially until continuous batching (multi-agent single graph)
/// is implemented. Order matters — first agent gets lowest-latency
/// decode (no compaction needed if it was already at front).
pub trait Scheduler: Send {
    /// Select which agents participate in this step's batch.
    ///
    /// Must respect `constraints.max_agents` — exceeding it is a bug.
    /// Returns empty vec if no agents are ready.
    fn assemble_batch(
        &mut self,
        entries: &[SchedulerEntry],
        constraints: &BatchConstraints,
    ) -> Vec<AgentId>;
}

/// Round-robin scheduler: cycles through ready agents, filling the
/// batch up to capacity in agent ID order starting after the last served.
///
/// Fair but not throughput-optimal. Useful as a baseline or when all
/// agents have equal priority.
pub struct RoundRobinScheduler {
    last_served: Option<AgentId>,
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self { last_served: None }
    }
}

impl Scheduler for RoundRobinScheduler {
    fn assemble_batch(
        &mut self,
        entries: &[SchedulerEntry],
        constraints: &BatchConstraints,
    ) -> Vec<AgentId> {
        let mut ready: Vec<_> = entries.iter().filter(|e| e.has_pending).collect();
        if ready.is_empty() {
            return Vec::new();
        }
        ready.sort_by_key(|e| e.id.0);

        let max = constraints.max_agents.min(ready.len());
        let start_idx = match self.last_served {
            Some(last) => ready
                .iter()
                .position(|e| e.id.0 > last.0)
                .unwrap_or(0),
            None => 0,
        };

        let mut batch = Vec::with_capacity(max);
        let n = ready.len();
        for i in 0..n {
            if batch.len() >= max {
                break;
            }
            let idx = (start_idx + i) % n;
            batch.push(ready[idx].id);
            self.last_served = Some(ready[idx].id);
        }
        batch
    }
}

/// Priority scheduler: fills the batch with the highest-priority
/// ready agents, respecting capacity constraints.
///
/// Starves low-priority agents when high-priority agents always have
/// work. This is intentional — per P6, the scheduler maximizes useful
/// throughput, not fairness.
pub struct PriorityScheduler;

impl PriorityScheduler {
    pub fn new() -> Self {
        Self
    }
}

impl Scheduler for PriorityScheduler {
    fn assemble_batch(
        &mut self,
        entries: &[SchedulerEntry],
        constraints: &BatchConstraints,
    ) -> Vec<AgentId> {
        let mut ready: Vec<_> = entries.iter().filter(|e| e.has_pending).collect();
        if ready.is_empty() {
            return Vec::new();
        }

        // Highest priority first, lowest ID breaks ties
        ready.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.id.0.cmp(&b.id.0))
        });

        let max = constraints.max_agents.min(ready.len());
        ready.iter().take(max).map(|e| e.id).collect()
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
            kv_slots: 0,
            resident: true,
        }
    }

    fn unconstrained() -> BatchConstraints {
        BatchConstraints {
            max_agents: 100,
            arena_free: 100,
            arena_capacity: 1000,
        }
    }

    fn limited(max: usize) -> BatchConstraints {
        BatchConstraints {
            max_agents: max,
            arena_free: max,
            arena_capacity: 1000,
        }
    }

    // ── Round-robin ────────────────────────────────────────────

    #[test]
    fn round_robin_fills_batch() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, true), entry(2, 0, true)];

        let batch = sched.assemble_batch(&entries, &unconstrained());
        assert_eq!(batch, vec![AgentId(0), AgentId(1), AgentId(2)]);
    }

    #[test]
    fn round_robin_respects_limit() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, true), entry(2, 0, true)];

        let batch = sched.assemble_batch(&entries, &limited(2));
        assert_eq!(batch, vec![AgentId(0), AgentId(1)]);

        // Next batch starts where we left off
        let batch = sched.assemble_batch(&entries, &limited(2));
        assert_eq!(batch, vec![AgentId(2), AgentId(0)]);
    }

    #[test]
    fn round_robin_skips_not_ready() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, false), entry(2, 0, true)];

        let batch = sched.assemble_batch(&entries, &unconstrained());
        assert_eq!(batch, vec![AgentId(0), AgentId(2)]);
    }

    #[test]
    fn round_robin_wraps_around() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, true), entry(1, 0, true)];

        let b1 = sched.assemble_batch(&entries, &limited(1));
        assert_eq!(b1, vec![AgentId(0)]);

        let b2 = sched.assemble_batch(&entries, &limited(1));
        assert_eq!(b2, vec![AgentId(1)]);

        let b3 = sched.assemble_batch(&entries, &limited(1));
        assert_eq!(b3, vec![AgentId(0)]); // wraps
    }

    #[test]
    fn round_robin_none_ready() {
        let mut sched = RoundRobinScheduler::new();
        let entries = vec![entry(0, 0, false), entry(1, 0, false)];
        assert!(sched.assemble_batch(&entries, &unconstrained()).is_empty());
    }

    // ── Priority ───────────────────────────────────────────────

    #[test]
    fn priority_fills_by_rank() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(0, 1, true), entry(1, 10, true), entry(2, 5, true)];

        let batch = sched.assemble_batch(&entries, &unconstrained());
        assert_eq!(batch, vec![AgentId(1), AgentId(2), AgentId(0)]);
    }

    #[test]
    fn priority_respects_limit() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(0, 1, true), entry(1, 10, true), entry(2, 5, true)];

        let batch = sched.assemble_batch(&entries, &limited(2));
        assert_eq!(batch, vec![AgentId(1), AgentId(2)]);
    }

    #[test]
    fn priority_breaks_ties_by_lowest_id() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(2, 5, true), entry(0, 5, true), entry(1, 5, true)];

        let batch = sched.assemble_batch(&entries, &unconstrained());
        assert_eq!(batch, vec![AgentId(0), AgentId(1), AgentId(2)]);
    }

    #[test]
    fn priority_skips_not_ready() {
        let mut sched = PriorityScheduler::new();
        let entries = vec![entry(0, 10, false), entry(1, 1, true)];

        let batch = sched.assemble_batch(&entries, &unconstrained());
        assert_eq!(batch, vec![AgentId(1)]);
    }

    #[test]
    fn priority_none_ready() {
        let mut sched = PriorityScheduler::new();
        let entries: Vec<SchedulerEntry> = vec![];
        assert!(sched.assemble_batch(&entries, &unconstrained()).is_empty());
    }
}
