use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;

use crate::arena::kv_arena::{ArenaConfig, KvArena};
use crate::arena::slot::{AgentId, SharedRegionId};
use crate::context::agent::AgentContext;
use crate::context::batch;
use crate::context::continuous;
use crate::context::decode;
use crate::context::scheduler::{BatchConstraints, Scheduler, SchedulerEntry};
use crate::context::shared_prefill;
use crate::error::InferenceError;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::grammar::GrammarEngine;
use crate::sampling::{Sampler, SamplerParams};
use crate::sparse::eviction::EvictionConfig;
use crate::tokenizer::Tokenizer;

// ── Shared / Mutable split ────────────────────────────────────────

/// Read-only model state, safe to share across threads via Arc.
pub struct EngineShared {
    pub reader: GgufReader,
    pub config: ModelConfig,
    pub tokenizer: Option<Tokenizer>,
}

/// Mutable inference state. Owned by a single thread (directly or via engine loop).
pub(crate) struct EngineMut {
    backend: Backend,
    arena: KvArena,
    agents: Vec<AgentContext>,
    next_agent_id: u32,
    eviction: Option<EvictionConfig>,
    arena_dirty: bool,
    scheduler: Option<Box<dyn Scheduler>>,
    /// Active generation states, keyed by AgentId.0.
    gen_states: HashMap<u32, GenerateState>,
}

impl EngineMut {
    fn run_eviction(&mut self) {
        let eviction = match &self.eviction {
            Some(e) => e,
            None => return,
        };

        let capacity = self.arena.capacity();
        let free_ratio = self.arena.free_slots() as f32 / capacity as f32;
        if free_ratio >= eviction.trigger_threshold {
            return;
        }

        let occupied = self.arena.occupied_count();
        let need = (eviction.evict_fraction * occupied as f32).ceil() as usize;
        if need == 0 {
            return;
        }

        let candidates = self.arena.evictable_slots();
        let to_evict = eviction.policy.select(&candidates, need);

        self.arena.mark_evict(&to_evict);
        self.arena_dirty = true;

        for agent in &mut self.agents {
            agent.rebuild_slots(&self.arena);
        }
    }

    fn needs_compact(&self) -> bool {
        self.agents.len() > 1 || self.arena_dirty
    }
}

// ── Command protocol ─────────────────────────────────────────────

type Reply<T> = mpsc::Sender<T>;

/// Commands sent from `EngineHandle` to the engine thread.
enum Cmd {
    // Fire-and-forget (invalid agent = logic bug, silently ignored)
    DestroyAgent(AgentId),
    SetEviction(EvictionConfig),
    EnqueueToken { agent: AgentId, token: i32 },
    SetPriority { agent: AgentId, priority: i32 },
    SetScheduler(Box<dyn Scheduler>),

    // With reply
    CreateAgent(Reply<AgentId>),
    CreateSharedRegion {
        tokens: Vec<i32>,
        reply: Reply<Result<SharedRegionId, InferenceError>>,
    },
    AttachShared {
        agent: AgentId,
        region: SharedRegionId,
        reply: Reply<Result<(), InferenceError>>,
    },
    DetachShared {
        agent: AgentId,
        region: SharedRegionId,
        reply: Reply<Result<(), InferenceError>>,
    },
    Decode {
        agent: AgentId,
        token: i32,
        reply: Reply<Result<Vec<f32>, InferenceError>>,
    },
    DecodeBatch {
        agent: AgentId,
        tokens: Vec<i32>,
        reply: Reply<Result<Vec<f32>, InferenceError>>,
    },

    // Non-blocking: sets up GenerateState, generation happens in tick loop.
    // Grammar constraint is determined by agent.grammar at each tick.
    Generate {
        agent: AgentId,
        prompt: Vec<i32>,
        params: SamplerParams,
        max_tokens: usize,
        token_tx: mpsc::Sender<i32>,
        reply: Reply<Result<Vec<i32>, InferenceError>>,
    },
    CancelGenerate { agent: AgentId },

    SetGrammar {
        agent: AgentId,
        gbnf: String,
        reply: Reply<Result<(), InferenceError>>,
    },
    ClearGrammar {
        agent: AgentId,
        reply: Reply<Result<(), InferenceError>>,
    },

    BackendName(Reply<String>),
    FreeKvSlots(Reply<usize>),
    KvCapacity(Reply<usize>),

    Shutdown,
}

/// Generation phase: prefill or sampling.
enum GenPhase {
    /// Ingesting prompt tokens in chunks. No output yet.
    Prefill {
        tokens: Vec<i32>,
        offset: usize,
        chunk_size: usize,
    },
    /// Prompt fully ingested; iteratively sampling output tokens.
    Sampling,
}

/// Default prefill chunk size (tokens per forward pass during prompt ingestion).
const PREFILL_CHUNK: usize = 256;

/// Per-agent generation state for the engine loop.
///
/// Grammar constraint is not stored here — it's checked dynamically from
/// `AgentContext.grammar` each tick. This makes grammar the single source
/// of truth and avoids synchronization invariants between two fields.
struct GenerateState {
    phase: GenPhase,
    remaining: usize,
    context: Vec<i32>,
    output: Vec<i32>,
    sampler: Sampler,
    token_tx: mpsc::Sender<i32>,
    reply: Option<Reply<Result<Vec<i32>, InferenceError>>>,
}

// ── Shared implementation functions ───────────────────────────────
//
// These are used by both InferenceEngine (direct) and the engine
// loop (channel-based) to avoid duplicating logic.

fn decode_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
    agent_id: AgentId,
    token_id: i32,
) -> Result<Vec<f32>, InferenceError> {
    m.run_eviction();

    let (effective_pos, shared_regions) = m
        .agents
        .iter()
        .find(|a| a.id == agent_id)
        .map(|a| (a.effective_seq_len(), a.shared_regions.clone()))
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;

    let logical_pos = effective_pos as u32;
    let new_slot = m.arena.alloc_slot(agent_id, logical_pos)?;

    let (kv_offset, final_slot) = if m.needs_compact() {
        if shared_regions.is_empty() {
            m.arena.compact_for_agent(agent_id);
        } else {
            m.arena
                .compact_for_batch_with_shared(&[agent_id], &shared_regions);
        }
        m.arena_dirty = false;
        for agent in &mut m.agents {
            agent.rebuild_slots(&m.arena);
        }
        let agent = m
            .agents
            .iter()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} lost during compaction", agent_id))
            })?;
        let last_slot = *agent.slots.last().ok_or_else(|| {
            InferenceError::Arena("agent has no slots after compaction".into())
        })?;
        (0, last_slot)
    } else {
        let agent = m
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;
        agent.push_slot(new_slot);
        (0, new_slot)
    };

    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;

    decode::decode_one(
        agent,
        token_id,
        &mut m.arena,
        &shared.reader,
        &shared.config,
        &m.backend,
        kv_offset,
        final_slot,
    )
}

fn decode_batch_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
    agent_id: AgentId,
    token_ids: &[i32],
) -> Result<Vec<f32>, InferenceError> {
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }

    m.run_eviction();

    let (seq_len_before, shared_seq_len, shared_regions) = m
        .agents
        .iter()
        .find(|a| a.id == agent_id)
        .map(|a| (a.seq_len, a.shared_seq_len, a.shared_regions.clone()))
        .ok_or_else(|| InferenceError::Agent(format!("agent {:?} not found", agent_id)))?;

    let effective_seq_len_before = shared_seq_len + seq_len_before;

    let n_tokens = token_ids.len();
    let mut new_slots = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        let logical_pos = (effective_seq_len_before + i) as u32;
        new_slots.push(m.arena.alloc_slot(agent_id, logical_pos)?);
    }

    let (kv_offset, updated_slots) = if m.needs_compact() {
        if shared_regions.is_empty() {
            m.arena.compact_for_agent(agent_id);
        } else {
            m.arena
                .compact_for_batch_with_shared(&[agent_id], &shared_regions);
        }
        m.arena_dirty = false;
        for agent in &mut m.agents {
            agent.rebuild_slots(&m.arena);
        }
        let agent = m
            .agents
            .iter()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} lost during compaction", agent_id))
            })?;
        let slots = agent.slots[seq_len_before..].to_vec();

        let agent = m
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} lost during compaction", agent_id))
            })?;
        agent.slots.truncate(seq_len_before);
        agent.seq_len = seq_len_before;

        (0, slots)
    } else {
        (0, new_slots)
    };

    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;

    batch::decode_batch(
        agent,
        token_ids,
        &updated_slots,
        &mut m.arena,
        &shared.reader,
        &shared.config,
        &m.backend,
        effective_seq_len_before,
        kv_offset,
    )
}

fn step_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
) -> Result<Vec<(AgentId, Vec<f32>)>, InferenceError> {
    let scheduler = match &mut m.scheduler {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };

    let entries: Vec<SchedulerEntry> = m
        .agents
        .iter()
        .map(|a| SchedulerEntry {
            id: a.id,
            priority: a.priority,
            seq_len: a.seq_len,
            has_pending: a.pending_token.is_some(),
            kv_slots: a.slots.len(),
            resident: true,
        })
        .collect();

    let constraints = BatchConstraints {
        max_agents: m.arena.free_slots(),
        arena_free: m.arena.free_slots(),
        arena_capacity: m.arena.capacity(),
    };

    let batch = scheduler.assemble_batch(&entries, &constraints);
    if batch.is_empty() {
        return Ok(Vec::new());
    }

    // Single-agent fast path: skip mask overhead
    if batch.len() == 1 {
        let agent_id = batch[0];
        let token = m
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .and_then(|a| a.pending_token.take())
            .ok_or_else(|| {
                InferenceError::Agent(format!(
                    "scheduler selected agent {:?} but no pending token",
                    agent_id
                ))
            })?;
        let logits = decode_impl(shared, m, agent_id, token)?;
        return Ok(vec![(agent_id, logits)]);
    }

    // ── Multi-agent continuous batching ──────────────────────

    m.run_eviction();

    let mut token_ids = Vec::with_capacity(batch.len());
    let mut positions = Vec::with_capacity(batch.len());
    for &agent_id in &batch {
        let agent = m
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;
        let token = agent.pending_token.take().ok_or_else(|| {
            InferenceError::Agent(format!(
                "scheduler selected agent {:?} but no pending token",
                agent_id
            ))
        })?;
        token_ids.push(token);
        positions.push(agent.effective_seq_len() as i32);
    }

    let mut new_slots = Vec::with_capacity(batch.len());
    for (i, &agent_id) in batch.iter().enumerate() {
        let logical_pos = positions[i] as u32;
        new_slots.push(m.arena.alloc_slot(agent_id, logical_pos)?);
    }

    let mut all_shared: Vec<SharedRegionId> = Vec::new();
    for &agent_id in &batch {
        if let Some(agent) = m.agents.iter().find(|a| a.id == agent_id) {
            for &region in &agent.shared_regions {
                if !all_shared.contains(&region) {
                    all_shared.push(region);
                }
            }
        }
    }

    let (shared_len, compact_ranges) = if all_shared.is_empty() {
        let ranges = m.arena.compact_for_batch(&batch);
        (0, ranges)
    } else {
        m.arena.compact_for_batch_with_shared(&batch, &all_shared)
    };
    m.arena_dirty = false;
    for agent in &mut m.agents {
        agent.rebuild_slots(&m.arena);
    }

    let ranges: Vec<(usize, usize)> = compact_ranges
        .iter()
        .map(|&(_, offset, count)| (offset, count))
        .collect();

    let total_kv_len = ranges
        .last()
        .map(|&(off, cnt)| off + cnt)
        .unwrap_or(0);

    let mut final_slots = Vec::with_capacity(batch.len());
    for &aid in &batch {
        let agent = m.agents.iter().find(|a| a.id == aid).ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} lost during compaction", aid))
        })?;
        let slot = agent.slots.last().copied().ok_or_else(|| {
            InferenceError::Arena("agent has no slots after compaction".into())
        })?;
        final_slots.push(slot);
    }

    let all_logits = continuous::decode_continuous_batch(
        &token_ids,
        &positions,
        &final_slots,
        &ranges,
        &mut m.arena,
        &shared.reader,
        &shared.config,
        &m.backend,
        total_kv_len,
        shared_len,
    )?;

    for (i, &agent_id) in batch.iter().enumerate() {
        let agent = m.agents.iter_mut().find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} lost after decode", agent_id))
            })?;
        agent.last_logits = all_logits[i].clone();
    }

    Ok(batch.into_iter().zip(all_logits).collect())
}

fn create_shared_region_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
    tokens: &[i32],
) -> Result<SharedRegionId, InferenceError> {
    if tokens.is_empty() {
        return Err(InferenceError::Arena(
            "shared region must have at least 1 token".into(),
        ));
    }

    let (region_id, slot_ids) = m.arena.create_shared_region(tokens.len())?;

    let (shared_len, _) =
        m.arena.compact_for_batch_with_shared(&[], &[region_id]);
    debug_assert_eq!(shared_len, tokens.len());

    for agent in &mut m.agents {
        agent.rebuild_slots(&m.arena);
    }

    shared_prefill::prefill_shared_region(
        tokens,
        &slot_ids,
        &mut m.arena,
        &shared.reader,
        &shared.config,
        &m.backend,
        0,
    )?;

    Ok(region_id)
}

fn generate_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
    agent_id: AgentId,
    prompt_tokens: &[i32],
    sampler: &mut Sampler,
    max_tokens: usize,
) -> Result<Vec<i32>, InferenceError> {
    let mut logits = if prompt_tokens.is_empty() {
        Vec::new()
    } else {
        decode_batch_impl(shared, m, agent_id, prompt_tokens)?
    };

    if logits.is_empty() {
        return Ok(Vec::new());
    }

    let mut context: Vec<i32> = prompt_tokens.to_vec();
    let mut output = Vec::with_capacity(max_tokens);
    let eos_id = shared.tokenizer.as_ref().and_then(|t| t.eos_id());

    for _ in 0..max_tokens {
        let window = sampler.params.repeat_window;
        let start = context.len().saturating_sub(window);
        let recent = &context[start..];

        let token = sampler.sample(&logits, recent);

        if Some(token) == eos_id {
            break;
        }

        output.push(token);
        context.push(token);

        logits = decode_impl(shared, m, agent_id, token)?;
    }

    Ok(output)
}

fn generate_constrained_impl(
    shared: &EngineShared,
    m: &mut EngineMut,
    agent_id: AgentId,
    prompt_tokens: &[i32],
    sampler: &mut Sampler,
    max_tokens: usize,
) -> Result<Vec<i32>, InferenceError> {
    let mut logits = if prompt_tokens.is_empty() {
        Vec::new()
    } else {
        decode_batch_impl(shared, m, agent_id, prompt_tokens)?
    };

    if logits.is_empty() {
        return Ok(Vec::new());
    }

    let mut context: Vec<i32> = prompt_tokens.to_vec();
    let mut output = Vec::with_capacity(max_tokens);
    let eos_id = shared.tokenizer.as_ref().and_then(|t| t.eos_id());
    let n_vocab = logits.len();

    for _ in 0..max_tokens {
        let window = sampler.params.repeat_window;
        let start = context.len().saturating_sub(window);
        let recent = &context[start..];

        let agent = m.agents.iter_mut().find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;
        let token = if let Some(ref mut grammar) = agent.grammar {
            let trie = shared.tokenizer.as_ref()
                .ok_or_else(|| InferenceError::Tokenize("no tokenizer for grammar masking".into()))?
                .token_trie();
            let mask = grammar.token_mask(trie, n_vocab);
            sampler.sample_with_grammar(&logits, recent, mask)
        } else {
            sampler.sample(&logits, recent)
        };

        if Some(token) == eos_id {
            break;
        }

        let agent = m.agents.iter_mut().find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;
        if let Some(ref mut grammar) = agent.grammar {
            if let Some(tok_data) = shared.tokenizer.as_ref().and_then(|t| t.token_data(token)) {
                grammar.accept_token(tok_data)?;
            }
            if grammar.is_complete() {
                output.push(token);
                break;
            }
        }

        output.push(token);
        context.push(token);

        logits = decode_impl(shared, m, agent_id, token)?;
    }

    Ok(output)
}

// ── Engine event loop (runs in dedicated thread) ─────────────────

fn engine_loop(
    shared: Arc<EngineShared>,
    mut engine: EngineMut,
    rx: mpsc::Receiver<Cmd>,
) {
    use crate::context::scheduler::RoundRobinScheduler;

    if engine.scheduler.is_none() {
        engine.scheduler = Some(Box::new(RoundRobinScheduler::new()));
    }

    let eos_id = shared.tokenizer.as_ref().and_then(|t| t.eos_id());

    loop {
        // Phase 1: Drain all pending commands (non-blocking)
        loop {
            match rx.try_recv() {
                Ok(cmd) => {
                    if handle_cmd(&shared, &mut engine, cmd) {
                        return;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }

        // Phase 2: Advance prefilling agents by one chunk each
        tick_prefill(&shared, &mut engine);

        // Phase 3: Sample from last step's logits for mid-generation agents
        tick_generators(&shared, &mut engine, eos_id);

        // Phase 4: Run one forward pass for sampling agents with pending tokens
        let has_pending = engine.agents.iter().any(|a| a.pending_token.is_some());

        if has_pending {
            match step_impl(&shared, &mut engine) {
                Ok(_results) => {
                    // Logits stored on agent.last_logits by step_impl.
                    // They'll be sampled on next tick (Phase 3).
                }
                Err(e) => {
                    abort_generators(&mut engine, e);
                }
            }
        }

        // Only park if there is truly no work: no pending tokens, no prefill chunks
        let has_prefill = engine.gen_states.values().any(|gs| {
            matches!(gs.phase, GenPhase::Prefill { .. })
        });
        if !has_pending && !has_prefill {
            // No work — block until next command arrives
            match rx.recv() {
                Ok(cmd) => {
                    if handle_cmd(&shared, &mut engine, cmd) {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    }
}

/// Returns true on Shutdown.
fn handle_cmd(
    shared: &EngineShared,
    engine: &mut EngineMut,
    cmd: Cmd,
) -> bool {
    match cmd {
        Cmd::Shutdown => return true,

        // Fire-and-forget
        Cmd::DestroyAgent(id) => {
            // Implicitly cancel any active generation before destroying
            if let Some(gs) = engine.gen_states.remove(&id.0) {
                if let Some(reply) = gs.reply {
                    let _ = reply.send(Err(InferenceError::Agent(
                        "generation cancelled (agent destroyed)".into(),
                    )));
                }
            }
            destroy_agent_impl(engine, id);
        }
        Cmd::EnqueueToken { agent, token } => {
            let _ = enqueue_token_impl(engine, agent, token);
        }
        Cmd::SetPriority { agent, priority } => {
            let _ = set_priority_impl(engine, agent, priority);
        }
        Cmd::SetEviction(config) => {
            engine.eviction = Some(config);
        }
        Cmd::SetScheduler(s) => {
            engine.scheduler = Some(s);
        }

        // With reply
        Cmd::CreateAgent(reply) => {
            let id = AgentId(engine.next_agent_id);
            engine.next_agent_id += 1;
            engine.agents.push(AgentContext::new(id));
            let _ = reply.send(id);
        }
        Cmd::CreateSharedRegion { tokens, reply } => {
            let result = create_shared_region_impl(shared, engine, &tokens);
            let _ = reply.send(result);
        }
        Cmd::AttachShared { agent, region, reply } => {
            let result = attach_shared_region_impl(engine, agent, region);
            let _ = reply.send(result);
        }
        Cmd::DetachShared { agent, region, reply } => {
            let result = detach_shared_region_impl(engine, agent, region);
            let _ = reply.send(result);
        }
        Cmd::Decode { agent, token, reply } => {
            let result = decode_impl(shared, engine, agent, token);
            let _ = reply.send(result);
        }
        Cmd::DecodeBatch { agent, tokens, reply } => {
            let result = decode_batch_impl(shared, engine, agent, &tokens);
            let _ = reply.send(result);
        }
        Cmd::Generate {
            agent, prompt, params, max_tokens, token_tx, reply,
        } => {
            handle_generate(
                shared, engine, agent, prompt, params, max_tokens,
                token_tx, reply,
            );
        }
        Cmd::CancelGenerate { agent } => {
            if let Some(gs) = engine.gen_states.remove(&agent.0) {
                if let Some(reply) = gs.reply {
                    let _ = reply.send(Err(InferenceError::Agent(
                        "generation cancelled".into(),
                    )));
                }
            }
        }
        Cmd::SetGrammar { agent, gbnf, reply } => {
            let result = if engine.gen_states.contains_key(&agent.0) {
                Err(InferenceError::Agent(
                    "cannot set grammar while generation is active".into(),
                ))
            } else {
                set_grammar_impl(engine, agent, &gbnf)
            };
            let _ = reply.send(result);
        }
        Cmd::ClearGrammar { agent, reply } => {
            let result = if engine.gen_states.contains_key(&agent.0) {
                Err(InferenceError::Agent(
                    "cannot clear grammar while generation is active".into(),
                ))
            } else {
                clear_grammar_impl(engine, agent)
            };
            let _ = reply.send(result);
        }
        Cmd::BackendName(reply) => {
            let _ = reply.send(engine.backend.name().to_owned());
        }
        Cmd::FreeKvSlots(reply) => {
            let _ = reply.send(engine.arena.free_slots());
        }
        Cmd::KvCapacity(reply) => {
            let _ = reply.send(engine.arena.capacity());
        }
    }
    false
}

fn handle_generate(
    _shared: &EngineShared,
    engine: &mut EngineMut,
    agent: AgentId,
    prompt: Vec<i32>,
    params: SamplerParams,
    max_tokens: usize,
    token_tx: mpsc::Sender<i32>,
    reply: Reply<Result<Vec<i32>, InferenceError>>,
) {
    let phase = if prompt.is_empty() {
        GenPhase::Sampling
    } else {
        GenPhase::Prefill {
            tokens: prompt,
            offset: 0,
            chunk_size: PREFILL_CHUNK,
        }
    };

    engine.gen_states.insert(
        agent.0,
        GenerateState {
            phase,
            remaining: max_tokens,
            context: Vec::new(),
            output: Vec::new(),
            sampler: Sampler::new(params),
            token_tx,
            reply: Some(reply),
        },
    );
}

/// Sample a token for generation.
///
/// If the agent has a grammar attached, applies grammar masking before sampling.
/// Grammar presence is checked dynamically — no separate `constrained` flag needed.
fn sample_gen_token(
    shared: &EngineShared,
    engine: &mut EngineMut,
    agent_id: AgentId,
    logits: &[f32],
    context: &[i32],
    sampler: &mut Sampler,
) -> i32 {
    let window = sampler.params.repeat_window;
    let start = context.len().saturating_sub(window);
    let recent = &context[start..];

    let agent = match engine.agents.iter_mut().find(|a| a.id == agent_id) {
        Some(a) => a,
        None => return sampler.sample(logits, recent),
    };

    if let Some(ref mut grammar) = agent.grammar {
        if let Some(trie) = shared.tokenizer.as_ref().map(|t| t.token_trie()) {
            let mask = grammar.token_mask(trie, logits.len());
            return sampler.sample_with_grammar(logits, recent, mask);
        }
    }

    sampler.sample(logits, recent)
}

/// Advance prefilling agents by one chunk each.
///
/// For each agent in `GenPhase::Prefill`, feeds the next chunk of prompt tokens
/// via `decode_batch_impl`. When the final chunk is processed, transitions to
/// `GenPhase::Sampling` — the logits from the last chunk are used for the first
/// sample on the next tick.
fn tick_prefill(
    shared: &EngineShared,
    engine: &mut EngineMut,
) {
    let agent_keys: Vec<u32> = engine.gen_states.keys().copied().collect();
    for key in agent_keys {
        // Remove to avoid borrow conflict with engine
        let mut gs = match engine.gen_states.remove(&key) {
            Some(gs) => gs,
            None => continue,
        };

        let (tokens, offset, chunk_size) = match &mut gs.phase {
            GenPhase::Prefill { tokens, offset, chunk_size } => {
                (tokens as &Vec<i32>, *offset, *chunk_size)
            }
            GenPhase::Sampling => {
                engine.gen_states.insert(key, gs);
                continue;
            }
        };

        let end = (offset + chunk_size).min(tokens.len());
        let chunk = &tokens[offset..end];
        let agent_id = AgentId(key);

        match decode_batch_impl(shared, engine, agent_id, chunk) {
            Ok(_logits) => {
                // logits stored on agent.last_logits by decode_batch_impl
                if end >= tokens.len() {
                    // Prefill complete — transition to sampling
                    gs.context = match &gs.phase {
                        GenPhase::Prefill { tokens, .. } => tokens.clone(),
                        _ => unreachable!(),
                    };
                    gs.phase = GenPhase::Sampling;
                    // Don't set pending_token — tick_generators will sample
                    // from last_logits on the next tick
                } else {
                    // More chunks remain — advance offset
                    if let GenPhase::Prefill { offset, .. } = &mut gs.phase {
                        *offset = end;
                    }
                }
                engine.gen_states.insert(key, gs);
            }
            Err(e) => {
                // Prefill failed — report error, don't re-insert
                if let Some(reply) = gs.reply {
                    let _ = reply.send(Err(e));
                }
            }
        }
    }
}

fn tick_generators(
    shared: &EngineShared,
    engine: &mut EngineMut,
    eos_id: Option<i32>,
) {
    let agent_keys: Vec<u32> = engine.gen_states.keys().copied().collect();
    for key in agent_keys {
        let agent = match engine.agents.iter().find(|a| a.id.0 == key) {
            Some(a) => a,
            None => {
                engine.gen_states.remove(&key);
                continue;
            }
        };

        // Only sample if we have logits and no pending token yet
        if agent.last_logits.is_empty() || agent.pending_token.is_some() {
            continue;
        }

        let logits = agent.last_logits.clone();
        let agent_id = AgentId(key);

        // Remove gen state temporarily so we can pass &mut engine freely
        let mut gs = match engine.gen_states.remove(&key) {
            Some(gs) => gs,
            None => continue,
        };

        // Only process agents in sampling phase (prefill handled by tick_prefill)
        if !matches!(gs.phase, GenPhase::Sampling) {
            engine.gen_states.insert(key, gs);
            continue;
        }

        // Sample token (grammar masking applied dynamically if agent has grammar)
        let token = sample_gen_token(
            shared, engine, agent_id, &logits, &gs.context, &mut gs.sampler,
        );

        // Advance grammar state if agent has one
        if let Some(agent) = engine.agents.iter_mut().find(|a| a.id.0 == key) {
            if let Some(ref mut grammar) = agent.grammar {
                if let Some(tok_data) = shared.tokenizer.as_ref().and_then(|t| t.token_data(token)) {
                    let _ = grammar.accept_token(tok_data);
                }
                if grammar.is_complete() {
                    let _ = gs.token_tx.send(token);
                    gs.output.push(token);
                    if let Some(reply) = gs.reply {
                        let _ = reply.send(Ok(gs.output));
                    }
                    continue; // gs consumed, don't re-insert
                }
            }
        }

        let is_eos = Some(token) == eos_id;
        let is_done = gs.remaining == 0 || is_eos;

        if !is_eos {
            if gs.token_tx.send(token).is_err() {
                // Receiver dropped — implicit cancellation
                if let Some(reply) = gs.reply {
                    let _ = reply.send(Err(InferenceError::Agent(
                        "generation cancelled (stream dropped)".into(),
                    )));
                }
                continue; // gs consumed, don't re-insert
            }
            gs.output.push(token);
            gs.context.push(token);
        }

        if is_done {
            if let Some(reply) = gs.reply {
                let _ = reply.send(Ok(gs.output));
            }
            // gs consumed, don't re-insert
        } else {
            gs.remaining -= 1;
            if let Some(a) = engine.agents.iter_mut().find(|a| a.id.0 == key) {
                a.pending_token = Some(token);
            }
            // Re-insert for next tick
            engine.gen_states.insert(key, gs);
        }
    }
}

fn abort_generators(engine: &mut EngineMut, error: InferenceError) {
    let keys: Vec<u32> = engine.gen_states.keys().copied().collect();
    for key in keys {
        if let Some(gs) = engine.gen_states.remove(&key) {
            if let Some(reply) = gs.reply {
                let _ = reply.send(Err(InferenceError::Agent(format!(
                    "forward pass failed: {}",
                    error
                ))));
            }
        }
    }
}

// ── InferenceEngine (single-threaded convenience wrapper) ─────────

/// Top-level inference engine.
///
/// Owns the model, backend, KV arena, and agent contexts.
/// Provides the public API: load → create_agent → decode.
///
/// For multi-threaded use, see [`EngineHandle`].
pub struct InferenceEngine {
    shared: EngineShared,
    mutable: EngineMut,
}

impl InferenceEngine {
    /// Load a GGUF model and initialize the inference engine.
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let (shared, mutable) = load_engine(path, None)?;
        Ok(Self { shared, mutable })
    }

    /// Load a model with a specific KV cache capacity.
    pub fn load_with_capacity(
        path: &Path,
        kv_capacity: usize,
    ) -> Result<Self, InferenceError> {
        let (shared, mutable) = load_engine(path, Some(kv_capacity))?;
        Ok(Self { shared, mutable })
    }

    /// Convert into a thread-safe engine owner with a dedicated engine thread.
    pub fn into_handle(self) -> EngineOwner {
        EngineOwner::from_parts(self.shared, self.mutable)
    }

    pub fn create_agent(&mut self) -> AgentId {
        let id = AgentId(self.mutable.next_agent_id);
        self.mutable.next_agent_id += 1;
        self.mutable.agents.push(AgentContext::new(id));
        id
    }

    pub fn set_eviction(&mut self, config: EvictionConfig) {
        self.mutable.eviction = Some(config);
    }

    pub fn create_shared_region(
        &mut self,
        tokens: &[i32],
    ) -> Result<SharedRegionId, InferenceError> {
        create_shared_region_impl(&self.shared, &mut self.mutable, tokens)
    }

    pub fn attach_shared_region(
        &mut self,
        agent_id: AgentId,
        region: SharedRegionId,
    ) -> Result<(), InferenceError> {
        attach_shared_region_impl(&mut self.mutable, agent_id, region)
    }

    pub fn detach_shared_region(
        &mut self,
        agent_id: AgentId,
        region: SharedRegionId,
    ) -> Result<(), InferenceError> {
        detach_shared_region_impl(&mut self.mutable, agent_id, region)
    }

    pub fn decode(
        &mut self,
        agent_id: AgentId,
        token_id: i32,
    ) -> Result<Vec<f32>, InferenceError> {
        decode_impl(&self.shared, &mut self.mutable, agent_id, token_id)
    }

    pub fn decode_batch(
        &mut self,
        agent_id: AgentId,
        token_ids: &[i32],
    ) -> Result<Vec<f32>, InferenceError> {
        decode_batch_impl(&self.shared, &mut self.mutable, agent_id, token_ids)
    }

    pub fn destroy_agent(&mut self, agent_id: AgentId) {
        destroy_agent_impl(&mut self.mutable, agent_id);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.shared.config
    }

    pub fn shared(&self) -> &EngineShared {
        &self.shared
    }

    pub fn backend_name(&self) -> &str {
        self.mutable.backend.name()
    }

    pub fn free_kv_slots(&self) -> usize {
        self.mutable.arena.free_slots()
    }

    pub fn kv_capacity(&self) -> usize {
        self.mutable.arena.capacity()
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.shared.tokenizer.as_ref()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, InferenceError> {
        encode_impl(&self.shared, text)
    }

    pub fn decode_tokens(&self, ids: &[i32]) -> Result<String, InferenceError> {
        decode_tokens_impl(&self.shared, ids)
    }

    pub fn generate(
        &mut self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        generate_impl(&self.shared, &mut self.mutable, agent_id, prompt_tokens, sampler, max_tokens)
    }

    pub fn generate_text(
        &mut self,
        agent_id: AgentId,
        prompt: &str,
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<String, InferenceError> {
        let prompt_tokens = self.encode(prompt)?;
        let output_tokens = self.generate(agent_id, &prompt_tokens, sampler, max_tokens)?;
        self.decode_tokens(&output_tokens)
    }

    pub fn set_grammar(
        &mut self,
        agent_id: AgentId,
        gbnf: &str,
    ) -> Result<(), InferenceError> {
        set_grammar_impl(&mut self.mutable, agent_id, gbnf)
    }

    pub fn clear_grammar(
        &mut self,
        agent_id: AgentId,
    ) -> Result<(), InferenceError> {
        clear_grammar_impl(&mut self.mutable, agent_id)
    }

    pub fn generate_constrained(
        &mut self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        generate_constrained_impl(
            &self.shared, &mut self.mutable, agent_id, prompt_tokens, sampler, max_tokens,
        )
    }

    pub fn set_scheduler(&mut self, scheduler: impl Scheduler + 'static) {
        self.mutable.scheduler = Some(Box::new(scheduler));
    }

    pub fn set_priority(
        &mut self,
        agent_id: AgentId,
        priority: i32,
    ) -> Result<(), InferenceError> {
        set_priority_impl(&mut self.mutable, agent_id, priority)
    }

    pub fn enqueue_token(
        &mut self,
        agent_id: AgentId,
        token: i32,
    ) -> Result<(), InferenceError> {
        enqueue_token_impl(&mut self.mutable, agent_id, token)
    }

    pub fn step(&mut self) -> Result<Vec<(AgentId, Vec<f32>)>, InferenceError> {
        step_impl(&self.shared, &mut self.mutable)
    }
}

// ── EngineHandle (channel-based, clonable) ────────────────────────

/// Thread-safe, clonable inference engine client.
///
/// Communicates with a dedicated engine thread via channels.
/// Lock-free methods: `config()`, `tokenizer()`, `encode()`, `decode_tokens()`
/// Channel methods: everything else
#[derive(Clone)]
pub struct EngineHandle {
    shared: Arc<EngineShared>,
    cmd_tx: mpsc::Sender<Cmd>,
}

impl EngineHandle {
    // ── Lock-free reads ──

    pub fn config(&self) -> &ModelConfig {
        &self.shared.config
    }

    pub fn shared(&self) -> &EngineShared {
        &self.shared
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.shared.tokenizer.as_ref()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, InferenceError> {
        encode_impl(&self.shared, text)
    }

    pub fn decode_tokens(&self, ids: &[i32]) -> Result<String, InferenceError> {
        decode_tokens_impl(&self.shared, ids)
    }

    // ── Fire-and-forget ──

    pub fn destroy_agent(&self, agent_id: AgentId) {
        let _ = self.cmd_tx.send(Cmd::DestroyAgent(agent_id));
    }

    pub fn set_eviction(&self, config: EvictionConfig) {
        let _ = self.cmd_tx.send(Cmd::SetEviction(config));
    }

    pub fn enqueue_token(&self, agent_id: AgentId, token: i32) {
        let _ = self.cmd_tx.send(Cmd::EnqueueToken {
            agent: agent_id,
            token,
        });
    }

    pub fn set_priority(&self, agent_id: AgentId, priority: i32) {
        let _ = self.cmd_tx.send(Cmd::SetPriority {
            agent: agent_id,
            priority,
        });
    }

    pub fn set_scheduler(&self, scheduler: impl Scheduler + 'static) {
        let _ = self.cmd_tx.send(Cmd::SetScheduler(Box::new(scheduler)));
    }

    pub fn cancel_generate(&self, agent_id: AgentId) {
        let _ = self.cmd_tx.send(Cmd::CancelGenerate { agent: agent_id });
    }

    // ── With reply (send + recv) ──

    pub fn create_agent(&self) -> AgentId {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx.send(Cmd::CreateAgent(tx)).unwrap();
        rx.recv().unwrap()
    }

    pub fn create_shared_region(
        &self,
        tokens: &[i32],
    ) -> Result<SharedRegionId, InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::CreateSharedRegion {
                tokens: tokens.to_vec(),
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn attach_shared_region(
        &self,
        agent_id: AgentId,
        region: SharedRegionId,
    ) -> Result<(), InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::AttachShared {
                agent: agent_id,
                region,
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn detach_shared_region(
        &self,
        agent_id: AgentId,
        region: SharedRegionId,
    ) -> Result<(), InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::DetachShared {
                agent: agent_id,
                region,
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn decode(
        &self,
        agent_id: AgentId,
        token_id: i32,
    ) -> Result<Vec<f32>, InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::Decode {
                agent: agent_id,
                token: token_id,
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn decode_batch(
        &self,
        agent_id: AgentId,
        token_ids: &[i32],
    ) -> Result<Vec<f32>, InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::DecodeBatch {
                agent: agent_id,
                tokens: token_ids.to_vec(),
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    /// Start streaming generation. Returns a [`GenerateStream`] that yields
    /// tokens as they're produced.
    ///
    /// If the agent has a grammar attached (via [`set_grammar`]), output tokens
    /// are grammar-constrained automatically. No separate "constrained" flag —
    /// grammar presence on the agent is the single source of truth.
    pub fn generate(
        &self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        params: SamplerParams,
        max_tokens: usize,
    ) -> GenerateStream {
        let (token_tx, token_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::Generate {
                agent: agent_id,
                prompt: prompt_tokens.to_vec(),
                params,
                max_tokens,
                token_tx,
                reply: result_tx,
            })
            .unwrap();
        GenerateStream { token_rx, result_rx }
    }

    /// Generate all tokens and collect them.
    pub fn generate_all(
        &self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        params: SamplerParams,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        self.generate(agent_id, prompt_tokens, params, max_tokens)
            .finish()
    }

    /// Generate text from a string prompt.
    pub fn generate_text(
        &self,
        agent_id: AgentId,
        prompt: &str,
        params: SamplerParams,
        max_tokens: usize,
    ) -> Result<String, InferenceError> {
        let prompt_tokens = self.encode(prompt)?;
        let output = self.generate_all(agent_id, &prompt_tokens, params, max_tokens)?;
        self.decode_tokens(&output)
    }

    pub fn set_grammar(
        &self,
        agent_id: AgentId,
        gbnf: &str,
    ) -> Result<(), InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::SetGrammar {
                agent: agent_id,
                gbnf: gbnf.to_owned(),
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn clear_grammar(
        &self,
        agent_id: AgentId,
    ) -> Result<(), InferenceError> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(Cmd::ClearGrammar {
                agent: agent_id,
                reply: tx,
            })
            .unwrap();
        rx.recv().unwrap()
    }

    pub fn backend_name(&self) -> String {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx.send(Cmd::BackendName(tx)).unwrap();
        rx.recv().unwrap()
    }

    pub fn free_kv_slots(&self) -> usize {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx.send(Cmd::FreeKvSlots(tx)).unwrap();
        rx.recv().unwrap()
    }

    pub fn kv_capacity(&self) -> usize {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx.send(Cmd::KvCapacity(tx)).unwrap();
        rx.recv().unwrap()
    }
}

// ── GenerateStream ────────────────────────────────────────────────

/// A stream of tokens produced during generation.
///
/// Tokens are available as they're produced by the engine thread.
/// Call [`next_token`] to receive tokens one at a time, or [`tokens`]
/// for an iterator. Call [`finish`] to consume the stream and get
/// the final result.
pub struct GenerateStream {
    token_rx: mpsc::Receiver<i32>,
    result_rx: mpsc::Receiver<Result<Vec<i32>, InferenceError>>,
}

impl GenerateStream {
    /// Receive the next generated token. Returns `None` when generation
    /// is complete (EOS, max tokens, or error).
    pub fn next_token(&self) -> Option<i32> {
        self.token_rx.recv().ok()
    }

    /// Consume the stream and return the final result.
    pub fn finish(self) -> Result<Vec<i32>, InferenceError> {
        // Drop token_rx to unblock engine if it's sending tokens
        drop(self.token_rx);
        self.result_rx.recv().unwrap()
    }

    /// Convert into an iterator over generated tokens.
    pub fn tokens(self) -> impl Iterator<Item = i32> {
        self.token_rx.into_iter()
    }
}

// ── EngineOwner ───────────────────────────────────────────────────

/// Owns the engine thread and provides access via [`EngineHandle`].
///
/// Dropping the owner sends `Shutdown` and joins the engine thread.
/// Clone the handle via [`handle()`] to share across threads.
pub struct EngineOwner {
    handle: EngineHandle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl EngineOwner {
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let (shared, mutable) = load_engine(path, None)?;
        Ok(Self::from_parts(shared, mutable))
    }

    pub fn load_with_capacity(
        path: &Path,
        kv_capacity: usize,
    ) -> Result<Self, InferenceError> {
        let (shared, mutable) = load_engine(path, Some(kv_capacity))?;
        Ok(Self::from_parts(shared, mutable))
    }

    fn from_parts(shared: EngineShared, mutable: EngineMut) -> Self {
        let shared = Arc::new(shared);
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let shared_clone = Arc::clone(&shared);
        let thread = std::thread::spawn(move || {
            engine_loop(shared_clone, mutable, cmd_rx);
        });
        let handle = EngineHandle { shared, cmd_tx };
        Self {
            handle,
            thread: Some(thread),
        }
    }

    /// Get a clonable handle to the engine.
    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    /// Shut down the engine thread and wait for it to finish.
    pub fn shutdown(mut self) {
        let _ = self.handle.cmd_tx.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for EngineOwner {
    fn drop(&mut self) {
        let _ = self.handle.cmd_tx.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl std::ops::Deref for EngineOwner {
    type Target = EngineHandle;
    fn deref(&self) -> &EngineHandle {
        &self.handle
    }
}

// ── Small shared helper functions ─────────────────────────────────

fn load_engine(
    path: &Path,
    capacity_override: Option<usize>,
) -> Result<(EngineShared, EngineMut), InferenceError> {
    let reader = GgufReader::open(path)?;
    let config = ModelConfig::from_gguf_metadata(&reader.metadata)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&reader.metadata).ok();

    let backend = Backend::init_best()?;

    let capacity = capacity_override.unwrap_or(config.n_ctx_max);
    let arena_config = ArenaConfig {
        capacity,
        n_layer: config.n_layer,
        head_dim: config.head_dim,
        n_kv_head: config.n_kv_head,
        dtype: QuantType::F16,
    };

    let arena = KvArena::with_backend(&arena_config, &backend)?;

    let shared = EngineShared {
        reader,
        config,
        tokenizer,
    };
    let mutable = EngineMut {
        backend,
        arena,
        agents: Vec::new(),
        next_agent_id: 0,
        eviction: None,
        arena_dirty: false,
        scheduler: None,
        gen_states: HashMap::new(),
    };

    Ok((shared, mutable))
}

fn encode_impl(shared: &EngineShared, text: &str) -> Result<Vec<i32>, InferenceError> {
    shared
        .tokenizer
        .as_ref()
        .map(|t| t.encode(text))
        .ok_or_else(|| InferenceError::Tokenize("no tokenizer loaded".into()))
}

fn decode_tokens_impl(shared: &EngineShared, ids: &[i32]) -> Result<String, InferenceError> {
    shared
        .tokenizer
        .as_ref()
        .map(|t| t.decode(ids))
        .ok_or_else(|| InferenceError::Tokenize("no tokenizer loaded".into()))
}

fn destroy_agent_impl(m: &mut EngineMut, agent_id: AgentId) {
    if let Some(agent) = m.agents.iter().find(|a| a.id == agent_id) {
        let regions: Vec<SharedRegionId> = agent.shared_regions.clone();
        for region in regions {
            let _ = m.arena.unref_shared_region(region);
        }
    }

    let had_slots = !m.arena.agent_slots(agent_id).is_empty();
    m.arena.free_agent(agent_id);
    m.agents.retain(|a| a.id != agent_id);
    if had_slots {
        m.arena_dirty = true;
    }
}

fn attach_shared_region_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
    region: SharedRegionId,
) -> Result<(), InferenceError> {
    let region_len = m
        .arena
        .shared_region_len(region)
        .ok_or_else(|| {
            InferenceError::Arena(format!("shared region {:?} not found", region))
        })?;

    m.arena.ref_shared_region(region)?;

    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.attach_shared(region, region_len);
    Ok(())
}

fn detach_shared_region_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
    region: SharedRegionId,
) -> Result<(), InferenceError> {
    let region_len = m
        .arena
        .shared_region_len(region)
        .ok_or_else(|| {
            InferenceError::Arena(format!("shared region {:?} not found", region))
        })?;

    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.detach_shared(region, region_len);

    m.arena.unref_shared_region(region)?;
    m.arena_dirty = true;
    Ok(())
}

fn set_grammar_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
    gbnf: &str,
) -> Result<(), InferenceError> {
    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.grammar = Some(GrammarEngine::new(gbnf)?);
    Ok(())
}

fn clear_grammar_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
) -> Result<(), InferenceError> {
    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.grammar = None;
    Ok(())
}

fn set_priority_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
    priority: i32,
) -> Result<(), InferenceError> {
    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.priority = priority;
    Ok(())
}

// ── Compile-time thread safety assertions ─────────────────────────

fn _assert_thread_safety() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<EngineShared>();
    assert_send_sync::<EngineHandle>();
    assert_send::<EngineMut>();
    assert_send::<Cmd>();
}

fn enqueue_token_impl(
    m: &mut EngineMut,
    agent_id: AgentId,
    token: i32,
) -> Result<(), InferenceError> {
    let agent = m
        .agents
        .iter_mut()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| {
            InferenceError::Agent(format!("agent {:?} not found", agent_id))
        })?;
    agent.pending_token = Some(token);
    Ok(())
}
