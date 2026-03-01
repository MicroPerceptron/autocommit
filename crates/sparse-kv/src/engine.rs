use std::path::Path;

use crate::arena::kv_arena::{ArenaConfig, KvArena};
use crate::arena::slot::AgentId;
use crate::context::agent::AgentContext;
use crate::context::batch;
use crate::context::decode;
use crate::context::scheduler::{Scheduler, SchedulerEntry};
use crate::error::InferenceError;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::grammar::GrammarEngine;
use crate::sampling::Sampler;
use crate::sparse::eviction::EvictionConfig;
use crate::tokenizer::Tokenizer;

/// Top-level inference engine.
///
/// Owns the model, backend, KV arena, and agent contexts.
/// Provides the public API: load → create_agent → decode.
pub struct InferenceEngine {
    backend: Backend,
    reader: GgufReader,
    config: ModelConfig,
    arena: KvArena,
    agents: Vec<AgentContext>,
    next_agent_id: u32,
    tokenizer: Option<Tokenizer>,
    eviction: Option<EvictionConfig>,
    /// Set when the arena has gaps from destroy_agent or eviction.
    /// When true, even single-agent decode requires compaction.
    /// Cleared after compact_for_agent.
    arena_dirty: bool,
    /// Optional scheduler for multi-agent step() loop.
    scheduler: Option<Box<dyn Scheduler>>,
}

impl InferenceEngine {
    /// Load a GGUF model and initialize the inference engine.
    ///
    /// Uses the best available backend (Metal on macOS, CPU fallback).
    /// KV cache capacity defaults to the model's max context length.
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let reader = GgufReader::open(path)?;
        let config = ModelConfig::from_gguf_metadata(&reader.metadata)?;
        let tokenizer = Tokenizer::from_gguf_metadata(&reader.metadata).ok();

        let backend = Backend::init_best()?;

        let arena_config = ArenaConfig {
            capacity: config.n_ctx_max,
            n_layer: config.n_layer,
            head_dim: config.head_dim,
            n_kv_head: config.n_kv_head,
            dtype: QuantType::F16,
        };

        let arena = KvArena::with_backend(&arena_config, &backend)?;

        Ok(Self {
            backend,
            reader,
            config,
            arena,
            agents: Vec::new(),
            next_agent_id: 0,
            tokenizer,
            eviction: None,
            arena_dirty: false,
            scheduler: None,
        })
    }

    /// Load a model with a specific KV cache capacity.
    pub fn load_with_capacity(
        path: &Path,
        kv_capacity: usize,
    ) -> Result<Self, InferenceError> {
        let reader = GgufReader::open(path)?;
        let config = ModelConfig::from_gguf_metadata(&reader.metadata)?;
        let tokenizer = Tokenizer::from_gguf_metadata(&reader.metadata).ok();

        let backend = Backend::init_best()?;

        let arena_config = ArenaConfig {
            capacity: kv_capacity,
            n_layer: config.n_layer,
            head_dim: config.head_dim,
            n_kv_head: config.n_kv_head,
            dtype: QuantType::F16,
        };

        let arena = KvArena::with_backend(&arena_config, &backend)?;

        Ok(Self {
            backend,
            reader,
            config,
            arena,
            agents: Vec::new(),
            next_agent_id: 0,
            tokenizer,
            eviction: None,
            arena_dirty: false,
            scheduler: None,
        })
    }

    /// Create a new inference agent. Returns its ID.
    pub fn create_agent(&mut self) -> AgentId {
        let id = AgentId(self.next_agent_id);
        self.next_agent_id += 1;
        self.agents.push(AgentContext::new(id));
        id
    }

    /// Set the eviction configuration. Enables automatic KV cache eviction
    /// when the arena runs low on free slots.
    pub fn set_eviction(&mut self, config: EvictionConfig) {
        self.eviction = Some(config);
    }

    /// Run eviction if the arena is below the free-slot threshold.
    ///
    /// Marks slots for eviction and rebuilds agent slot lists, but does NOT
    /// compact the arena (no data movement). Sets `arena_dirty` so the next
    /// decode/decode_batch call handles compaction if needed.
    fn run_eviction(&mut self) {
        let (trigger_threshold, evict_fraction) = match &self.eviction {
            Some(e) => (e.trigger_threshold, e.evict_fraction),
            None => return,
        };

        let capacity = self.arena.capacity();
        let free_ratio = self.arena.free_slots() as f32 / capacity as f32;
        if free_ratio >= trigger_threshold {
            return;
        }

        let occupied = self.arena.occupied_count();
        let need = (evict_fraction * occupied as f32).ceil() as usize;
        if need == 0 {
            return;
        }

        let candidates = self.arena.evictable_slots();
        let to_evict = self.eviction.as_ref().unwrap().policy.select(&candidates, need);

        self.arena.mark_evict(&to_evict);
        self.arena_dirty = true;

        // Rebuild agent slot lists to remove evicted entries
        for agent in &mut self.agents {
            agent.rebuild_slots(&self.arena);
        }
    }

    /// Whether the arena needs compaction before a forward pass.
    ///
    /// True when multiple agents share the arena (slots may be interleaved)
    /// or when eviction/destroy created gaps (arena_dirty).
    fn needs_compact(&self) -> bool {
        self.agents.len() > 1 || self.arena_dirty
    }

    /// Decode a single token for an agent, returning logits [n_vocab].
    ///
    /// Automatically handles eviction (if configured), compaction,
    /// and multi-agent KV isolation.
    ///
    /// **P1 compliance**: In single-agent mode with a clean arena, no data
    /// movement occurs. Compaction is only performed when multiple agents
    /// share the arena or after eviction/destroy creates gaps.
    pub fn decode(
        &mut self,
        agent_id: AgentId,
        token_id: i32,
    ) -> Result<Vec<f32>, InferenceError> {
        self.run_eviction();

        // Pre-allocate the slot BEFORE any compaction
        let seq_len_before = self
            .agents
            .iter()
            .find(|a| a.id == agent_id)
            .map(|a| a.seq_len)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;

        let logical_pos = seq_len_before as u32;
        let new_slot = self.arena.alloc_slot(agent_id, logical_pos)?;

        let (kv_offset, final_slot) = if self.needs_compact() {
            // Multi-agent or dirty arena: compact so target agent is contiguous
            let (offset, _) = self.arena.compact_for_agent(agent_id);
            self.arena_dirty = false;
            for agent in &mut self.agents {
                agent.rebuild_slots(&self.arena);
            }
            // After rebuild, the new slot is the last in the agent's list
            let agent = self
                .agents
                .iter()
                .find(|a| a.id == agent_id)
                .unwrap();
            (offset, *agent.slots.last().unwrap())
        } else {
            // Single agent, clean arena: no data movement needed
            let agent = self
                .agents
                .iter_mut()
                .find(|a| a.id == agent_id)
                .unwrap();
            agent.push_slot(new_slot);
            (0, new_slot)
        };

        let agent = self
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .unwrap();

        decode::decode_one(
            agent,
            token_id,
            &mut self.arena,
            &self.reader,
            &self.config,
            &self.backend,
            kv_offset,
            final_slot,
        )
    }

    /// Decode a batch of tokens for an agent, returning logits for the last token.
    ///
    /// Processes all tokens in a single ggml graph with causal masking.
    /// More efficient than calling `decode()` in a loop for prefill.
    ///
    /// **P1 compliance**: Skips compaction in single-agent mode with clean arena.
    pub fn decode_batch(
        &mut self,
        agent_id: AgentId,
        token_ids: &[i32],
    ) -> Result<Vec<f32>, InferenceError> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        self.run_eviction();

        let seq_len_before = self
            .agents
            .iter()
            .find(|a| a.id == agent_id)
            .map(|a| a.seq_len)
            .ok_or_else(|| InferenceError::Agent(format!("agent {:?} not found", agent_id)))?;

        // Pre-allocate slots for all tokens
        let n_tokens = token_ids.len();
        let mut new_slots = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let logical_pos = (seq_len_before + i) as u32;
            new_slots.push(self.arena.alloc_slot(agent_id, logical_pos)?);
        }

        let (kv_offset, updated_slots) = if self.needs_compact() {
            // Compact: target agent at front, new slots contiguous
            let (offset, _) = self.arena.compact_for_agent(agent_id);
            self.arena_dirty = false;
            for agent in &mut self.agents {
                agent.rebuild_slots(&self.arena);
            }
            // After rebuild, new slots are the last n_tokens of agent's list
            let agent = self
                .agents
                .iter()
                .find(|a| a.id == agent_id)
                .unwrap();
            let slots = agent.slots[seq_len_before..].to_vec();

            // Reset agent state — batch::decode_batch will re-push
            let agent = self
                .agents
                .iter_mut()
                .find(|a| a.id == agent_id)
                .unwrap();
            agent.slots.truncate(seq_len_before);
            agent.seq_len = seq_len_before;

            (offset, slots)
        } else {
            // Single agent, clean arena: alloc'd slots are already correct
            (0, new_slots)
        };

        let agent = self
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .unwrap();

        batch::decode_batch(
            agent,
            token_ids,
            &updated_slots,
            &mut self.arena,
            &self.reader,
            &self.config,
            &self.backend,
            seq_len_before,
            kv_offset,
        )
    }

    /// Destroy an agent and free its KV cache slots.
    ///
    /// Sets `arena_dirty` if the agent had slots, since freeing creates
    /// gaps that require compaction before the next forward pass.
    pub fn destroy_agent(&mut self, agent_id: AgentId) {
        let had_slots = !self.arena.agent_slots(agent_id).is_empty();
        self.arena.free_agent(agent_id);
        self.agents.retain(|a| a.id != agent_id);
        if had_slots {
            self.arena_dirty = true;
        }
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the backend name (e.g. "CPU", "Metal").
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Number of free KV cache slots available.
    pub fn free_kv_slots(&self) -> usize {
        self.arena.free_slots()
    }

    /// Total KV cache capacity.
    pub fn kv_capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Get the tokenizer, if the GGUF file contained tokenizer metadata.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Encode text to tokens. Returns Err if no tokenizer is available.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, InferenceError> {
        self.tokenizer
            .as_ref()
            .map(|t| t.encode(text))
            .ok_or_else(|| InferenceError::Tokenize("no tokenizer loaded".into()))
    }

    /// Decode tokens to text. Returns Err if no tokenizer is available.
    pub fn decode_tokens(&self, ids: &[i32]) -> Result<String, InferenceError> {
        self.tokenizer
            .as_ref()
            .map(|t| t.decode(ids))
            .ok_or_else(|| InferenceError::Tokenize("no tokenizer loaded".into()))
    }

    /// Generate tokens autoregressively.
    ///
    /// Uses batched prefill for the prompt, then single-token decode for sampling.
    pub fn generate(
        &mut self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        // Batched prefill: process all prompt tokens in one graph
        let mut logits = if prompt_tokens.is_empty() {
            Vec::new()
        } else {
            self.decode_batch(agent_id, prompt_tokens)?
        };

        if logits.is_empty() {
            return Ok(Vec::new());
        }

        // Track context for repetition penalty
        let mut context: Vec<i32> = prompt_tokens.to_vec();
        let mut output = Vec::with_capacity(max_tokens);
        let eos_id = self.tokenizer().and_then(|t| t.eos_id());

        for _ in 0..max_tokens {
            // Repetition penalty window: last N tokens from context
            let window = sampler.params.repeat_window;
            let start = context.len().saturating_sub(window);
            let recent = &context[start..];

            let token = sampler.sample(&logits, recent);

            if Some(token) == eos_id {
                break;
            }

            output.push(token);
            context.push(token);

            logits = self.decode(agent_id, token)?;
        }

        Ok(output)
    }

    /// Generate text from a prompt string.
    ///
    /// Convenience wrapper: encode prompt → generate → decode output.
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

    /// Set a GBNF grammar constraint for an agent.
    ///
    /// All subsequent generation for this agent will be constrained to
    /// produce output matching the grammar. The grammar state is tracked
    /// per-agent and advances with each generated token.
    pub fn set_grammar(
        &mut self,
        agent_id: AgentId,
        gbnf: &str,
    ) -> Result<(), InferenceError> {
        let agent = self
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;
        agent.grammar = Some(GrammarEngine::new(gbnf)?);
        Ok(())
    }

    /// Clear grammar constraint for an agent.
    pub fn clear_grammar(&mut self, agent_id: AgentId) {
        if let Some(agent) = self.agents.iter_mut().find(|a| a.id == agent_id) {
            agent.grammar = None;
        }
    }

    /// Generate tokens with grammar-constrained sampling.
    ///
    /// Like `generate()`, but uses the agent's grammar (if set) to mask
    /// invalid tokens before sampling. The grammar state advances with
    /// each generated token.
    pub fn generate_constrained(
        &mut self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        // Batched prefill
        let mut logits = if prompt_tokens.is_empty() {
            Vec::new()
        } else {
            self.decode_batch(agent_id, prompt_tokens)?
        };

        if logits.is_empty() {
            return Ok(Vec::new());
        }

        let mut context: Vec<i32> = prompt_tokens.to_vec();
        let mut output = Vec::with_capacity(max_tokens);
        let eos_id = self.tokenizer().and_then(|t| t.eos_id());
        let n_vocab = logits.len();

        for _ in 0..max_tokens {
            let window = sampler.params.repeat_window;
            let start = context.len().saturating_sub(window);
            let recent = &context[start..];

            // Build grammar mask if grammar is set
            let agent = self.agents.iter_mut().find(|a| a.id == agent_id).unwrap();
            let token = if let Some(ref mut grammar) = agent.grammar {
                let trie = self.tokenizer.as_ref()
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

            // Advance grammar state
            let agent = self.agents.iter_mut().find(|a| a.id == agent_id).unwrap();
            if let Some(ref mut grammar) = agent.grammar {
                if let Some(tok_data) = self.tokenizer.as_ref().and_then(|t| t.token_data(token)) {
                    grammar.accept_token(tok_data)?;
                }
                if grammar.is_complete() {
                    output.push(token);
                    break;
                }
            }

            output.push(token);
            context.push(token);

            logits = self.decode(agent_id, token)?;
        }

        Ok(output)
    }

    // ── Scheduler integration (P6) ──────────────────────────────

    /// Set the scheduling policy for multi-agent step() loop.
    pub fn set_scheduler(&mut self, scheduler: impl Scheduler + 'static) {
        self.scheduler = Some(Box::new(scheduler));
    }

    /// Set scheduling priority for an agent. Higher = more important.
    pub fn set_priority(&mut self, agent_id: AgentId, priority: i32) {
        if let Some(agent) = self.agents.iter_mut().find(|a| a.id == agent_id) {
            agent.priority = priority;
        }
    }

    /// Enqueue a token for an agent. Consumed by the next `step()` call
    /// that selects this agent.
    pub fn enqueue_token(&mut self, agent_id: AgentId, token: i32) {
        if let Some(agent) = self.agents.iter_mut().find(|a| a.id == agent_id) {
            agent.pending_token = Some(token);
        }
    }

    /// Scheduler-driven decode step.
    ///
    /// Asks the scheduler which agent should run next, consumes that
    /// agent's pending token, and returns `(agent_id, logits)`.
    ///
    /// Returns `Ok(None)` if no agent has pending work or no scheduler is set.
    pub fn step(&mut self) -> Result<Option<(AgentId, Vec<f32>)>, InferenceError> {
        let scheduler = match &mut self.scheduler {
            Some(s) => s,
            None => return Ok(None),
        };

        let entries: Vec<SchedulerEntry> = self
            .agents
            .iter()
            .map(|a| SchedulerEntry {
                id: a.id,
                priority: a.priority,
                seq_len: a.seq_len,
                has_pending: a.pending_token.is_some(),
            })
            .collect();

        let agent_id = match scheduler.next(&entries) {
            Some(id) => id,
            None => return Ok(None),
        };

        let token = self
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

        let logits = self.decode(agent_id, token)?;
        Ok(Some((agent_id, logits)))
    }
}
