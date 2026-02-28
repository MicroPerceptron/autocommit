use std::path::Path;

use crate::arena::kv_arena::{ArenaConfig, KvArena};
use crate::arena::slot::AgentId;
use crate::context::agent::AgentContext;
use crate::context::decode;
use crate::error::InferenceError;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::grammar::GrammarEngine;
use crate::sampling::Sampler;
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
        })
    }

    /// Create a new inference agent. Returns its ID.
    pub fn create_agent(&mut self) -> AgentId {
        let id = AgentId(self.next_agent_id);
        self.next_agent_id += 1;
        self.agents.push(AgentContext::new(id));
        id
    }

    /// Decode a single token for an agent, returning logits [n_vocab].
    pub fn decode(
        &mut self,
        agent_id: AgentId,
        token_id: i32,
    ) -> Result<Vec<f32>, InferenceError> {
        let agent = self
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| {
                InferenceError::Agent(format!("agent {:?} not found", agent_id))
            })?;

        decode::decode_one(
            agent,
            token_id,
            &mut self.arena,
            &self.reader,
            &self.config,
            &self.backend,
        )
    }

    /// Destroy an agent and free its KV cache slots.
    pub fn destroy_agent(&mut self, agent_id: AgentId) {
        self.arena.free_agent(agent_id);
        self.agents.retain(|a| a.id != agent_id);
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
    /// Feeds `prompt_tokens` through the model one at a time (prefill),
    /// then samples up to `max_tokens` new tokens (or until EOS).
    pub fn generate(
        &mut self,
        agent_id: AgentId,
        prompt_tokens: &[i32],
        sampler: &mut Sampler,
        max_tokens: usize,
    ) -> Result<Vec<i32>, InferenceError> {
        // Prefill: feed all prompt tokens to build KV cache
        let mut logits = Vec::new();
        for &tok in prompt_tokens {
            logits = self.decode(agent_id, tok)?;
        }

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
        // Prefill
        let mut logits = Vec::new();
        for &tok in prompt_tokens {
            logits = self.decode(agent_id, tok)?;
        }

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
}
