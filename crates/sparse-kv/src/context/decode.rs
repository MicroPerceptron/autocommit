use crate::arena::kv_arena::KvArena;
use crate::context::agent::AgentContext;
use crate::error::InferenceError;
use crate::forward;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::tensor::graph::ComputeGraph;

/// Decode a single token for an agent, returning logits.
///
/// This is the core inference loop:
/// 1. Allocate a KV slot for the new token
/// 2. Build a ggml compute graph for the full forward pass
/// 3. Execute the graph
/// 4. Extract logits
pub fn decode_one(
    agent: &mut AgentContext,
    token_id: i32,
    arena: &mut KvArena,
    reader: &GgufReader,
    config: &ModelConfig,
) -> Result<Vec<f32>, InferenceError> {
    // Allocate a slot for this token position
    let logical_pos = agent.seq_len as u32;
    let new_slot = arena.alloc_slot(agent.id, logical_pos)?;
    agent.push_slot(new_slot);

    // Build the position map for this agent
    let position_map = arena.agent_position_map(agent.id);

    // Estimate memory needed for the compute graph
    // Generous allocation: ~256MB should cover most models
    let mem_size = 256 * 1024 * 1024;
    let mut graph = ComputeGraph::new(mem_size)?;

    // Embedding lookup
    let tok_embd = reader.tensor_data("token_embd.weight")?;
    let mut hidden = forward::embedding::embed_tokens(&mut graph, &[token_id], tok_embd);

    // Per-layer transformer blocks
    for l in 0..config.n_layer {
        let prefix = format!("blk.{l}");

        // Attention norm
        let attn_norm = reader.tensor_data(&format!("{prefix}.attn_norm.weight"))?;
        let normed = forward::norm::rms_norm(&mut graph, hidden, attn_norm, config.norm_eps);

        // Self-attention
        let attn_weights = forward::attention::AttentionWeights {
            wq: reader.tensor_data(&format!("{prefix}.attn_q.weight"))?,
            wk: reader.tensor_data(&format!("{prefix}.attn_k.weight"))?,
            wv: reader.tensor_data(&format!("{prefix}.attn_v.weight"))?,
            wo: reader.tensor_data(&format!("{prefix}.attn_output.weight"))?,
        };
        let attn_out = forward::attention::attention_layer(
            &mut graph,
            normed,
            &attn_weights,
            arena,
            l,
            new_slot,
            &position_map,
            config,
        );

        // Residual connection
        hidden = graph.add(hidden, attn_out);

        // FFN norm
        let ffn_norm = reader.tensor_data(&format!("{prefix}.ffn_norm.weight"))?;
        let normed = forward::norm::rms_norm(&mut graph, hidden, ffn_norm, config.norm_eps);

        // Feed-forward
        let w_gate = reader.tensor_data(&format!("{prefix}.ffn_gate.weight"))?;
        let w_up = reader.tensor_data(&format!("{prefix}.ffn_up.weight"))?;
        let w_down = reader.tensor_data(&format!("{prefix}.ffn_down.weight"))?;
        let ffn_out = forward::ffn::ffn_silu(&mut graph, normed, w_gate, w_up, w_down);

        // Residual connection
        hidden = graph.add(hidden, ffn_out);
    }

    // Output norm
    let output_norm = reader.tensor_data("output_norm.weight")?;
    let hidden = forward::norm::rms_norm(&mut graph, hidden, output_norm, config.norm_eps);

    // Output projection -> logits
    let output_weight = reader.tensor_data("output.weight")?;
    let _logits = forward::output::output_projection(&mut graph, hidden, output_weight);

    // TODO: Execute the compute graph via ggml_backend_graph_compute
    // and extract logits from the output tensor.
    //
    // For now, return empty logits — the graph is built but not yet
    // executed. Full execution requires:
    // 1. ggml_backend initialization (CPU or Metal)
    // 2. ggml_backend_graph_plan + ggml_backend_graph_compute
    // 3. Reading the output tensor's data buffer

    let logits = vec![0.0f32; config.n_vocab];
    agent.last_logits = logits.clone();

    Ok(logits)
}
