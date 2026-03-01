use ggml_sys::ffi;

use crate::arena::kv_arena::KvArena;
use crate::arena::slot::SlotId;
use crate::context::agent::AgentContext;
use crate::error::InferenceError;
use crate::forward;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::tensor::graph::ComputeGraph;

/// Decode a single token for an agent, returning logits.
///
/// The caller is responsible for:
/// 1. Allocating the KV slot (`arena.alloc_slot`)
/// 2. Updating the agent's slot list and seq_len
/// 3. Compacting the arena if needed (multi-agent)
///
/// This function builds the ggml graph, executes it, and extracts logits.
pub fn decode_one(
    agent: &mut AgentContext,
    token_id: i32,
    arena: &mut KvArena,
    reader: &GgufReader,
    config: &ModelConfig,
    backend: &Backend,
    kv_offset: usize,
    new_slot: SlotId,
) -> Result<Vec<f32>, InferenceError> {
    // Agent state already updated by caller: seq_len includes the new token.
    // logical_pos = the new token's position in the sequence.
    let logical_pos = (agent.effective_seq_len() - 1) as u32;
    let seq_len = agent.effective_seq_len() as i64;

    // Graph context: enough memory for tensor metadata.
    // With no_alloc=true, the gallocr handles actual buffer allocation.
    let graph_overhead = 256 * 1024 * 1024; // 256 MB for graph metadata
    let mut graph = ComputeGraph::new(graph_overhead, true)?;

    // Create position tensor (shared across all layers)
    // Data will be set after graph allocation, before compute.
    let pos_tensor = graph.new_tensor_1d(QuantType::I32, 1);
    graph.set_input(pos_tensor);

    // Embedding lookup
    let tok_embd = reader.tensor_data("token_embd.weight")?;
    let (embedding, ids_tensor) = forward::embedding::embed_tokens(&mut graph, 1, tok_embd);
    let mut hidden = embedding;

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
            pos_tensor,
            seq_len,
            kv_offset,
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
    let logits = forward::output::output_projection(&mut graph, hidden, output_weight);

    // Mark logits as output so the allocator doesn't free it
    graph.set_output(logits);

    // Build the compute graph
    graph.build_forward(logits);

    // Allocate tensors on the backend (no execution yet)
    graph.alloc_graph(backend)?;

    // Set input data AFTER allocation, BEFORE execution.
    // Token IDs for embedding lookup.
    unsafe {
        ffi::ggml_backend_tensor_set(
            ids_tensor.as_ptr(),
            &token_id as *const i32 as *const std::os::raw::c_void,
            0,
            std::mem::size_of::<i32>(),
        );
    }
    // The position tensor needs the current logical position.
    unsafe {
        ffi::ggml_backend_tensor_set(
            pos_tensor.as_ptr(),
            &logical_pos as *const u32 as *const std::os::raw::c_void,
            0,
            std::mem::size_of::<i32>(),
        );
    }

    // Execute the graph
    graph.execute(backend)?;

    // Extract logits from the output tensor
    let n_vocab = config.n_vocab;
    let mut logits_data = vec![0.0f32; n_vocab];
    unsafe {
        ffi::ggml_backend_tensor_get(
            logits.as_ptr(),
            logits_data.as_mut_ptr() as *mut std::os::raw::c_void,
            0,
            n_vocab * std::mem::size_of::<f32>(),
        );
    }

    agent.last_logits = logits_data.clone();

    Ok(logits_data)
}
