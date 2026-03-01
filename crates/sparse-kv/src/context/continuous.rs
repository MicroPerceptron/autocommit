use ggml_sys::ffi;

use crate::arena::kv_arena::KvArena;
use crate::arena::slot::SlotId;
use crate::error::InferenceError;
use crate::forward;
use crate::gguf::GgufReader;
use crate::model::config::ModelConfig;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::tensor::graph::ComputeGraph;

/// Decode one token per agent in a single ggml graph (continuous batching).
///
/// All agents share the same forward pass — FFN/MLP compute is amortized
/// across agents. A block-diagonal attention mask ensures each agent only
/// attends to its own KV context.
///
/// - `token_ids`: one token per agent, in batch order
/// - `positions`: logical position for each agent's token (typically agent.seq_len)
/// - `new_slots`: pre-allocated SlotId per agent (potentially scattered)
/// - `ranges`: `(kv_offset, kv_len)` per agent from `compact_for_batch`
/// - `total_kv_len`: total occupied KV entries visible in the arena
///
/// Returns `Vec<Vec<f32>>` — logits for each agent in input order.
///
/// The caller is responsible for updating agent state (push_slot, last_logits)
/// after this function returns.
pub fn decode_continuous_batch(
    token_ids: &[i32],
    positions: &[i32],
    new_slots: &[SlotId],
    ranges: &[(usize, usize)],
    arena: &mut KvArena,
    reader: &GgufReader,
    config: &ModelConfig,
    backend: &Backend,
    total_kv_len: usize,
    shared_len: usize,
) -> Result<Vec<Vec<f32>>, InferenceError> {
    let n_agents = token_ids.len();
    assert_eq!(n_agents, positions.len());
    assert_eq!(n_agents, new_slots.len());
    assert_eq!(n_agents, ranges.len());

    let n_agents_i64 = n_agents as i64;
    let total_kv_len_i64 = total_kv_len as i64;

    // Graph context
    let graph_overhead = 256 * 1024 * 1024;
    let mut graph = ComputeGraph::new(graph_overhead, true)?;

    // Position tensor: heterogeneous positions (each agent at its own seq_len)
    let pos_tensor = graph.new_tensor_1d(QuantType::I32, n_agents_i64);
    graph.set_input(pos_tensor);

    // Block-diagonal mask for agent isolation (shared-aware when shared prefix exists)
    let (mask_tensor, mask_data) = if shared_len > 0 {
        forward::mask::block_diagonal_mask_with_shared(
            &mut graph,
            shared_len,
            ranges,
            total_kv_len_i64,
        )
    } else {
        forward::mask::block_diagonal_mask(&mut graph, ranges, total_kv_len_i64)
    };

    // Embedding lookup for n_agents tokens
    let tok_embd = reader.tensor_data("token_embd.weight")?;
    let (embedding, ids_tensor) =
        forward::embedding::embed_tokens(&mut graph, n_agents_i64, tok_embd);
    let mut hidden = embedding;

    // Per-layer transformer blocks
    for l in 0..config.n_layer {
        let prefix = format!("blk.{l}");

        // Attention norm
        let attn_norm = reader.tensor_data(&format!("{prefix}.attn_norm.weight"))?;
        let normed = forward::norm::rms_norm(&mut graph, hidden, attn_norm, config.norm_eps);

        // Self-attention (continuous batching)
        let attn_weights = forward::attention::AttentionWeights {
            wq: reader.tensor_data(&format!("{prefix}.attn_q.weight"))?,
            wk: reader.tensor_data(&format!("{prefix}.attn_k.weight"))?,
            wv: reader.tensor_data(&format!("{prefix}.attn_v.weight"))?,
            wo: reader.tensor_data(&format!("{prefix}.attn_output.weight"))?,
        };
        let attn_out = forward::attention::attention_layer_continuous(
            &mut graph,
            normed,
            &attn_weights,
            arena,
            l,
            new_slots,
            pos_tensor,
            total_kv_len_i64,
            mask_tensor,
            config,
        );

        hidden = graph.add(hidden, attn_out);

        // FFN norm
        let ffn_norm = reader.tensor_data(&format!("{prefix}.ffn_norm.weight"))?;
        let normed = forward::norm::rms_norm(&mut graph, hidden, ffn_norm, config.norm_eps);

        // Feed-forward
        let w_gate = reader.tensor_data(&format!("{prefix}.ffn_gate.weight"))?;
        let w_up = reader.tensor_data(&format!("{prefix}.ffn_up.weight"))?;
        let w_down = reader.tensor_data(&format!("{prefix}.ffn_down.weight"))?;
        let ffn_out = forward::ffn::ffn_silu(&mut graph, normed, w_gate, w_up, w_down);

        hidden = graph.add(hidden, ffn_out);
    }

    // Output norm
    let output_norm = reader.tensor_data("output_norm.weight")?;
    let hidden = forward::norm::rms_norm(&mut graph, hidden, output_norm, config.norm_eps);

    // Output projection → logits [n_vocab, n_agents]
    let output_weight = reader.tensor_data("output.weight")?;
    let logits_all = forward::output::output_projection(&mut graph, hidden, output_weight);

    graph.set_output(logits_all);
    graph.build_forward(logits_all);
    graph.alloc_graph(backend)?;

    // Set input data: token IDs
    unsafe {
        ffi::ggml_backend_tensor_set(
            ids_tensor.as_ptr(),
            token_ids.as_ptr() as *const std::os::raw::c_void,
            0,
            n_agents * std::mem::size_of::<i32>(),
        );
    }

    // Set input data: heterogeneous positions
    unsafe {
        ffi::ggml_backend_tensor_set(
            pos_tensor.as_ptr(),
            positions.as_ptr() as *const std::os::raw::c_void,
            0,
            n_agents * std::mem::size_of::<i32>(),
        );
    }

    // Set input data: block-diagonal mask
    unsafe {
        ffi::ggml_backend_tensor_set(
            mask_tensor.as_ptr(),
            mask_data.as_ptr() as *const std::os::raw::c_void,
            0,
            mask_data.len() * std::mem::size_of::<f32>(),
        );
    }

    // Execute
    graph.execute(backend)?;

    // Extract ALL agents' logits from [n_vocab, n_agents]
    let n_vocab = config.n_vocab;
    let total_logits = n_vocab * n_agents;
    let mut all_logits = vec![0.0f32; total_logits];
    unsafe {
        ffi::ggml_backend_tensor_get(
            logits_all.as_ptr(),
            all_logits.as_mut_ptr() as *mut std::os::raw::c_void,
            0,
            total_logits * std::mem::size_of::<f32>(),
        );
    }

    // Split into per-agent logit vectors
    let mut result = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        let start = i * n_vocab;
        result.push(all_logits[start..start + n_vocab].to_vec());
    }

    Ok(result)
}
