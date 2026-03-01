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

/// Decode a batch of tokens for an agent, returning logits for the LAST token.
///
/// Processes `n_tokens` tokens in a single ggml graph with causal masking.
/// The new slots must already be allocated and contiguous in the arena.
///
/// - `seq_len_before`: number of existing KV entries for this agent BEFORE this batch
/// - `new_slots`: pre-allocated SlotIds for the batch (must be consecutive in arena)
/// - `kv_offset`: arena slot index where this agent's KV starts
pub fn decode_batch(
    agent: &mut AgentContext,
    token_ids: &[i32],
    new_slots: &[SlotId],
    arena: &mut KvArena,
    reader: &GgufReader,
    config: &ModelConfig,
    backend: &Backend,
    seq_len_before: usize,
    kv_offset: usize,
) -> Result<Vec<f32>, InferenceError> {
    let n_tokens = token_ids.len();
    assert_eq!(n_tokens, new_slots.len());

    let seq_len_before_i64 = seq_len_before as i64;
    let n_tokens_i64 = n_tokens as i64;

    // Graph context
    let graph_overhead = 256 * 1024 * 1024;
    let mut graph = ComputeGraph::new(graph_overhead, true)?;

    // Position tensor: n_tokens entries
    let pos_tensor = graph.new_tensor_1d(QuantType::I32, n_tokens_i64);
    graph.set_input(pos_tensor);

    // Causal prefill mask
    let (mask_tensor, mask_data) =
        forward::mask::causal_prefill_mask(&mut graph, seq_len_before_i64, n_tokens_i64);

    // Embedding lookup for n_tokens
    let tok_embd = reader.tensor_data("token_embd.weight")?;
    let (embedding, ids_tensor) =
        forward::embedding::embed_tokens(&mut graph, n_tokens_i64, tok_embd);
    let mut hidden = embedding;

    // Per-layer transformer blocks
    for l in 0..config.n_layer {
        let prefix = format!("blk.{l}");

        // Attention norm
        let attn_norm = reader.tensor_data(&format!("{prefix}.attn_norm.weight"))?;
        let normed = forward::norm::rms_norm(&mut graph, hidden, attn_norm, config.norm_eps);

        // Self-attention (batched)
        let attn_weights = forward::attention::AttentionWeights {
            wq: reader.tensor_data(&format!("{prefix}.attn_q.weight"))?,
            wk: reader.tensor_data(&format!("{prefix}.attn_k.weight"))?,
            wv: reader.tensor_data(&format!("{prefix}.attn_v.weight"))?,
            wo: reader.tensor_data(&format!("{prefix}.attn_output.weight"))?,
        };
        let attn_out = forward::attention::attention_layer_batched(
            &mut graph,
            normed,
            &attn_weights,
            arena,
            l,
            new_slots,
            pos_tensor,
            seq_len_before_i64,
            kv_offset,
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

    // Output projection → logits [n_vocab, n_tokens]
    let output_weight = reader.tensor_data("output.weight")?;
    let logits_all = forward::output::output_projection(&mut graph, hidden, output_weight);

    // Extract logits for the LAST token only
    let n_vocab = config.n_vocab as i64;
    let last_byte_offset = ((n_tokens_i64 - 1) * n_vocab) as usize * std::mem::size_of::<f32>();
    let logits_last = graph.view_1d(logits_all, n_vocab, last_byte_offset);

    graph.set_output(logits_last);
    graph.build_forward(logits_last);
    graph.alloc_graph(backend)?;

    // Set input data: token IDs
    unsafe {
        ffi::ggml_backend_tensor_set(
            ids_tensor.as_ptr(),
            token_ids.as_ptr() as *const std::os::raw::c_void,
            0,
            n_tokens * std::mem::size_of::<i32>(),
        );
    }

    // Set input data: positions (logical_pos for each token)
    let positions: Vec<i32> = (0..n_tokens)
        .map(|i| (seq_len_before + i) as i32)
        .collect();
    unsafe {
        ffi::ggml_backend_tensor_set(
            pos_tensor.as_ptr(),
            positions.as_ptr() as *const std::os::raw::c_void,
            0,
            n_tokens * std::mem::size_of::<i32>(),
        );
    }

    // Set input data: causal mask
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

    // Extract last-token logits
    let n_vocab = config.n_vocab;
    let mut logits_data = vec![0.0f32; n_vocab];
    unsafe {
        ffi::ggml_backend_tensor_get(
            logits_last.as_ptr(),
            logits_data.as_mut_ptr() as *mut std::os::raw::c_void,
            0,
            n_vocab * std::mem::size_of::<f32>(),
        );
    }

    // Update agent state
    for &slot in new_slots {
        agent.push_slot(slot);
    }
    agent.last_logits = logits_data.clone();

    Ok(logits_data)
}
