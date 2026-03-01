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

/// Prefill KV data for a shared region (no agent context).
///
/// Runs the same transformer forward pass as `decode_batch`, populating
/// K/V entries in the arena for the given slots. The output logits are
/// discarded — the purpose is only to populate the shared KV cache.
///
/// - `token_ids`: the shared prefix tokens
/// - `slots`: pre-allocated SlotIds for the shared region
/// - `kv_offset`: arena offset where the shared region starts (typically 0)
///
/// After this call, the slots contain valid K/V data and can be
/// attached to agents via `attach_shared_region`.
pub fn prefill_shared_region(
    token_ids: &[i32],
    slots: &[SlotId],
    arena: &mut KvArena,
    reader: &GgufReader,
    config: &ModelConfig,
    backend: &Backend,
    kv_offset: usize,
) -> Result<(), InferenceError> {
    let n_tokens = token_ids.len();
    assert_eq!(n_tokens, slots.len());

    if n_tokens == 0 {
        return Ok(());
    }

    let n_tokens_i64 = n_tokens as i64;
    let seq_len_before: i64 = 0; // shared region is always a fresh prefill

    // Graph context
    let graph_overhead = 256 * 1024 * 1024;
    let mut graph = ComputeGraph::new(graph_overhead, true)?;

    // Position tensor: n_tokens entries
    let pos_tensor = graph.new_tensor_1d(QuantType::I32, n_tokens_i64);
    graph.set_input(pos_tensor);

    // Causal prefill mask (standard lower-triangular)
    let (mask_tensor, mask_data) =
        forward::mask::causal_prefill_mask(&mut graph, seq_len_before, n_tokens_i64);

    // Embedding lookup
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

        // Self-attention (batched prefill — same as decode_batch)
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
            slots,
            pos_tensor,
            seq_len_before,
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

    // We need a valid output to build the graph, but we discard logits.
    // Use the hidden state as the output (cheapest option — no output projection).
    graph.set_output(hidden);
    graph.build_forward(hidden);
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

    // Set input data: positions (0, 1, 2, ..., n_tokens-1)
    let positions: Vec<i32> = (0..n_tokens as i32).collect();
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

    // Execute — populates K/V in arena, discards output
    graph.execute(backend)?;

    Ok(())
}
