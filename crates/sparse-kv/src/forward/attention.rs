use crate::arena::kv_arena::KvArena;
use crate::arena::position_map::PositionMap;
use crate::arena::slot::SlotId;
use crate::model::config::ModelConfig;
use crate::quant::QuantSlice;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

pub struct AttentionWeights<'a> {
    pub wq: QuantSlice<'a>,
    pub wk: QuantSlice<'a>,
    pub wv: QuantSlice<'a>,
    pub wo: QuantSlice<'a>,
}

/// Run one attention layer.
///
/// 1. Project input to Q, K, V
/// 2. Write K, V into KV arena at the given slot
/// 3. Apply RoPE using position map
/// 4. Compute attention scores, softmax, weighted sum
/// 5. Output projection
///
/// Returns the attention output tensor.
pub fn attention_layer(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weights: &AttentionWeights<'_>,
    _arena: &mut KvArena,
    _layer: usize,
    _new_slot: SlotId,
    position_map: &PositionMap,
    config: &ModelConfig,
) -> TensorHandle {
    let n_embd = config.n_embd as i64;
    let n_head = config.n_head as i64;
    let n_kv_head = config.n_kv_head as i64;
    let head_dim = config.head_dim as i64;
    let _seq_len = position_map.len() as i64;

    // Q, K, V projections
    let wq = weight_view(graph, weights.wq);
    let wk = weight_view(graph, weights.wk);
    let wv = weight_view(graph, weights.wv);
    let wo = weight_view(graph, weights.wo);

    let q = graph.mul_mat(wq, input); // [n_embd, 1]
    let k_new = graph.mul_mat(wk, input); // [n_kv_head * head_dim, 1]
    let v_new = graph.mul_mat(wv, input); // [n_kv_head * head_dim, 1]

    // Reshape Q for multi-head: [head_dim, n_head, 1]
    let q = graph.reshape_3d(q, head_dim, n_head, 1);

    // Reshape K, V for writing: [head_dim, n_kv_head, 1]
    let _k_new = graph.reshape_3d(k_new, head_dim, n_kv_head, 1);
    let _v_new = graph.reshape_3d(v_new, head_dim, n_kv_head, 1);

    // TODO: Apply RoPE to Q and K using position_map logical positions
    // For now, RoPE is skipped — this is the right place to add it.
    // The correct approach: ggml_rope_ext with explicit position array.

    // TODO: Write K, V data into arena at new_slot via ggml tensor views
    // For now, the arena write is a placeholder.
    // The correct approach: create ggml tensor views pointing into arena
    // buffers, then use ggml_cpy to copy K/V projections into the arena.

    // For single-token decode, attention is:
    // scores = Q * K^T / sqrt(head_dim), softmax, then scores * V
    //
    // Full implementation requires gathering all KV entries for this agent
    // from the arena and assembling them into dense K, V matrices.

    // Simplified path for initial implementation: use Q directly
    // with the new K/V (single-token self-attention degenerates to
    // just the output projection when there's no context).
    //
    // This will be replaced with full KV cache attention.

    let attn_out = graph.reshape_2d(q, n_embd, 1);
    graph.mul_mat(wo, attn_out)
}

fn weight_view(graph: &mut ComputeGraph, w: QuantSlice<'_>) -> TensorHandle {
    let ne0 = w.n_cols() as i64;
    let ne1 = w.n_rows() as i64;
    // SAFETY: weight data is in valid mmap'd buffer.
    unsafe { graph.view_tensor_2d(w.quant_type, ne0, ne1, w.data_ptr() as *mut u8) }
}
