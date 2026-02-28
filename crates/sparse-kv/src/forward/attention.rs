use crate::arena::kv_arena::KvArena;
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

/// Run one attention layer for a single new token.
///
/// 1. Project input to Q, K, V
/// 2. Apply RoPE to Q and K_new using the position tensor
/// 3. Write K_new, V_new into the KV arena at the given slot
/// 4. Gather full K, V context for this agent from the arena
/// 5. Compute flash attention: Q * K^T / sqrt(head_dim) → softmax → * V
/// 6. Output projection
///
/// `pos_tensor` is a pre-allocated I32 tensor of length 1, created by the caller
/// and filled with the current logical position before graph execution.
pub fn attention_layer(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weights: &AttentionWeights<'_>,
    arena: &mut KvArena,
    layer: usize,
    new_slot: SlotId,
    pos_tensor: TensorHandle,
    seq_len: i64,
    config: &ModelConfig,
) -> TensorHandle {
    let n_embd = config.n_embd as i64;
    let n_head = config.n_head as i64;
    let n_kv_head = config.n_kv_head as i64;
    let head_dim = config.head_dim as i64;
    let entry_size = arena.entry_size();
    let kv_dtype = arena.dtype();

    // Q, K, V projections from input
    let wq = weight_view(graph, weights.wq);
    let wk = weight_view(graph, weights.wk);
    let wv = weight_view(graph, weights.wv);
    let wo = weight_view(graph, weights.wo);

    let q = graph.mul_mat(wq, input); // [n_embd, 1]
    let k_new = graph.mul_mat(wk, input); // [n_kv_head * head_dim, 1]
    let v_new = graph.mul_mat(wv, input); // [n_kv_head * head_dim, 1]

    // Reshape for multi-head layout: [head_dim, n_heads, 1]
    let q = graph.reshape_3d(q, head_dim, n_head, 1);
    let k_new = graph.reshape_3d(k_new, head_dim, n_kv_head, 1);
    let v_new = graph.reshape_3d(v_new, head_dim, n_kv_head, 1);

    let q = graph.rope_ext(q, pos_tensor, head_dim as i32, config.rope_theta);
    let k_new = graph.rope_ext(k_new, pos_tensor, head_dim as i32, config.rope_theta);

    // Write K_new and V_new into the arena at the new slot
    // Create views into the arena's K and V buffers at the slot offset
    let slot_idx = new_slot.0 as usize;

    // K arena: full key buffer for this layer
    // SAFETY: layer and slot are valid, pointers live for the graph's lifetime.
    let k_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_key_ptr(layer),
        )
    };
    graph.set_input(k_arena_full);

    // V arena: full value buffer for this layer
    let v_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_val_ptr(layer),
        )
    };
    graph.set_input(v_arena_full);

    // Create views at the specific slot offset for writing
    let slot_byte_offset = slot_idx * entry_size;
    let k_slot_view = graph.view_1d(k_arena_full, head_dim * n_kv_head, slot_byte_offset);
    let v_slot_view = graph.view_1d(v_arena_full, head_dim * n_kv_head, slot_byte_offset);

    // Flatten k_new and v_new for copy (they're [head_dim, n_kv_head, 1])
    let k_new_flat = graph.reshape_2d(k_new, head_dim * n_kv_head, 1);
    let v_new_flat = graph.reshape_2d(v_new, head_dim * n_kv_head, 1);
    let k_new_flat = graph.view_1d(k_new_flat, head_dim * n_kv_head, 0);
    let v_new_flat = graph.view_1d(v_new_flat, head_dim * n_kv_head, 0);

    // Copy new K/V into the arena
    let _k_cpy = graph.cpy(k_new_flat, k_slot_view);
    let _v_cpy = graph.cpy(v_new_flat, v_slot_view);

    // Gather the full K, V context for attention (all seq_len slots)
    // After compact(), slots 0..seq_len are contiguous in the arena.
    // Create views spanning exactly seq_len entries.
    let elem_bytes = kv_dtype.row_size(1); // bytes per scalar element
    let nb1_kv = (head_dim * n_kv_head) as usize * elem_bytes;

    let k_full = graph.view_3d(
        k_arena_full,
        head_dim,
        n_kv_head,
        seq_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        0,
    );

    let v_full = graph.view_3d(
        v_arena_full,
        head_dim,
        n_kv_head,
        seq_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        0,
    );

    // Flash attention: Q [head_dim, n_head, 1] × K [head_dim, n_kv_head, seq_len]
    // flash_attn_ext handles GQA (n_head > n_kv_head) automatically
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = graph.flash_attn_ext(q, k_full, v_full, None, scale);

    // attn_out is [head_dim, n_head, 1] — reshape to [n_embd, 1]
    let attn_out = graph.reshape_2d(attn_out, n_embd, 1);

    // Output projection
    graph.mul_mat(wo, attn_out)
}

fn weight_view(graph: &mut ComputeGraph, w: QuantSlice<'_>) -> TensorHandle {
    let ne0 = w.n_cols() as i64;
    let ne1 = w.n_rows() as i64;
    // SAFETY: weight data is in valid mmap'd buffer.
    unsafe { graph.view_tensor_2d(w.quant_type, ne0, ne1, w.data_ptr() as *mut u8) }
}
