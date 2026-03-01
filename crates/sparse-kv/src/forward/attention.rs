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
///
/// `kv_offset` is the slot index where this agent's KV context begins in the arena.
/// After `compact_for_agent()`, the target agent's slots are at `kv_offset..kv_offset+seq_len`.
/// For single-agent use, pass `kv_offset = 0`.
pub fn attention_layer(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weights: &AttentionWeights<'_>,
    arena: &mut KvArena,
    layer: usize,
    new_slot: SlotId,
    pos_tensor: TensorHandle,
    seq_len: i64,
    kv_offset: usize,
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

    // Gather the full K, V context for attention (seq_len slots for this agent).
    // After compact_for_agent(), this agent's slots are contiguous at kv_offset..kv_offset+seq_len.
    let elem_bytes = kv_dtype.row_size(1); // bytes per scalar element
    let nb1_kv = (head_dim * n_kv_head) as usize * elem_bytes;
    let kv_byte_offset = kv_offset * nb1_kv;

    let k_full = graph.view_3d(
        k_arena_full,
        head_dim,
        n_kv_head,
        seq_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        kv_byte_offset,
    );

    let v_full = graph.view_3d(
        v_arena_full,
        head_dim,
        n_kv_head,
        seq_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        kv_byte_offset,
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

/// Run one attention layer for a batch of new tokens (batched prefill/decode).
///
/// Like `attention_layer` but processes `n_tokens` tokens simultaneously.
/// Requires a causal mask to prevent future-token attention.
///
/// - `input`: [n_embd, n_tokens]
/// - `pos_tensor`: I32 tensor of length n_tokens (logical positions)
/// - `new_slots`: SlotIds for the n_tokens new entries (must be contiguous in arena)
/// - `seq_len`: number of existing KV entries BEFORE this batch
/// - `kv_offset`: arena slot index where this agent's KV starts
/// - `mask`: causal mask [kv_len, n_tokens] where kv_len = seq_len + n_tokens
///
/// Returns [n_embd, n_tokens] — output for all tokens.
pub fn attention_layer_batched(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weights: &AttentionWeights<'_>,
    arena: &mut KvArena,
    layer: usize,
    new_slots: &[SlotId],
    pos_tensor: TensorHandle,
    seq_len: i64,
    kv_offset: usize,
    mask: TensorHandle,
    config: &ModelConfig,
) -> TensorHandle {
    let n_embd = config.n_embd as i64;
    let n_head = config.n_head as i64;
    let n_kv_head = config.n_kv_head as i64;
    let head_dim = config.head_dim as i64;
    let n_tokens = new_slots.len() as i64;
    let entry_size = arena.entry_size();
    let kv_dtype = arena.dtype();

    // Q, K, V projections: input is [n_embd, n_tokens]
    let wq = weight_view(graph, weights.wq);
    let wk = weight_view(graph, weights.wk);
    let wv = weight_view(graph, weights.wv);
    let wo = weight_view(graph, weights.wo);

    let q = graph.mul_mat(wq, input); // [n_embd, n_tokens]
    let k_new = graph.mul_mat(wk, input); // [n_kv_head * head_dim, n_tokens]
    let v_new = graph.mul_mat(wv, input); // [n_kv_head * head_dim, n_tokens]

    // Reshape for multi-head layout
    let q = graph.reshape_3d(q, head_dim, n_head, n_tokens);
    let k_new = graph.reshape_3d(k_new, head_dim, n_kv_head, n_tokens);
    let v_new = graph.reshape_3d(v_new, head_dim, n_kv_head, n_tokens);

    // Apply RoPE to all n_tokens positions at once
    let q = graph.rope_ext(q, pos_tensor, head_dim as i32, config.rope_theta);
    let k_new = graph.rope_ext(k_new, pos_tensor, head_dim as i32, config.rope_theta);

    // Arena buffer views for this layer
    let k_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_key_ptr(layer),
        )
    };
    graph.set_input(k_arena_full);

    let v_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_val_ptr(layer),
        )
    };
    graph.set_input(v_arena_full);

    // Write K_new/V_new for all n_tokens into consecutive arena slots
    let first_new_idx = new_slots[0].0 as usize;
    let new_byte_offset = first_new_idx * entry_size;
    let total_new_elements = head_dim * n_kv_head * n_tokens;

    let k_new_flat = graph.reshape_2d(k_new, total_new_elements, 1);
    let k_new_flat = graph.view_1d(k_new_flat, total_new_elements, 0);
    let k_arena_view = graph.view_1d(k_arena_full, total_new_elements, new_byte_offset);
    let _k_cpy = graph.cpy(k_new_flat, k_arena_view);

    let v_new_flat = graph.reshape_2d(v_new, total_new_elements, 1);
    let v_new_flat = graph.view_1d(v_new_flat, total_new_elements, 0);
    let v_arena_view = graph.view_1d(v_arena_full, total_new_elements, new_byte_offset);
    let _v_cpy = graph.cpy(v_new_flat, v_arena_view);

    // Full K/V context: existing seq_len + n_tokens new entries
    let kv_len = seq_len + n_tokens;
    let elem_bytes = kv_dtype.row_size(1);
    let nb1_kv = (head_dim * n_kv_head) as usize * elem_bytes;
    let kv_byte_offset = kv_offset * nb1_kv;

    let k_full = graph.view_3d(
        k_arena_full,
        head_dim,
        n_kv_head,
        kv_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        kv_byte_offset,
    );

    let v_full = graph.view_3d(
        v_arena_full,
        head_dim,
        n_kv_head,
        kv_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        kv_byte_offset,
    );

    // Flash attention with causal mask
    // Q [head_dim, n_head, n_tokens] × K [head_dim, n_kv_head, kv_len]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = graph.flash_attn_ext(q, k_full, v_full, Some(mask), scale);

    // attn_out is [head_dim, n_head, n_tokens] → [n_embd, n_tokens]
    let attn_out = graph.reshape_2d(attn_out, n_embd, n_tokens);

    // Output projection
    graph.mul_mat(wo, attn_out)
}

/// Attention layer for continuous batching (multi-agent, 1 token each).
///
/// Like `attention_layer_batched` but with **scattered K/V writes** — each
/// agent's new K/V goes to a different arena slot (not consecutive).
///
/// - `input`: [n_embd, n_agents]
/// - `new_slots`: one SlotId per agent (NOT necessarily consecutive)
/// - `pos_tensor`: I32 [n_agents] with heterogeneous positions
/// - `total_kv_len`: total KV entries visible in the arena after compaction
/// - `mask`: block-diagonal mask [total_kv_len, n_agents]
pub fn attention_layer_continuous(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weights: &AttentionWeights<'_>,
    arena: &mut KvArena,
    layer: usize,
    new_slots: &[SlotId],
    pos_tensor: TensorHandle,
    total_kv_len: i64,
    mask: TensorHandle,
    config: &ModelConfig,
) -> TensorHandle {
    let n_embd = config.n_embd as i64;
    let n_head = config.n_head as i64;
    let n_kv_head = config.n_kv_head as i64;
    let head_dim = config.head_dim as i64;
    let n_agents = new_slots.len() as i64;
    let entry_size = arena.entry_size();
    let kv_dtype = arena.dtype();

    // Q, K, V projections: input is [n_embd, n_agents]
    let wq = weight_view(graph, weights.wq);
    let wk = weight_view(graph, weights.wk);
    let wv = weight_view(graph, weights.wv);
    let wo = weight_view(graph, weights.wo);

    let q = graph.mul_mat(wq, input);
    let k_new = graph.mul_mat(wk, input);
    let v_new = graph.mul_mat(wv, input);

    // Reshape for multi-head layout
    let q = graph.reshape_3d(q, head_dim, n_head, n_agents);
    let k_new = graph.reshape_3d(k_new, head_dim, n_kv_head, n_agents);
    let v_new = graph.reshape_3d(v_new, head_dim, n_kv_head, n_agents);

    // Apply RoPE (heterogeneous positions handled natively)
    let q = graph.rope_ext(q, pos_tensor, head_dim as i32, config.rope_theta);
    let k_new = graph.rope_ext(k_new, pos_tensor, head_dim as i32, config.rope_theta);

    // Arena buffer views for this layer
    let k_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_key_ptr(layer),
        )
    };
    graph.set_input(k_arena_full);

    let v_arena_full = unsafe {
        graph.view_tensor_1d(
            kv_dtype,
            (arena.capacity() as i64) * head_dim * n_kv_head,
            arena.layer_val_ptr(layer),
        )
    };
    graph.set_input(v_arena_full);

    // Scattered K/V writes: one copy per agent
    let entry_elements = head_dim * n_kv_head;
    let k_new_flat = graph.reshape_2d(k_new, entry_elements, n_agents);
    let v_new_flat = graph.reshape_2d(v_new, entry_elements, n_agents);

    for i in 0..new_slots.len() {
        let slot_idx = new_slots[i].0 as usize;
        let slot_byte_offset = slot_idx * entry_size;
        let token_byte_offset = i * entry_size;

        // Extract token i's K and copy to its arena slot
        let k_token = graph.view_1d(k_new_flat, entry_elements, token_byte_offset);
        let k_slot = graph.view_1d(k_arena_full, entry_elements, slot_byte_offset);
        let _k_cpy = graph.cpy(k_token, k_slot);

        // Extract token i's V and copy to its arena slot
        let v_token = graph.view_1d(v_new_flat, entry_elements, token_byte_offset);
        let v_slot = graph.view_1d(v_arena_full, entry_elements, slot_byte_offset);
        let _v_cpy = graph.cpy(v_token, v_slot);
    }

    // Full K/V context: all agents' KV contiguous after compaction
    let elem_bytes = kv_dtype.row_size(1);
    let nb1_kv = (head_dim * n_kv_head) as usize * elem_bytes;

    let k_full = graph.view_3d(
        k_arena_full,
        head_dim,
        n_kv_head,
        total_kv_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        0, // offset 0: all agents start from the beginning after compact_for_batch
    );

    let v_full = graph.view_3d(
        v_arena_full,
        head_dim,
        n_kv_head,
        total_kv_len,
        head_dim as usize * elem_bytes,
        nb1_kv,
        0,
    );

    // Flash attention with block-diagonal mask
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = graph.flash_attn_ext(q, k_full, v_full, Some(mask), scale);

    // [head_dim, n_head, n_agents] → [n_embd, n_agents]
    let attn_out = graph.reshape_2d(attn_out, n_embd, n_agents);

    // Output projection
    graph.mul_mat(wo, attn_out)
}

fn weight_view(graph: &mut ComputeGraph, w: QuantSlice<'_>) -> TensorHandle {
    let ne0 = w.n_cols() as i64;
    let ne1 = w.n_rows() as i64;
    // SAFETY: weight data is in valid mmap'd buffer.
    unsafe { graph.view_tensor_2d(w.quant_type, ne0, ne1, w.data_ptr() as *mut u8) }
}
