use crate::quant::QuantType;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// Create a causal prefill mask tensor and return (tensor_handle, mask_data).
///
/// Shape: [kv_len, n_tokens] where kv_len = seq_len + n_tokens.
/// - `seq_len`: number of existing KV entries before this batch
/// - `n_tokens`: number of new tokens being processed in this batch
///
/// Mask values:
/// - 0.0 where attention is allowed (kv_pos <= seq_len + q_pos)
/// - -inf where blocked (kv_pos > seq_len + q_pos)
///
/// The returned Vec<f32> must be written to the tensor after graph allocation
/// via `ggml_backend_tensor_set`.
pub fn causal_prefill_mask(
    graph: &mut ComputeGraph,
    seq_len: i64,
    n_tokens: i64,
) -> (TensorHandle, Vec<f32>) {
    let kv_len = seq_len + n_tokens;
    let mask_tensor = graph.new_tensor_2d(QuantType::F32, kv_len, n_tokens);
    graph.set_input(mask_tensor);

    let mut data = vec![0.0f32; (kv_len * n_tokens) as usize];
    for q in 0..n_tokens {
        for kv in 0..kv_len {
            // q-th new token (at absolute position seq_len + q) can attend to
            // all KV positions up to and including seq_len + q.
            if kv > seq_len + q {
                data[(q * kv_len + kv) as usize] = f32::NEG_INFINITY;
            }
        }
    }

    (mask_tensor, data)
}

/// Build a block-diagonal mask for continuous batching (multi-agent decode).
///
/// Shape: [total_kv_len, n_agents] (F32)
/// - 0.0 where attention is allowed
/// - -inf where blocked
///
/// Agent i (query column i) can only attend to KV positions
/// in `[ranges[i].0, ranges[i].0 + ranges[i].1)`.
///
/// `ranges`: one per agent, `(kv_offset, kv_len)` where `kv_len` includes
/// the new token being decoded in this step.
///
/// `total_kv_len`: total number of occupied KV entries visible in the arena
/// (must be >= max(offset + kv_len) across all agents).
pub fn block_diagonal_mask(
    graph: &mut ComputeGraph,
    ranges: &[(usize, usize)],
    total_kv_len: i64,
) -> (TensorHandle, Vec<f32>) {
    let n_agents = ranges.len() as i64;
    let mask_tensor = graph.new_tensor_2d(QuantType::F32, total_kv_len, n_agents);
    graph.set_input(mask_tensor);

    let mut data = vec![f32::NEG_INFINITY; (total_kv_len * n_agents) as usize];
    for (q, &(offset, kv_len)) in ranges.iter().enumerate() {
        for kv in offset..offset + kv_len {
            data[q * (total_kv_len as usize) + kv] = 0.0;
        }
    }

    (mask_tensor, data)
}

/// Build a mask for continuous batching with a shared prefix region.
///
/// Each agent sees:
/// - The shared prefix at `[0, shared_len)` (common to all agents)
/// - Its own private KV range at `[private_offset, private_offset + private_len)`
///
/// Shape: [total_kv_len, n_agents] (F32)
///
/// When `shared_len == 0`, this degenerates to `block_diagonal_mask`
/// with contiguous ranges.
pub fn block_diagonal_mask_with_shared(
    graph: &mut ComputeGraph,
    shared_len: usize,
    private_ranges: &[(usize, usize)], // (offset, private_len) per agent
    total_kv_len: i64,
) -> (TensorHandle, Vec<f32>) {
    let n_agents = private_ranges.len() as i64;
    let mask_tensor = graph.new_tensor_2d(QuantType::F32, total_kv_len, n_agents);
    graph.set_input(mask_tensor);

    let mut data = vec![f32::NEG_INFINITY; (total_kv_len * n_agents) as usize];
    for (q, &(offset, private_len)) in private_ranges.iter().enumerate() {
        let row = q * (total_kv_len as usize);
        // Allow shared prefix
        for kv in 0..shared_len {
            data[row + kv] = 0.0;
        }
        // Allow private range
        for kv in offset..offset + private_len {
            data[row + kv] = 0.0;
        }
    }

    (mask_tensor, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_pure_prefill() {
        // Pure prefill: seq_len=0, n_tokens=4
        // Expected: lower-triangular mask
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();
        let (_, data) = causal_prefill_mask(&mut graph, 0, 4);

        let kv_len = 4;
        // q=0: can see kv=0 only
        assert_eq!(data[0 * kv_len + 0], 0.0);
        assert_eq!(data[0 * kv_len + 1], f32::NEG_INFINITY);
        assert_eq!(data[0 * kv_len + 2], f32::NEG_INFINITY);
        assert_eq!(data[0 * kv_len + 3], f32::NEG_INFINITY);

        // q=1: can see kv=0,1
        assert_eq!(data[1 * kv_len + 0], 0.0);
        assert_eq!(data[1 * kv_len + 1], 0.0);
        assert_eq!(data[1 * kv_len + 2], f32::NEG_INFINITY);
        assert_eq!(data[1 * kv_len + 3], f32::NEG_INFINITY);

        // q=3: can see all
        assert_eq!(data[3 * kv_len + 0], 0.0);
        assert_eq!(data[3 * kv_len + 1], 0.0);
        assert_eq!(data[3 * kv_len + 2], 0.0);
        assert_eq!(data[3 * kv_len + 3], 0.0);
    }

    #[test]
    fn causal_mask_with_existing_context() {
        // seq_len=3, n_tokens=2
        // kv_len = 5
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();
        let (_, data) = causal_prefill_mask(&mut graph, 3, 2);

        let kv_len = 5;
        // q=0 (abs pos 3): can see kv 0..3 (existing) + kv 3 (self) = 0..3
        assert_eq!(data[0 * kv_len + 0], 0.0);
        assert_eq!(data[0 * kv_len + 1], 0.0);
        assert_eq!(data[0 * kv_len + 2], 0.0);
        assert_eq!(data[0 * kv_len + 3], 0.0);
        assert_eq!(data[0 * kv_len + 4], f32::NEG_INFINITY);

        // q=1 (abs pos 4): can see all 5
        assert_eq!(data[1 * kv_len + 0], 0.0);
        assert_eq!(data[1 * kv_len + 4], 0.0);
    }

    #[test]
    fn causal_mask_single_token() {
        // Single token decode: seq_len=10, n_tokens=1
        // Should be all-visible (no masking needed, but consistent)
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();
        let (_, data) = causal_prefill_mask(&mut graph, 10, 1);

        let kv_len = 11;
        // q=0 can see all 11 kv entries
        for kv in 0..kv_len {
            assert_eq!(data[kv], 0.0, "kv={kv} should be visible");
        }
    }

    #[test]
    fn causal_mask_dimensions() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();
        let (_, data) = causal_prefill_mask(&mut graph, 5, 3);
        assert_eq!(data.len(), 8 * 3); // kv_len=8, n_tokens=3
    }

    // ── Block-diagonal mask ──────────────────────────────────────

    #[test]
    fn block_diag_two_agents() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();

        // Agent 0: KV at [0..3), Agent 1: KV at [3..6)
        let ranges = vec![(0, 3), (3, 3)];
        let total_kv = 6i64;
        let (_, data) = block_diagonal_mask(&mut graph, &ranges, total_kv);

        assert_eq!(data.len(), 6 * 2);

        // Query 0 (agent 0): sees kv 0,1,2; blocked 3,4,5
        assert_eq!(data[0 * 6 + 0], 0.0);
        assert_eq!(data[0 * 6 + 1], 0.0);
        assert_eq!(data[0 * 6 + 2], 0.0);
        assert_eq!(data[0 * 6 + 3], f32::NEG_INFINITY);
        assert_eq!(data[0 * 6 + 4], f32::NEG_INFINITY);
        assert_eq!(data[0 * 6 + 5], f32::NEG_INFINITY);

        // Query 1 (agent 1): blocked 0,1,2; sees 3,4,5
        assert_eq!(data[1 * 6 + 0], f32::NEG_INFINITY);
        assert_eq!(data[1 * 6 + 1], f32::NEG_INFINITY);
        assert_eq!(data[1 * 6 + 2], f32::NEG_INFINITY);
        assert_eq!(data[1 * 6 + 3], 0.0);
        assert_eq!(data[1 * 6 + 4], 0.0);
        assert_eq!(data[1 * 6 + 5], 0.0);
    }

    #[test]
    fn block_diag_three_agents() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();

        // Different KV lengths: 2, 4, 1
        let ranges = vec![(0, 2), (2, 4), (6, 1)];
        let total_kv = 7i64;
        let (_, data) = block_diagonal_mask(&mut graph, &ranges, total_kv);

        assert_eq!(data.len(), 7 * 3);

        // Agent 0: sees [0,1], blocked rest
        assert_eq!(data[0 * 7 + 0], 0.0);
        assert_eq!(data[0 * 7 + 1], 0.0);
        assert_eq!(data[0 * 7 + 2], f32::NEG_INFINITY);

        // Agent 1: sees [2,3,4,5], blocked rest
        assert_eq!(data[1 * 7 + 1], f32::NEG_INFINITY);
        assert_eq!(data[1 * 7 + 2], 0.0);
        assert_eq!(data[1 * 7 + 5], 0.0);
        assert_eq!(data[1 * 7 + 6], f32::NEG_INFINITY);

        // Agent 2: sees [6] only
        assert_eq!(data[2 * 7 + 5], f32::NEG_INFINITY);
        assert_eq!(data[2 * 7 + 6], 0.0);
    }

    // ── Shared prefix mask ─────────────────────────────────────

    #[test]
    fn shared_prefix_two_agents() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();

        // Shared prefix: 3 slots at [0..3)
        // Agent 0 private: [3..5)  (2 slots)
        // Agent 1 private: [5..8)  (3 slots)
        // total_kv = 8
        let private_ranges = vec![(3, 2), (5, 3)];
        let (_, data) = block_diagonal_mask_with_shared(&mut graph, 3, &private_ranges, 8);

        assert_eq!(data.len(), 8 * 2);

        // Agent 0: sees [0..3) shared + [3..5) private, blocked [5..8)
        for kv in 0..5 {
            assert_eq!(data[0 * 8 + kv], 0.0, "agent 0 should see kv={kv}");
        }
        for kv in 5..8 {
            assert_eq!(data[0 * 8 + kv], f32::NEG_INFINITY, "agent 0 blocked at kv={kv}");
        }

        // Agent 1: sees [0..3) shared + [5..8) private, blocked [3..5)
        for kv in 0..3 {
            assert_eq!(data[1 * 8 + kv], 0.0, "agent 1 should see shared kv={kv}");
        }
        for kv in 3..5 {
            assert_eq!(data[1 * 8 + kv], f32::NEG_INFINITY, "agent 1 blocked at kv={kv}");
        }
        for kv in 5..8 {
            assert_eq!(data[1 * 8 + kv], 0.0, "agent 1 should see private kv={kv}");
        }
    }

    #[test]
    fn shared_only_no_private() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();

        // Agents with shared prefix only, no private slots yet
        let private_ranges = vec![(3, 0), (3, 0)];
        let (_, data) = block_diagonal_mask_with_shared(&mut graph, 3, &private_ranges, 3);

        // Both agents see only the shared prefix [0..3)
        for q in 0..2 {
            for kv in 0..3 {
                assert_eq!(data[q * 3 + kv], 0.0);
            }
        }
    }

    #[test]
    fn block_diag_single_agent() {
        let graph_size = 1024 * 1024;
        let mut graph = ComputeGraph::new(graph_size, true).unwrap();

        let ranges = vec![(0, 5)];
        let (_, data) = block_diagonal_mask(&mut graph, &ranges, 5);

        // Single agent sees everything
        for kv in 0..5 {
            assert_eq!(data[kv], 0.0);
        }
    }
}
