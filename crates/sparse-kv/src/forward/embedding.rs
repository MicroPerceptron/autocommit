use crate::quant::QuantSlice;
use crate::quant::QuantType;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// Look up token embeddings from the embedding weight matrix.
/// Returns a tensor of shape [n_embd, n_tokens].
pub fn embed_tokens(
    graph: &mut ComputeGraph,
    token_ids: &[i32],
    weights: QuantSlice<'_>,
) -> TensorHandle {
    let n_tokens = token_ids.len() as i64;

    // Create token ID tensor as I32 (index type for get_rows)
    let ids = graph.new_tensor_1d(QuantType::I32, n_tokens);
    // SAFETY: ids tensor was just allocated as I32 with sufficient size for n_tokens elements.
    unsafe {
        let ptr = (*ids.as_ptr()).data as *mut i32;
        std::ptr::copy_nonoverlapping(token_ids.as_ptr(), ptr, token_ids.len());
    }

    // Create weight matrix view
    let n_embd = weights.n_cols() as i64;
    let n_vocab = weights.n_rows() as i64;
    // SAFETY: weights.data_ptr() points into valid mmap'd buffer.
    let embd_weights = unsafe {
        graph.view_tensor_2d(weights.quant_type, n_embd, n_vocab, weights.data_ptr() as *mut u8)
    };

    // get_rows extracts rows from the weight matrix by token IDs
    graph.get_rows(embd_weights, ids)
}
