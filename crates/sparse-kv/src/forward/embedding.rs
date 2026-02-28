use crate::quant::QuantSlice;
use crate::quant::QuantType;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// Look up token embeddings from the embedding weight matrix.
///
/// Returns `(embedding_output, token_ids_tensor)`. The `token_ids_tensor`
/// is marked as input and must be filled with token IDs via
/// `ggml_backend_tensor_set` after graph allocation.
pub fn embed_tokens(
    graph: &mut ComputeGraph,
    n_tokens: i64,
    weights: QuantSlice<'_>,
) -> (TensorHandle, TensorHandle) {
    // Create token ID tensor as I32 (index type for get_rows)
    // Marked as input so the gallocr preserves it.
    let ids = graph.new_tensor_1d(QuantType::I32, n_tokens);
    graph.set_input(ids);

    // Create weight matrix view
    let n_embd = weights.n_cols() as i64;
    let n_vocab = weights.n_rows() as i64;
    // SAFETY: weights.data_ptr() points into valid mmap'd buffer.
    let embd_weights = unsafe {
        graph.view_tensor_2d(weights.quant_type, n_embd, n_vocab, weights.data_ptr() as *mut u8)
    };

    // get_rows extracts rows from the weight matrix by token IDs
    let output = graph.get_rows(embd_weights, ids);
    (output, ids)
}
