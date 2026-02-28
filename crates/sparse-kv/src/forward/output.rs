use crate::quant::QuantSlice;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// Output projection: hidden -> logits.
/// Computes logits = W_output * hidden, shape [n_vocab, n_tokens].
pub fn output_projection(
    graph: &mut ComputeGraph,
    hidden: TensorHandle,
    weights: QuantSlice<'_>,
) -> TensorHandle {
    let ne0 = weights.n_cols() as i64;
    let ne1 = weights.n_rows() as i64;
    // SAFETY: weight data is in valid mmap'd buffer.
    let w = unsafe { graph.view_tensor_2d(weights.quant_type, ne0, ne1, weights.data_ptr() as *mut u8) };

    graph.mul_mat(w, hidden)
}
