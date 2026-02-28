use crate::quant::QuantSlice;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// SwiGLU feed-forward network.
///
/// output = W_down * (silu(W_gate * x) * (W_up * x))
pub fn ffn_silu(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    w_gate: QuantSlice<'_>,
    w_up: QuantSlice<'_>,
    w_down: QuantSlice<'_>,
) -> TensorHandle {
    let gate_w = weight_view(graph, w_gate);
    let up_w = weight_view(graph, w_up);
    let down_w = weight_view(graph, w_down);

    // gate = W_gate * x
    let gate = graph.mul_mat(gate_w, input);
    // gate = silu(gate)
    let gate = graph.silu(gate);

    // up = W_up * x
    let up = graph.mul_mat(up_w, input);

    // gate_up = gate * up (element-wise)
    let gate_up = graph.mul(gate, up);

    // output = W_down * gate_up
    graph.mul_mat(down_w, gate_up)
}

fn weight_view(graph: &mut ComputeGraph, w: QuantSlice<'_>) -> TensorHandle {
    let ne0 = w.n_cols() as i64;
    let ne1 = w.n_rows() as i64;
    // SAFETY: weight data pointer is into a valid mmap'd buffer.
    unsafe { graph.view_tensor_2d(w.quant_type, ne0, ne1, w.data_ptr() as *mut u8) }
}
