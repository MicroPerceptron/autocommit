use crate::quant::QuantSlice;
use crate::quant::QuantType;
use crate::tensor::graph::ComputeGraph;
use crate::tensor::handle::TensorHandle;

/// RMS normalization: output = (input / rms(input)) * weight
pub fn rms_norm(
    graph: &mut ComputeGraph,
    input: TensorHandle,
    weight: QuantSlice<'_>,
    eps: f32,
) -> TensorHandle {
    let normed = graph.rms_norm(input, eps);

    // Weight is always F32 for norm layers
    let n_embd = weight.n_cols() as i64;
    // SAFETY: weight.data_ptr() points into a valid mmap'd buffer.
    let w = unsafe { graph.view_tensor_1d(QuantType::F32, n_embd, weight.data_ptr() as *mut u8) };

    graph.mul(normed, w)
}
