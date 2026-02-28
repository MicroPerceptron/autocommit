use ggml_sys::ffi;

use crate::quant::QuantType;

/// Non-owning view of a ggml_tensor. Lifetime is bounded by the ComputeGraph
/// that owns the ggml_context in which this tensor was allocated.
#[derive(Debug, Clone, Copy)]
pub struct TensorHandle {
    ptr: *mut ffi::ggml_tensor,
}

impl TensorHandle {
    /// # Safety
    /// `ptr` must be a valid, non-null ggml_tensor pointer from a live ggml_context.
    pub unsafe fn from_raw(ptr: *mut ffi::ggml_tensor) -> Self {
        debug_assert!(!ptr.is_null());
        Self { ptr }
    }

    pub fn as_ptr(self) -> *mut ffi::ggml_tensor {
        self.ptr
    }

    pub fn shape(self) -> [i64; 4] {
        // SAFETY: ptr is valid for the lifetime of the ComputeGraph.
        unsafe { (*self.ptr).ne }
    }

    pub fn n_elements(self) -> i64 {
        // SAFETY: querying a valid tensor.
        unsafe { ffi::ggml_nelements(self.ptr) }
    }

    pub fn quant_type(self) -> Option<QuantType> {
        // SAFETY: reading type field from valid tensor.
        let t = unsafe { (*self.ptr).type_ };
        QuantType::from_ggml_type(t)
    }

    /// Get a pointer to the tensor's data buffer.
    /// # Safety
    /// The tensor must have been allocated with data or computed.
    pub unsafe fn data_ptr(self) -> *mut u8 {
        // SAFETY: caller guarantees tensor has been allocated/computed.
        unsafe { (*self.ptr).data as *mut u8 }
    }
}
