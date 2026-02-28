use ggml_sys::ffi;

use crate::error::InferenceError;

/// RAII wrapper around a ggml backend buffer.
///
/// Backend buffers hold tensor data in memory visible to the compute backend
/// (CPU heap, Metal shared memory, CUDA device memory, etc.).
pub struct BackendBuffer {
    ptr: ffi::ggml_backend_buffer_t,
}

// SAFETY: BackendBuffer is used on a single thread. The underlying buffer
// is not shared across threads.
unsafe impl Send for BackendBuffer {}

impl BackendBuffer {
    /// Wrap a raw backend buffer pointer.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null buffer returned by ggml.
    pub(crate) unsafe fn from_raw(ptr: ffi::ggml_backend_buffer_t) -> Self {
        debug_assert!(!ptr.is_null());
        Self { ptr }
    }

    pub fn as_ptr(&self) -> ffi::ggml_backend_buffer_t {
        self.ptr
    }

    /// Get the base pointer of the buffer's memory.
    pub fn base_ptr(&self) -> *mut u8 {
        // SAFETY: ptr is valid for the lifetime of this wrapper.
        unsafe { ffi::ggml_backend_buffer_get_base(self.ptr) as *mut u8 }
    }

    /// Size of the buffer in bytes.
    pub fn size(&self) -> usize {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_buffer_get_size(self.ptr) }
    }

    /// Clear the buffer to a specific byte value.
    pub fn clear(&mut self, value: u8) {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_buffer_clear(self.ptr, value) }
    }

    /// Whether this buffer is host-accessible (CPU-side memory).
    pub fn is_host(&self) -> bool {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_buffer_is_host(self.ptr) }
    }

    /// Set the buffer usage hint.
    pub fn set_usage(&mut self, usage: ffi::ggml_backend_buffer_usage) {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_buffer_set_usage(self.ptr, usage) }
    }

    /// Initialize a tensor within this buffer at a given address.
    pub fn init_tensor(
        &mut self,
        tensor: *mut ffi::ggml_tensor,
    ) -> Result<(), InferenceError> {
        // SAFETY: buffer and tensor are valid.
        let status = unsafe { ffi::ggml_backend_buffer_init_tensor(self.ptr, tensor) };
        if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(InferenceError::Compute(
                "failed to init tensor in backend buffer".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for BackendBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by ggml and is freed exactly once.
            unsafe { ffi::ggml_backend_buffer_free(self.ptr) }
        }
    }
}
