use std::ffi::CStr;

use ggml_sys::ffi;

use crate::error::InferenceError;
use crate::tensor::buffer::BackendBuffer;

/// RAII wrapper around a ggml compute backend.
///
/// Represents a compute device (CPU, Metal, CUDA, etc.) that can execute
/// compute graphs and allocate device-visible memory.
pub struct Backend {
    ptr: ffi::ggml_backend_t,
}

// SAFETY: Backend is used on a single thread.
unsafe impl Send for Backend {}

impl Backend {
    /// Initialize the best available backend (GPU if available, CPU fallback).
    pub fn init_best() -> Result<Self, InferenceError> {
        // SAFETY: ggml_backend_init_best() selects the best device.
        let ptr = unsafe { ffi::ggml_backend_init_best() };
        if ptr.is_null() {
            return Err(InferenceError::Compute(
                "failed to initialize any ggml backend".into(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Initialize the CPU backend.
    pub fn cpu() -> Result<Self, InferenceError> {
        // SAFETY: ggml_backend_cpu_init() initializes the CPU backend.
        let ptr = unsafe { ffi::ggml_backend_cpu_init() };
        if ptr.is_null() {
            return Err(InferenceError::Compute(
                "failed to initialize CPU backend".into(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Initialize the Metal backend (macOS only).
    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, InferenceError> {
        // SAFETY: ggml_backend_metal_init() initializes the Metal backend.
        let ptr = unsafe { ffi::ggml_backend_metal_init() };
        if ptr.is_null() {
            return Err(InferenceError::Compute(
                "failed to initialize Metal backend".into(),
            ));
        }
        Ok(Self { ptr })
    }

    pub fn as_ptr(&self) -> ffi::ggml_backend_t {
        self.ptr
    }

    /// Get the backend's human-readable name.
    pub fn name(&self) -> &str {
        // SAFETY: ptr is valid, ggml_backend_name returns a static string.
        unsafe {
            let cstr = ffi::ggml_backend_name(self.ptr);
            CStr::from_ptr(cstr).to_str().unwrap_or("unknown")
        }
    }

    /// Allocate a buffer on this backend.
    pub fn alloc_buffer(&self, size: usize) -> Result<BackendBuffer, InferenceError> {
        // SAFETY: ptr is valid.
        let buf = unsafe { ffi::ggml_backend_alloc_buffer(self.ptr, size) };
        if buf.is_null() {
            return Err(InferenceError::Compute(format!(
                "failed to allocate {size} bytes on backend '{}'",
                self.name()
            )));
        }
        // SAFETY: buf is a valid non-null buffer.
        Ok(unsafe { BackendBuffer::from_raw(buf) })
    }

    /// Get the default buffer type for this backend.
    pub fn default_buffer_type(&self) -> ffi::ggml_backend_buffer_type_t {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_get_default_buffer_type(self.ptr) }
    }

    /// Execute a compute graph on this backend.
    pub fn graph_compute(
        &self,
        cgraph: *mut ffi::ggml_cgraph,
    ) -> Result<(), InferenceError> {
        // SAFETY: ptr and cgraph are valid.
        let status = unsafe { ffi::ggml_backend_graph_compute(self.ptr, cgraph) };
        if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(InferenceError::Compute(format!(
                "graph compute failed on backend '{}' (status={status})",
                self.name()
            )));
        }
        Ok(())
    }

    /// Synchronize the backend (wait for pending operations).
    pub fn synchronize(&self) {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_synchronize(self.ptr) }
    }

    /// Get the maximum allocation size for this backend.
    pub fn max_size(&self) -> usize {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_get_max_size(self.ptr) }
    }

    /// Get alignment requirements for this backend.
    pub fn alignment(&self) -> usize {
        // SAFETY: ptr is valid.
        unsafe { ffi::ggml_backend_get_alignment(self.ptr) }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by ggml and is freed exactly once.
            unsafe { ffi::ggml_backend_free(self.ptr) }
        }
    }
}
