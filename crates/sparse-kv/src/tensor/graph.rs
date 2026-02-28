use std::ptr;

use ggml_sys::ffi;

use crate::error::InferenceError;
use crate::quant::QuantType;
use crate::tensor::handle::TensorHandle;

/// RAII wrapper around a ggml compute graph.
/// Owns the ggml_context that holds all tensor allocations for one forward pass.
pub struct ComputeGraph {
    ctx: *mut ffi::ggml_context,
}

// SAFETY: ComputeGraph is used on a single thread within one forward pass.
// The ggml_context it wraps is not shared.
unsafe impl Send for ComputeGraph {}

impl ComputeGraph {
    /// Allocate a new ggml context with the given memory budget (bytes).
    pub fn new(mem_size: usize) -> Result<Self, InferenceError> {
        let params = ffi::ggml_init_params {
            mem_size,
            mem_buffer: ptr::null_mut(),
            no_alloc: false,
        };

        // SAFETY: ggml_init allocates a context with the given params.
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err(InferenceError::Compute(
                "failed to allocate ggml context".into(),
            ));
        }

        Ok(Self { ctx })
    }

    /// Create a new tensor owned by this context.
    pub fn new_tensor_1d(&mut self, dtype: QuantType, ne0: i64) -> TensorHandle {
        // SAFETY: ctx is valid, ggml_new_tensor_1d allocates within ctx.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_new_tensor_1d(
                self.ctx,
                dtype.to_ggml_type(),
                ne0,
            ))
        }
    }

    pub fn new_tensor_2d(&mut self, dtype: QuantType, ne0: i64, ne1: i64) -> TensorHandle {
        // SAFETY: ctx is valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_new_tensor_2d(
                self.ctx,
                dtype.to_ggml_type(),
                ne0,
                ne1,
            ))
        }
    }

    pub fn new_tensor_3d(
        &mut self,
        dtype: QuantType,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> TensorHandle {
        // SAFETY: ctx is valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_new_tensor_3d(
                self.ctx,
                dtype.to_ggml_type(),
                ne0,
                ne1,
                ne2,
            ))
        }
    }

    /// Create a view tensor pointing at external data (zero-copy).
    ///
    /// # Safety
    /// `data` must remain valid for the lifetime of this ComputeGraph.
    pub unsafe fn view_tensor_1d(
        &mut self,
        dtype: QuantType,
        ne0: i64,
        data: *mut u8,
    ) -> TensorHandle {
        // SAFETY: caller guarantees data pointer validity for ComputeGraph lifetime.
        unsafe {
            let t = ffi::ggml_new_tensor_1d(self.ctx, dtype.to_ggml_type(), ne0);
            (*t).data = data as *mut std::os::raw::c_void;
            TensorHandle::from_raw(t)
        }
    }

    /// Create a 2D view tensor pointing at external data (zero-copy).
    ///
    /// # Safety
    /// `data` must remain valid for the lifetime of this ComputeGraph.
    pub unsafe fn view_tensor_2d(
        &mut self,
        dtype: QuantType,
        ne0: i64,
        ne1: i64,
        data: *mut u8,
    ) -> TensorHandle {
        // SAFETY: caller guarantees data pointer validity for ComputeGraph lifetime.
        unsafe {
            let t = ffi::ggml_new_tensor_2d(self.ctx, dtype.to_ggml_type(), ne0, ne1);
            (*t).data = data as *mut std::os::raw::c_void;
            TensorHandle::from_raw(t)
        }
    }

    // ── ggml op wrappers ──

    pub fn mul_mat(&mut self, a: TensorHandle, b: TensorHandle) -> TensorHandle {
        // SAFETY: both tensors belong to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_mul_mat(self.ctx, a.as_ptr(), b.as_ptr())) }
    }

    pub fn add(&mut self, a: TensorHandle, b: TensorHandle) -> TensorHandle {
        // SAFETY: both tensors belong to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_add(self.ctx, a.as_ptr(), b.as_ptr())) }
    }

    pub fn mul(&mut self, a: TensorHandle, b: TensorHandle) -> TensorHandle {
        // SAFETY: both tensors belong to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_mul(self.ctx, a.as_ptr(), b.as_ptr())) }
    }

    pub fn rms_norm(&mut self, a: TensorHandle, eps: f32) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_rms_norm(self.ctx, a.as_ptr(), eps)) }
    }

    pub fn silu(&mut self, a: TensorHandle) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_silu(self.ctx, a.as_ptr())) }
    }

    pub fn soft_max(&mut self, a: TensorHandle) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_soft_max(self.ctx, a.as_ptr())) }
    }

    pub fn scale(&mut self, a: TensorHandle, s: f32) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_scale(self.ctx, a.as_ptr(), s)) }
    }

    pub fn permute(
        &mut self,
        a: TensorHandle,
        ax0: i32,
        ax1: i32,
        ax2: i32,
        ax3: i32,
    ) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_permute(self.ctx, a.as_ptr(), ax0, ax1, ax2, ax3))
        }
    }

    pub fn reshape_2d(&mut self, a: TensorHandle, ne0: i64, ne1: i64) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_reshape_2d(self.ctx, a.as_ptr(), ne0, ne1)) }
    }

    pub fn reshape_3d(
        &mut self,
        a: TensorHandle,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_reshape_3d(self.ctx, a.as_ptr(), ne0, ne1, ne2))
        }
    }

    pub fn cont(&mut self, a: TensorHandle) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_cont(self.ctx, a.as_ptr())) }
    }

    pub fn transpose(&mut self, a: TensorHandle) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_transpose(self.ctx, a.as_ptr())) }
    }

    pub fn get_rows(&mut self, a: TensorHandle, b: TensorHandle) -> TensorHandle {
        // SAFETY: both tensors belong to this ctx.
        unsafe { TensorHandle::from_raw(ffi::ggml_get_rows(self.ctx, a.as_ptr(), b.as_ptr())) }
    }

    pub fn diag_mask_inf(&mut self, a: TensorHandle, n_past: i32) -> TensorHandle {
        // SAFETY: tensor belongs to this ctx.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_diag_mask_inf(self.ctx, a.as_ptr(), n_past))
        }
    }

    pub fn ctx_ptr(&self) -> *mut ffi::ggml_context {
        self.ctx
    }
}

impl Drop for ComputeGraph {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx was allocated in new() and is freed exactly once here.
            unsafe {
                ffi::ggml_free(self.ctx);
            }
        }
    }
}
