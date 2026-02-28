use std::ptr;

use ggml_sys::ffi;

use crate::error::InferenceError;
use crate::quant::QuantType;
use crate::tensor::backend::Backend;
use crate::tensor::handle::TensorHandle;

/// RAII wrapper around a ggml compute graph.
/// Owns the ggml_context that holds all tensor allocations for one forward pass.
pub struct ComputeGraph {
    ctx: *mut ffi::ggml_context,
    cgraph: *mut ffi::ggml_cgraph,
    galloc: ffi::ggml_gallocr_t,
}

// SAFETY: ComputeGraph is used on a single thread within one forward pass.
// The ggml_context it wraps is not shared.
unsafe impl Send for ComputeGraph {}

impl ComputeGraph {
    /// Allocate a new ggml context with the given memory budget (bytes).
    ///
    /// When `no_alloc` is true, tensors are metadata-only — the backend
    /// allocator assigns real memory during `compute()`.
    pub fn new(mem_size: usize, no_alloc: bool) -> Result<Self, InferenceError> {
        let params = ffi::ggml_init_params {
            mem_size,
            mem_buffer: ptr::null_mut(),
            no_alloc,
        };

        // SAFETY: ggml_init allocates a context with the given params.
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err(InferenceError::Compute(
                "failed to allocate ggml context".into(),
            ));
        }

        Ok(Self {
            ctx,
            cgraph: ptr::null_mut(),
            galloc: ptr::null_mut(),
        })
    }

    /// Build a forward compute graph rooted at the given output tensor.
    pub fn build_forward(&mut self, output: TensorHandle) {
        // SAFETY: ctx is valid. ggml_new_graph allocates within ctx.
        unsafe {
            self.cgraph = ffi::ggml_new_graph(self.ctx);
            ffi::ggml_build_forward_expand(self.cgraph, output.as_ptr());
        }
    }

    /// Allocate graph buffers on the given backend.
    /// After this call, input tensors can be filled via `ggml_backend_tensor_set`.
    pub fn alloc_graph(&mut self, backend: &Backend) -> Result<(), InferenceError> {
        if self.cgraph.is_null() {
            return Err(InferenceError::Compute(
                "no graph built — call build_forward first".into(),
            ));
        }

        // SAFETY: backend is valid, cgraph is built.
        unsafe {
            let buft = ffi::ggml_backend_get_default_buffer_type(backend.as_ptr());
            self.galloc = ffi::ggml_gallocr_new(buft);
            if self.galloc.is_null() {
                return Err(InferenceError::Compute(
                    "failed to create graph allocator".into(),
                ));
            }

            if !ffi::ggml_gallocr_alloc_graph(self.galloc, self.cgraph) {
                return Err(InferenceError::Compute(
                    "graph allocation failed".into(),
                ));
            }
        }

        Ok(())
    }

    /// Execute the compute graph on the given backend.
    /// Must be called after `alloc_graph` and after setting input tensor data.
    pub fn execute(&self, backend: &Backend) -> Result<(), InferenceError> {
        if self.cgraph.is_null() {
            return Err(InferenceError::Compute(
                "no graph to execute".into(),
            ));
        }
        backend.graph_compute(self.cgraph)
    }

    // ── Tensor creation ──

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

    // ── ggml view ops (into existing tensors) ──

    /// Create a 1D view into an existing tensor at a byte offset.
    pub fn view_1d(&mut self, src: TensorHandle, ne0: i64, offset: usize) -> TensorHandle {
        // SAFETY: src tensor is valid, offset is within bounds.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_view_1d(self.ctx, src.as_ptr(), ne0, offset))
        }
    }

    /// Create a 2D view into an existing tensor.
    pub fn view_2d(
        &mut self,
        src: TensorHandle,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> TensorHandle {
        // SAFETY: src tensor is valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_view_2d(
                self.ctx,
                src.as_ptr(),
                ne0,
                ne1,
                nb1,
                offset,
            ))
        }
    }

    /// Create a 3D view into an existing tensor.
    pub fn view_3d(
        &mut self,
        src: TensorHandle,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> TensorHandle {
        // SAFETY: src tensor is valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_view_3d(
                self.ctx,
                src.as_ptr(),
                ne0,
                ne1,
                ne2,
                nb1,
                nb2,
                offset,
            ))
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

    /// Copy tensor `a` into tensor `b` (may involve type conversion).
    /// Returns `b` with the copy operation attached.
    pub fn cpy(&mut self, src: TensorHandle, dst: TensorHandle) -> TensorHandle {
        // SAFETY: both tensors belong to this ctx.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_cpy(self.ctx, src.as_ptr(), dst.as_ptr()))
        }
    }

    /// Apply RoPE (Rotary Position Embedding) with explicit position tensor.
    ///
    /// - `a`: input tensor [head_dim, n_heads, seq_len]
    /// - `positions`: I32 tensor with logical position for each token
    /// - `freq_base`: RoPE theta (typically 10000.0)
    /// - `n_dims`: number of dimensions to rotate (typically head_dim)
    pub fn rope_ext(
        &mut self,
        a: TensorHandle,
        positions: TensorHandle,
        n_dims: i32,
        freq_base: f32,
    ) -> TensorHandle {
        // SAFETY: tensors are valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_rope_ext(
                self.ctx,
                a.as_ptr(),
                positions.as_ptr(),
                ptr::null_mut(), // freq_factors (c parameter) — not used
                n_dims,
                0,     // mode: 0 = default
                0,     // n_ctx_orig: 0 = not used for standard RoPE
                freq_base,
                1.0,   // freq_scale: 1.0 = no scaling
                0.0,   // ext_factor: 0.0 = no YaRN extension
                1.0,   // attn_factor: 1.0 = default
                0.0,   // beta_fast: not used without YaRN
                0.0,   // beta_slow: not used without YaRN
            ))
        }
    }

    /// Flash attention: Q * K^T / sqrt(d) → mask → softmax → * V
    ///
    /// - `q`: [head_dim, n_heads, n_tokens]
    /// - `k`: [head_dim, n_kv_heads, kv_len]
    /// - `v`: [head_dim, n_kv_heads, kv_len]
    /// - `mask`: [kv_len, n_tokens] or null for no mask
    /// - `scale`: typically 1/sqrt(head_dim)
    pub fn flash_attn_ext(
        &mut self,
        q: TensorHandle,
        k: TensorHandle,
        v: TensorHandle,
        mask: Option<TensorHandle>,
        scale: f32,
    ) -> TensorHandle {
        let mask_ptr = mask.map_or(ptr::null_mut(), |m| m.as_ptr());
        // SAFETY: all tensors are valid.
        unsafe {
            TensorHandle::from_raw(ffi::ggml_flash_attn_ext(
                self.ctx,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                mask_ptr,
                scale,
                0.0, // max_bias: 0.0 = no ALiBi
                0.0, // logit_softcap: 0.0 = disabled
            ))
        }
    }

    /// Mark a tensor as an input (will not be overwritten by the allocator).
    pub fn set_input(&mut self, t: TensorHandle) {
        // SAFETY: tensor is valid.
        unsafe { ffi::ggml_set_input(t.as_ptr()) }
    }

    /// Mark a tensor as an output (will not be freed by the allocator).
    pub fn set_output(&mut self, t: TensorHandle) {
        // SAFETY: tensor is valid.
        unsafe { ffi::ggml_set_output(t.as_ptr()) }
    }

    pub fn ctx_ptr(&self) -> *mut ffi::ggml_context {
        self.ctx
    }
}

impl Drop for ComputeGraph {
    fn drop(&mut self) {
        // Free graph allocator first (it references buffer types, not the context)
        if !self.galloc.is_null() {
            // SAFETY: galloc was allocated in compute() and freed exactly once.
            unsafe { ffi::ggml_gallocr_free(self.galloc) }
        }
        if !self.ctx.is_null() {
            // SAFETY: ctx was allocated in new() and is freed exactly once.
            unsafe { ffi::ggml_free(self.ctx) }
        }
    }
}
