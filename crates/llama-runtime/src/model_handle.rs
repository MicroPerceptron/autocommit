use std::ffi::CString;
use std::path::Path;

use llama_sys::ffi;

use crate::error::RuntimeError;

#[derive(Debug)]
pub(crate) struct ModelHandle {
    ptr: *mut ffi::llama_model,
    vocab: *const ffi::llama_vocab,
    n_embd: usize,
    has_encoder: bool,
    has_decoder: bool,
}

// SAFETY: ModelHandle wraps an immutable llama_model after construction. All mutations happen
// through context objects, and model lifetime is managed by Arc + Drop.
unsafe impl Send for ModelHandle {}
// SAFETY: read-only model access is thread-safe for shared ownership in this crate design.
unsafe impl Sync for ModelHandle {}

impl ModelHandle {
    pub(crate) fn load(model_path: &Path, profile: &str) -> Result<Self, RuntimeError> {
        if !model_path.exists() {
            return Err(RuntimeError::Embed(format!(
                "embedding model not found: {}",
                model_path.display()
            )));
        }

        let path_cstr = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::Embed(format!(
                "embedding model path contains interior NUL: {}",
                model_path.display()
            ))
        })?;

        let ptr = unsafe {
            // SAFETY: FFI call with default params and a valid NUL-terminated path.
            let mut mparams = ffi::llama_model_default_params();
            if profile.eq_ignore_ascii_case("cpu") {
                mparams.n_gpu_layers = 0;
            }
            ffi::llama_model_load_from_file(path_cstr.as_ptr(), mparams)
        };

        if ptr.is_null() {
            return Err(RuntimeError::Embed(format!(
                "failed to load embedding model from {}",
                model_path.display()
            )));
        }

        let vocab = unsafe {
            // SAFETY: model pointer is valid while ModelHandle owns it.
            ffi::llama_model_get_vocab(ptr)
        };
        if vocab.is_null() {
            unsafe {
                // SAFETY: ptr was allocated by llama_model_load_from_file and must be freed on error.
                ffi::llama_model_free(ptr);
            }
            return Err(RuntimeError::Embed(
                "failed to fetch model vocabulary".to_string(),
            ));
        }

        let n_embd = unsafe {
            // SAFETY: model pointer is valid while ModelHandle owns it.
            let out = ffi::llama_model_n_embd_out(ptr);
            let dim = if out > 0 {
                out
            } else {
                ffi::llama_model_n_embd(ptr)
            };
            if dim <= 0 { 0 } else { dim as usize }
        };

        if n_embd == 0 {
            unsafe {
                // SAFETY: ptr was allocated by llama_model_load_from_file and must be freed on error.
                ffi::llama_model_free(ptr);
            }
            return Err(RuntimeError::Embed(
                "model reports invalid embedding dimension".to_string(),
            ));
        }

        let (has_encoder, has_decoder) = unsafe {
            // SAFETY: model pointer is valid while ModelHandle owns it.
            (
                ffi::llama_model_has_encoder(ptr),
                ffi::llama_model_has_decoder(ptr),
            )
        };

        Ok(Self {
            ptr,
            vocab,
            n_embd,
            has_encoder,
            has_decoder,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::llama_model {
        self.ptr
    }

    pub(crate) fn vocab(&self) -> *const ffi::llama_vocab {
        self.vocab
    }

    pub(crate) fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub(crate) fn has_encoder(&self) -> bool {
        self.has_encoder
    }

    pub(crate) fn has_decoder(&self) -> bool {
        self.has_decoder
    }
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: pointer is owned by this handle and dropped exactly once.
            if !self.ptr.is_null() {
                ffi::llama_model_free(self.ptr);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}
