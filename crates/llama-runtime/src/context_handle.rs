use std::ffi::CString;
use std::slice;
use std::sync::Arc;

use llama_sys::ffi;

use crate::error::RuntimeError;
use crate::model_handle::ModelHandle;

const EMBEDDING_SEQ_ID: ffi::llama_seq_id = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContextMode {
    Generation,
    Embedding,
}

#[derive(Debug)]
pub(crate) struct ContextHandle {
    model: Arc<ModelHandle>,
    ptr: *mut ffi::llama_context,
    mode: ContextMode,
    pooling_type: ffi::llama_pooling_type,
}

// SAFETY: ContextHandle is never accessed concurrently in this crate; Engine wraps it in Mutex
// and only hands out mutable access for FFI calls.
unsafe impl Send for ContextHandle {}

impl ContextHandle {
    pub(crate) fn new_generation(model: Arc<ModelHandle>) -> Result<Self, RuntimeError> {
        Self::new(model, ContextMode::Generation)
    }

    pub(crate) fn new_embedding(model: Arc<ModelHandle>) -> Result<Self, RuntimeError> {
        Self::new(model, ContextMode::Embedding)
    }

    fn new(model: Arc<ModelHandle>, mode: ContextMode) -> Result<Self, RuntimeError> {
        let ptr = unsafe {
            // SAFETY: context params are initialized from llama defaults and configured by mode.
            let mut cparams = ffi::llama_context_default_params();
            cparams.n_seq_max = 1;

            match mode {
                ContextMode::Generation => {
                    cparams.embeddings = false;
                    cparams.pooling_type = ffi::llama_pooling_type_LLAMA_POOLING_TYPE_NONE;
                }
                ContextMode::Embedding => {
                    cparams.embeddings = true;
                    cparams.pooling_type = ffi::llama_pooling_type_LLAMA_POOLING_TYPE_MEAN;
                }
            }

            let ctx = ffi::llama_init_from_model(model.as_ptr(), cparams);
            if !ctx.is_null() && mode == ContextMode::Embedding {
                ffi::llama_set_embeddings(ctx, true);
            }
            ctx
        };

        if ptr.is_null() {
            return Err(RuntimeError::Embed(match mode {
                ContextMode::Generation => "failed to initialize generation context".to_string(),
                ContextMode::Embedding => "failed to initialize embedding context".to_string(),
            }));
        }

        if mode == ContextMode::Embedding && model.has_encoder() && model.has_decoder() {
            unsafe {
                // SAFETY: pointer is valid and owned by this constructor path.
                ffi::llama_free(ptr);
            }
            return Err(RuntimeError::Embed(
                "encoder-decoder models are not currently supported for embedding extraction"
                    .to_string(),
            ));
        }

        let pooling_type = unsafe {
            // SAFETY: pointer is valid while ContextHandle owns it.
            ffi::llama_pooling_type(ptr)
        };

        Ok(Self {
            model,
            ptr,
            mode,
            pooling_type,
        })
    }

    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        if self.mode != ContextMode::Embedding {
            return Err(RuntimeError::Embed(
                "embed called on non-embedding context".to_string(),
            ));
        }

        let tokens = self.tokenize(text)?;
        if tokens.is_empty() {
            return Err(RuntimeError::Embed(
                "tokenization produced an empty token list".to_string(),
            ));
        }

        let mut batch_handle = BatchHandle::new(tokens.len())?;
        let batch = batch_handle.as_mut();

        unsafe {
            // SAFETY: ptr is valid while ContextHandle is alive.
            let mem = ffi::llama_get_memory(self.ptr);
            ffi::llama_memory_clear(mem, true);

            for (idx, token) in tokens.iter().copied().enumerate() {
                *batch.token.add(idx) = token;
                *batch.pos.add(idx) = idx as i32;
                *batch.n_seq_id.add(idx) = 1;

                let seq_slot = *batch.seq_id.add(idx);
                if seq_slot.is_null() {
                    return Err(RuntimeError::Embed(
                        "batch sequence slot allocation failed".to_string(),
                    ));
                }

                *seq_slot = EMBEDDING_SEQ_ID;
                *batch.logits.add(idx) = 1;
            }

            batch.n_tokens = tokens.len() as i32;
        }

        let status = unsafe {
            // SAFETY: context and batch are initialized and valid.
            if self.model.has_encoder() && !self.model.has_decoder() {
                ffi::llama_encode(self.ptr, *batch)
            } else {
                ffi::llama_decode(self.ptr, *batch)
            }
        };

        if status != 0 {
            return Err(RuntimeError::Embed(format!(
                "llama embedding pass failed with status {status}"
            )));
        }

        let emb_ptr = unsafe {
            // SAFETY: result pointers are owned by llama context and valid after successful encode/decode.
            if self.pooling_type == ffi::llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
                ffi::llama_get_embeddings_ith(self.ptr, batch.n_tokens - 1)
            } else {
                ffi::llama_get_embeddings_seq(self.ptr, EMBEDDING_SEQ_ID)
            }
        };

        if emb_ptr.is_null() {
            return Err(RuntimeError::Embed(
                "llama returned null embedding pointer".to_string(),
            ));
        }

        let vector = unsafe {
            // SAFETY: emb_ptr points to at least n_embd contiguous floats for the selected output.
            slice::from_raw_parts(emb_ptr, self.model.n_embd()).to_vec()
        };

        Ok(vector)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<ffi::llama_token>, RuntimeError> {
        let text_cstr = CString::new(text)
            .map_err(|_| RuntimeError::Embed("input text contains interior NUL".to_string()))?;

        let text_len = i32::try_from(text.len())
            .map_err(|_| RuntimeError::Embed("input text length exceeds i32".to_string()))?;

        let initial_cap = text.len().saturating_add(8);
        let initial_cap = i32::try_from(initial_cap)
            .map_err(|_| RuntimeError::Embed("token buffer size exceeds i32".to_string()))?;

        let mut tokens = vec![0 as ffi::llama_token; initial_cap as usize];

        let mut n_tokens = unsafe {
            // SAFETY: vocab pointer is valid and buffers are allocated for n_tokens_max elements.
            ffi::llama_tokenize(
                self.model.vocab(),
                text_cstr.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                initial_cap,
                true,
                true,
            )
        };

        if n_tokens == i32::MIN {
            return Err(RuntimeError::Embed(
                "tokenization failed: input too large for int32 token count".to_string(),
            ));
        }

        if n_tokens < 0 {
            let required = n_tokens
                .checked_neg()
                .ok_or_else(|| RuntimeError::Embed("tokenization size overflow".to_string()))?;

            tokens.resize(required as usize, 0);

            let check = unsafe {
                // SAFETY: resized token buffer has exactly required capacity requested by previous call.
                ffi::llama_tokenize(
                    self.model.vocab(),
                    text_cstr.as_ptr(),
                    text_len,
                    tokens.as_mut_ptr(),
                    required,
                    true,
                    true,
                )
            };

            if check != required {
                return Err(RuntimeError::Embed(format!(
                    "tokenization retry mismatch: expected {required}, got {check}"
                )));
            }

            n_tokens = check;
        }

        if n_tokens <= 0 {
            return Err(RuntimeError::Embed(
                "tokenization produced no tokens".to_string(),
            ));
        }

        tokens.truncate(n_tokens as usize);
        Ok(tokens)
    }
}

impl Drop for ContextHandle {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: pointer is owned by this handle and dropped exactly once.
            if !self.ptr.is_null() {
                ffi::llama_free(self.ptr);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

#[derive(Debug)]
struct BatchHandle {
    batch: ffi::llama_batch,
}

impl BatchHandle {
    fn new(n_tokens: usize) -> Result<Self, RuntimeError> {
        let n_tokens_i32 = i32::try_from(n_tokens)
            .map_err(|_| RuntimeError::Embed("batch token count exceeds i32".to_string()))?;

        let batch = unsafe {
            // SAFETY: llama_batch_init allocates internal arrays for requested capacity.
            ffi::llama_batch_init(n_tokens_i32, 0, 1)
        };

        if batch.token.is_null()
            || batch.pos.is_null()
            || batch.n_seq_id.is_null()
            || batch.seq_id.is_null()
            || batch.logits.is_null()
        {
            unsafe {
                // SAFETY: safe to free partial/failed allocations from llama_batch_init.
                ffi::llama_batch_free(batch);
            }
            return Err(RuntimeError::Embed(
                "failed to allocate llama batch buffers".to_string(),
            ));
        }

        Ok(Self { batch })
    }

    fn as_mut(&mut self) -> &mut ffi::llama_batch {
        &mut self.batch
    }
}

impl Drop for BatchHandle {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: batch was allocated by llama_batch_init and must be freed once.
            ffi::llama_batch_free(self.batch);
        }
    }
}
