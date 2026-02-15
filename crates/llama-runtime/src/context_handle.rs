use std::ffi::{CStr, CString};
use std::fs;
use std::path::Path;
use std::slice;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
    #[allow(dead_code)]
    seq_capacity: usize,
}

// SAFETY: ContextHandle is never accessed concurrently in this crate; Engine wraps it in Mutex
// and only hands out mutable access for FFI calls.
unsafe impl Send for ContextHandle {}

impl ContextHandle {
    pub(crate) fn new_generation(
        model: Arc<ModelHandle>,
        cpu_only: bool,
    ) -> Result<Self, RuntimeError> {
        Self::new(model, ContextMode::Generation, cpu_only)
    }

    pub(crate) fn new_embedding(
        model: Arc<ModelHandle>,
        cpu_only: bool,
    ) -> Result<Self, RuntimeError> {
        Self::new(model, ContextMode::Embedding, cpu_only)
    }

    fn new(
        model: Arc<ModelHandle>,
        mode: ContextMode,
        cpu_only: bool,
    ) -> Result<Self, RuntimeError> {
        let ptr = Self::init_context(&model, mode, cpu_only);

        if ptr.is_null() {
            let (regs, devs) = unsafe {
                // SAFETY: pure backend metadata queries.
                (ffi::ggml_backend_reg_count(), ffi::ggml_backend_dev_count())
            };
            let mode_label = match mode {
                ContextMode::Generation => "generation",
                ContextMode::Embedding => "embedding",
            };
            let inventory = backend_inventory_summary(devs.min(8));
            return Err(RuntimeError::Inference(format!(
                "failed to initialize {mode_label} context (cpu_only={cpu_only}, ggml_backends={regs}, ggml_devices={devs}, inventory={inventory})"
            )));
        }

        let seq_capacity = unsafe {
            // SAFETY: ptr is a valid context returned by llama_init_from_model.
            let raw = ffi::llama_n_seq_max(ptr);
            if raw > 0 { raw as usize } else { 1 }
        };
        if std::env::var("AUTOCOMMIT_LLAMA_LOG")
            .ok()
            .as_deref()
            .map(|v| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
        {
            eprintln!(
                "autocommit context initialized ({mode:?}): seq_capacity={seq_capacity}, cpu_only={cpu_only}"
            );
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
            seq_capacity,
        })
    }

    fn init_context(
        model: &ModelHandle,
        mode: ContextMode,
        cpu_only: bool,
    ) -> *mut ffi::llama_context {
        unsafe {
            // SAFETY: context params copied from fitted model defaults and adjusted by mode.
            let mut cparams = model.context_params();
            if cpu_only {
                cparams.offload_kqv = false;
                cparams.op_offload = false;
                cparams.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }

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
            if !ctx.is_null() {
                ffi::llama_set_embeddings(ctx, mode == ContextMode::Embedding);
            }
            ctx
        }
    }

    fn context_can_shift(&self) -> bool {
        let mem = unsafe {
            // SAFETY: self.ptr is a live context pointer while ContextHandle is alive.
            ffi::llama_get_memory(self.ptr)
        };
        if mem.is_null() {
            return false;
        }
        unsafe {
            // SAFETY: memory handle was returned from a live context.
            ffi::llama_memory_can_shift(mem)
        }
    }

    fn try_shift_context_window(
        &mut self,
        seq_id: ffi::llama_seq_id,
        keep_tokens: usize,
        next_pos: &mut i32,
        n_ctx_seq: usize,
    ) -> Result<bool, RuntimeError> {
        if !self.model.context_shift_enabled() || !self.context_can_shift() {
            return Ok(false);
        }
        if (*next_pos as usize) < n_ctx_seq.saturating_sub(1) {
            return Ok(true);
        }

        let keep = keep_tokens.min((*next_pos).max(0) as usize);
        let shiftable = (*next_pos).max(0) as usize;
        let shiftable = shiftable.saturating_sub(keep);
        if shiftable <= 1 {
            return Ok(false);
        }

        let n_discard = (shiftable / 2).max(1);
        let pos_keep = i32::try_from(keep)
            .map_err(|_| RuntimeError::Inference("keep token count exceeds i32".to_string()))?;
        let pos_discard_end = i32::try_from(keep.saturating_add(n_discard)).map_err(|_| {
            RuntimeError::Inference("context discard position exceeds i32".to_string())
        })?;
        let pos_cur = *next_pos;

        let mem = unsafe {
            // SAFETY: self.ptr is a live context pointer while ContextHandle is alive.
            ffi::llama_get_memory(self.ptr)
        };
        if mem.is_null() {
            return Ok(false);
        }

        let removed = unsafe {
            // SAFETY: memory handle is valid and sequence id/range are bounded by current context positions.
            ffi::llama_memory_seq_rm(mem, seq_id, pos_keep, pos_discard_end)
        };
        if !removed {
            return Ok(false);
        }

        unsafe {
            // SAFETY: shift the remaining tokens in-place for the same valid sequence.
            ffi::llama_memory_seq_add(
                mem,
                seq_id,
                pos_discard_end,
                pos_cur,
                pos_keep - pos_discard_end,
            );
            ffi::llama_synchronize(self.ptr);
        }

        *next_pos = (*next_pos).saturating_sub(i32::try_from(n_discard).unwrap_or(i32::MAX));
        if *next_pos < pos_keep {
            *next_pos = pos_keep;
        }

        Ok((*next_pos as usize) < n_ctx_seq.saturating_sub(1))
    }

    #[allow(dead_code)]
    pub(crate) fn max_sequences(&self) -> usize {
        self.seq_capacity
    }

    pub(crate) fn load_state_file(
        &mut self,
        path: &Path,
    ) -> Result<Vec<ffi::llama_token>, RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::State(
                "state loading is only supported for generation contexts".to_string(),
            ));
        }

        let path_c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::State(format!(
                "state path contains interior NUL byte: {}",
                path.display()
            ))
        })?;

        let n_ctx_seq = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_ctx_seq(self.ptr)
        } as usize;
        let capacity = n_ctx_seq.max(1);
        let mut tokens = vec![0 as ffi::llama_token; capacity];
        let mut token_count: usize = 0;

        let ok = unsafe {
            // SAFETY: context pointer and output buffers are valid for this call.
            ffi::llama_state_load_file(
                self.ptr,
                path_c.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len(),
                &mut token_count,
            )
        };

        if !ok {
            return Err(RuntimeError::State(format!(
                "failed to load state file: {}",
                path.display()
            )));
        }

        if token_count > tokens.len() {
            return Err(RuntimeError::State(format!(
                "loaded token count {} exceeds capacity {} for {}",
                token_count,
                tokens.len(),
                path.display()
            )));
        }

        tokens.truncate(token_count);
        Ok(tokens)
    }

    pub(crate) fn save_state_file(
        &mut self,
        path: &Path,
        tokens: &[ffi::llama_token],
    ) -> Result<(), RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::State(
                "state saving is only supported for generation contexts".to_string(),
            ));
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                RuntimeError::State(format!(
                    "failed to create state directory {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let path_c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::State(format!(
                "state path contains interior NUL byte: {}",
                path.display()
            ))
        })?;

        let token_ptr = if tokens.is_empty() {
            std::ptr::null()
        } else {
            tokens.as_ptr()
        };

        let ok = unsafe {
            // SAFETY: context pointer, path, and optional token buffer are valid for this call.
            ffi::llama_state_save_file(self.ptr, path_c.as_ptr(), token_ptr, tokens.len())
        };

        if !ok {
            return Err(RuntimeError::State(format!(
                "failed to save state file: {}",
                path.display()
            )));
        }

        Ok(())
    }

    pub(crate) fn warmup(&mut self) -> Result<(), RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::Inference(
                "warmup is only supported for generation contexts".to_string(),
            ));
        }

        unsafe {
            // SAFETY: context pointer is valid while handle is alive.
            ffi::llama_set_warmup(self.ptr, true);
        }

        let result = (|| {
            let mut tokens = Vec::with_capacity(2);
            let bos = unsafe {
                // SAFETY: model vocabulary pointer is valid for the context lifetime.
                ffi::llama_vocab_bos(self.model.vocab())
            };
            let eos = unsafe {
                // SAFETY: model vocabulary pointer is valid for the context lifetime.
                ffi::llama_vocab_eos(self.model.vocab())
            };

            if bos != ffi::LLAMA_TOKEN_NULL {
                tokens.push(bos);
            }
            if eos != ffi::LLAMA_TOKEN_NULL {
                tokens.push(eos);
            }
            if tokens.is_empty() {
                tokens.push(0);
            }

            self.decode_prompt_tokens(&tokens, 0)
        })();

        unsafe {
            // SAFETY: context pointer is valid while handle is alive.
            let mem = ffi::llama_get_memory(self.ptr);
            ffi::llama_memory_clear(mem, true);
            ffi::llama_synchronize(self.ptr);
            ffi::llama_perf_context_reset(self.ptr);
            ffi::llama_set_warmup(self.ptr, false);
        }

        result
    }

    #[allow(dead_code)]
    pub(crate) fn decode_sequences(
        &mut self,
        prompts: &[String],
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::Inference(
                "decode_sequences called on non-generation context".to_string(),
            ));
        }

        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        if prompts.len() > self.seq_capacity {
            return Err(RuntimeError::Inference(format!(
                "requested {} sequences exceeds generation capacity {}",
                prompts.len(),
                self.seq_capacity
            )));
        }

        let mut tokenized = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let tokens = self.tokenize(prompt)?;
            if tokens.is_empty() {
                return Err(RuntimeError::Inference(
                    "generation tokenization produced no tokens".to_string(),
                ));
            }

            tokenized.push(tokens);
        }

        let max_batch_tokens = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_batch(self.ptr)
        } as usize;

        if max_batch_tokens == 0 {
            return Err(RuntimeError::Inference(
                "llama context reports n_batch=0".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(prompts.len());
        let mut start = 0usize;

        while start < tokenized.len() {
            let mut end = start;
            let mut total_tokens = 0usize;

            while end < tokenized.len() {
                let next_len = tokenized[end].len();

                if next_len > max_batch_tokens {
                    return Err(RuntimeError::Inference(format!(
                        "single sequence requires {next_len} tokens, exceeding n_batch={max_batch_tokens}"
                    )));
                }

                let next_total = total_tokens
                    .checked_add(next_len)
                    .ok_or_else(|| RuntimeError::Inference("token count overflow".to_string()))?;

                if end > start && next_total > max_batch_tokens {
                    break;
                }

                total_tokens = next_total;
                end += 1;
            }

            outputs.extend(self.decode_tokenized_sequences(&tokenized[start..end], total_tokens)?);
            start = end;
        }

        Ok(outputs)
    }

    #[allow(dead_code)]
    fn decode_tokenized_sequences(
        &mut self,
        tokenized: &[Vec<ffi::llama_token>],
        total_tokens: usize,
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        let mut batch_handle = BatchHandle::new(total_tokens)?;
        let batch = batch_handle.as_mut();

        unsafe {
            // SAFETY: ptr is valid while ContextHandle is alive.
            let mem = ffi::llama_get_memory(self.ptr);
            ffi::llama_memory_clear(mem, true);

            let mut cursor = 0usize;
            for (seq_idx, tokens) in tokenized.iter().enumerate() {
                let seq_id = i32::try_from(seq_idx)
                    .map_err(|_| RuntimeError::Inference("sequence id exceeds i32".to_string()))?;

                for (pos_idx, token) in tokens.iter().copied().enumerate() {
                    *batch.token.add(cursor) = token;
                    *batch.pos.add(cursor) = pos_idx as i32;
                    *batch.n_seq_id.add(cursor) = 1;

                    let seq_slot = *batch.seq_id.add(cursor);
                    if seq_slot.is_null() {
                        return Err(RuntimeError::Inference(
                            "batch sequence slot allocation failed".to_string(),
                        ));
                    }

                    *seq_slot = seq_id;
                    *batch.logits.add(cursor) = if pos_idx + 1 == tokens.len() { 1 } else { 0 };
                    cursor += 1;
                }
            }

            batch.n_tokens = i32::try_from(cursor).map_err(|_| {
                RuntimeError::Inference("batch token count exceeds i32".to_string())
            })?;
        }

        let status = unsafe {
            // SAFETY: context and batch are initialized and valid.
            ffi::llama_decode(self.ptr, *batch)
        };

        if status != 0 {
            return Err(RuntimeError::Inference(format!(
                "llama multi-sequence decode failed with status {status}"
            )));
        }

        let mut out = Vec::with_capacity(tokenized.len());
        for out_idx in 0..tokenized.len() {
            let emb_ptr = unsafe {
                // SAFETY: output slot exists for each token flagged with logits=1.
                ffi::llama_get_embeddings_ith(self.ptr, out_idx as i32)
            };

            if emb_ptr.is_null() {
                return Err(RuntimeError::Inference(format!(
                    "llama returned null embedding pointer for output index {out_idx}"
                )));
            }

            let vector = unsafe {
                // SAFETY: emb_ptr points to at least n_embd contiguous floats.
                slice::from_raw_parts(emb_ptr, self.model.n_embd()).to_vec()
            };
            out.push(vector);
        }

        Ok(out)
    }

    pub(crate) fn generate_text(
        &mut self,
        prompt: &str,
        grammar: Option<&str>,
        requested_max_tokens: usize,
    ) -> Result<String, RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::Inference(
                "generate_text called on non-generation context".to_string(),
            ));
        }

        if requested_max_tokens == 0 {
            return Ok(String::new());
        }

        let prompt_tokens = self.tokenize(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::Inference(
                "prompt tokenization produced no tokens".to_string(),
            ));
        }
        let mut prompt_tokens = prompt_tokens;
        let n_ctx_seq = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_ctx_seq(self.ptr)
        } as usize;
        let max_prompt_tokens = n_ctx_seq.saturating_sub(64).max(1);
        if prompt_tokens.len() > max_prompt_tokens {
            prompt_tokens.truncate(max_prompt_tokens);
        }

        let keep_tokens = self
            .model
            .context_shift_keep_tokens(prompt_tokens.len(), n_ctx_seq);
        let ctx_shift_enabled = self.model.context_shift_enabled() && self.context_can_shift();
        let remaining_slots = n_ctx_seq
            .saturating_sub(prompt_tokens.len())
            .saturating_sub(1);
        let max_tokens = if ctx_shift_enabled {
            requested_max_tokens
        } else {
            requested_max_tokens.min(remaining_slots)
        };
        if max_tokens == 0 {
            return Ok(String::new());
        }

        self.decode_prompt_tokens(&prompt_tokens, 0)?;

        let mut sampler = SamplerChain::new(self.model.vocab(), grammar)?;
        let mut generated = String::new();
        let mut next_pos = i32::try_from(prompt_tokens.len())
            .map_err(|_| RuntimeError::Inference("prompt token count exceeds i32".to_string()))?;

        for _ in 0..max_tokens {
            if grammar.is_some() && is_complete_json_object(&generated) {
                break;
            }

            let token = sampler.sample(self.ptr, -1)?;

            if token == ffi::LLAMA_TOKEN_NULL {
                break;
            }

            if unsafe { ffi::llama_vocab_is_eog(self.model.vocab(), token) } {
                break;
            }

            let piece = self.token_to_piece(token)?;
            sampler.accept(token);

            if !piece.is_empty() {
                generated.push_str(&piece);
            }
            if grammar.is_some() && is_complete_json_object(&generated) {
                break;
            }
            if (next_pos as usize) >= n_ctx_seq.saturating_sub(1) {
                let shifted =
                    self.try_shift_context_window(0, keep_tokens, &mut next_pos, n_ctx_seq)?;
                if !shifted {
                    break;
                }
            }
            self.decode_single_token(token, next_pos, 0)?;
            next_pos = next_pos.saturating_add(1);
        }

        Ok(generated)
    }

    pub(crate) fn generate_texts_with_budgets(
        &mut self,
        prompts: &[String],
        grammar: Option<&str>,
        requested_max_tokens: &[usize],
    ) -> Result<Vec<Result<String, RuntimeError>>, RuntimeError> {
        if self.mode != ContextMode::Generation {
            return Err(RuntimeError::Inference(
                "generate_texts_with_budgets called on non-generation context".to_string(),
            ));
        }

        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        if prompts.len() != requested_max_tokens.len() {
            return Err(RuntimeError::Inference(format!(
                "prompt/budget length mismatch: prompts={}, budgets={}",
                prompts.len(),
                requested_max_tokens.len()
            )));
        }

        if prompts.len() > self.seq_capacity {
            return Err(RuntimeError::Inference(format!(
                "requested {} sequences exceeds generation capacity {}",
                prompts.len(),
                self.seq_capacity
            )));
        }

        let n_ctx_seq = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_ctx_seq(self.ptr)
        } as usize;
        let max_batch_tokens = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_batch(self.ptr)
        } as usize;
        if max_batch_tokens == 0 {
            return Err(RuntimeError::Inference(
                "llama context reports n_batch=0".to_string(),
            ));
        }

        let max_prompt_tokens = n_ctx_seq.saturating_sub(64).max(1);
        let mut tokenized = Vec::with_capacity(prompts.len());
        let mut per_seq_budget = Vec::with_capacity(prompts.len());
        let mut keep_tokens = Vec::with_capacity(prompts.len());
        let ctx_shift_enabled = self.model.context_shift_enabled() && self.context_can_shift();

        for (prompt, requested) in prompts.iter().zip(requested_max_tokens.iter().copied()) {
            let mut tokens = self.tokenize(prompt)?;
            if tokens.is_empty() {
                return Err(RuntimeError::Inference(
                    "prompt tokenization produced no tokens".to_string(),
                ));
            }
            if tokens.len() > max_prompt_tokens {
                tokens.truncate(max_prompt_tokens);
            }
            keep_tokens.push(
                self.model
                    .context_shift_keep_tokens(tokens.len(), n_ctx_seq),
            );

            let remaining_slots = n_ctx_seq.saturating_sub(tokens.len()).saturating_sub(1);
            let budget = if ctx_shift_enabled {
                requested
            } else {
                requested.min(remaining_slots)
            };
            tokenized.push(tokens);
            per_seq_budget.push(budget);
        }

        unsafe {
            // SAFETY: ptr is valid while ContextHandle is alive.
            let mem = ffi::llama_get_memory(self.ptr);
            ffi::llama_memory_clear(mem, true);
        }

        self.decode_prompt_tokens_multi(&tokenized, max_batch_tokens)?;

        let mut samplers = Vec::with_capacity(prompts.len());
        let mut outputs = vec![String::new(); prompts.len()];
        let mut generated = vec![0usize; prompts.len()];
        let mut last_output_idx: Vec<usize> = (0..prompts.len()).collect();
        let mut next_pos = tokenized
            .iter()
            .map(|tokens| {
                i32::try_from(tokens.len()).map_err(|_| {
                    RuntimeError::Inference("prompt token count exceeds i32".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut errors: Vec<Option<RuntimeError>> = (0..prompts.len()).map(|_| None).collect();
        let mut active = vec![false; prompts.len()];

        for idx in 0..prompts.len() {
            if per_seq_budget[idx] == 0 {
                samplers.push(None);
                continue;
            }
            match SamplerChain::new(self.model.vocab(), grammar) {
                Ok(s) => {
                    active[idx] = true;
                    samplers.push(Some(s));
                }
                Err(err) => {
                    errors[idx] = Some(err);
                    samplers.push(None);
                }
            }
        }

        while active.iter().any(|flag| *flag) {
            let mut pending_decode: Vec<(usize, ffi::llama_token)> = Vec::new();

            for seq_idx in 0..prompts.len() {
                if !active[seq_idx] {
                    continue;
                }
                if generated[seq_idx] >= per_seq_budget[seq_idx] {
                    active[seq_idx] = false;
                    continue;
                }

                let Some(sampler) = samplers[seq_idx].as_mut() else {
                    active[seq_idx] = false;
                    continue;
                };

                let token = match sampler.sample(self.ptr, last_output_idx[seq_idx] as i32) {
                    Ok(t) => t,
                    Err(err) => {
                        errors[seq_idx] = Some(err);
                        active[seq_idx] = false;
                        continue;
                    }
                };

                if token == ffi::LLAMA_TOKEN_NULL
                    || unsafe { ffi::llama_vocab_is_eog(self.model.vocab(), token) }
                {
                    active[seq_idx] = false;
                    continue;
                }

                let piece = match self.token_to_piece(token) {
                    Ok(p) => p,
                    Err(err) => {
                        errors[seq_idx] = Some(err);
                        active[seq_idx] = false;
                        continue;
                    }
                };

                sampler.accept(token);
                if !piece.is_empty() {
                    outputs[seq_idx].push_str(&piece);
                }
                generated[seq_idx] = generated[seq_idx].saturating_add(1);

                if grammar.is_some() && is_complete_json_object(&outputs[seq_idx]) {
                    active[seq_idx] = false;
                    continue;
                }
                if generated[seq_idx] >= per_seq_budget[seq_idx] {
                    active[seq_idx] = false;
                    continue;
                }
                if (next_pos[seq_idx] as usize) >= n_ctx_seq.saturating_sub(1) {
                    let seq_id = i32::try_from(seq_idx).map_err(|_| {
                        RuntimeError::Inference("sequence id exceeds i32".to_string())
                    })?;
                    match self.try_shift_context_window(
                        seq_id,
                        keep_tokens[seq_idx],
                        &mut next_pos[seq_idx],
                        n_ctx_seq,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            active[seq_idx] = false;
                            continue;
                        }
                        Err(err) => {
                            errors[seq_idx] = Some(err);
                            active[seq_idx] = false;
                            continue;
                        }
                    }
                }

                pending_decode.push((seq_idx, token));
            }

            if pending_decode.is_empty() {
                continue;
            }

            if pending_decode.len() > max_batch_tokens {
                return Err(RuntimeError::Inference(format!(
                    "active generation sequences {} exceed n_batch {}",
                    pending_decode.len(),
                    max_batch_tokens
                )));
            }

            let mut batch_handle = BatchHandle::new(pending_decode.len())?;
            let batch = batch_handle.as_mut();

            unsafe {
                // SAFETY: batch buffers are allocated by llama_batch_init and writable.
                for (idx, (seq_idx, token)) in pending_decode.iter().copied().enumerate() {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = next_pos[seq_idx];
                    *batch.n_seq_id.add(idx) = 1;

                    let seq_slot = *batch.seq_id.add(idx);
                    if seq_slot.is_null() {
                        return Err(RuntimeError::Inference(
                            "batch sequence slot allocation failed".to_string(),
                        ));
                    }

                    *seq_slot = i32::try_from(seq_idx).map_err(|_| {
                        RuntimeError::Inference("sequence id exceeds i32".to_string())
                    })?;
                    *batch.logits.add(idx) = 1;
                }

                batch.n_tokens = i32::try_from(pending_decode.len()).map_err(|_| {
                    RuntimeError::Inference("batch token count exceeds i32".to_string())
                })?;
            }

            let status = unsafe {
                // SAFETY: context and batch are initialized and valid.
                ffi::llama_decode(self.ptr, *batch)
            };

            if status != 0 {
                for (seq_idx, _) in pending_decode.iter().copied() {
                    errors[seq_idx] = Some(RuntimeError::Inference(format!(
                        "llama multi-sequence token decode failed with status {status}"
                    )));
                    active[seq_idx] = false;
                }
                continue;
            }

            for (out_idx, (seq_idx, _)) in pending_decode.iter().copied().enumerate() {
                next_pos[seq_idx] = next_pos[seq_idx].saturating_add(1);
                last_output_idx[seq_idx] = out_idx;
            }
        }

        let mut results = Vec::with_capacity(prompts.len());
        for idx in 0..prompts.len() {
            if let Some(err) = errors[idx].take() {
                results.push(Err(err));
            } else {
                results.push(Ok(std::mem::take(&mut outputs[idx])));
            }
        }

        Ok(results)
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

    fn decode_prompt_tokens(
        &mut self,
        tokens: &[ffi::llama_token],
        seq_id: ffi::llama_seq_id,
    ) -> Result<(), RuntimeError> {
        let max_batch_tokens = unsafe {
            // SAFETY: pure context metadata query.
            ffi::llama_n_batch(self.ptr)
        } as usize;

        if max_batch_tokens == 0 {
            return Err(RuntimeError::Inference(
                "llama context reports n_batch=0".to_string(),
            ));
        }

        unsafe {
            // SAFETY: ptr is valid while ContextHandle is alive.
            let mem = ffi::llama_get_memory(self.ptr);
            ffi::llama_memory_clear(mem, true);
        }

        let mut start = 0usize;
        while start < tokens.len() {
            let end = (start + max_batch_tokens).min(tokens.len());
            let chunk = &tokens[start..end];

            let mut batch_handle = BatchHandle::new(chunk.len())?;
            let batch = batch_handle.as_mut();

            unsafe {
                for (idx, token) in chunk.iter().copied().enumerate() {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = i32::try_from(start + idx).map_err(|_| {
                        RuntimeError::Inference("prompt token position exceeds i32".to_string())
                    })?;
                    *batch.n_seq_id.add(idx) = 1;

                    let seq_slot = *batch.seq_id.add(idx);
                    if seq_slot.is_null() {
                        return Err(RuntimeError::Inference(
                            "batch sequence slot allocation failed".to_string(),
                        ));
                    }

                    *seq_slot = seq_id;
                    let is_last = end == tokens.len() && idx + 1 == chunk.len();
                    *batch.logits.add(idx) = if is_last { 1 } else { 0 };
                }

                batch.n_tokens = i32::try_from(chunk.len()).map_err(|_| {
                    RuntimeError::Inference("batch token count exceeds i32".to_string())
                })?;
            }

            let status = unsafe {
                // SAFETY: context and batch are initialized and valid.
                ffi::llama_decode(self.ptr, *batch)
            };

            if status != 0 {
                return Err(RuntimeError::Inference(format!(
                    "llama prompt decode failed with status {status}"
                )));
            }

            start = end;
        }

        Ok(())
    }

    fn decode_prompt_tokens_multi(
        &mut self,
        tokenized: &[Vec<ffi::llama_token>],
        max_batch_tokens: usize,
    ) -> Result<(), RuntimeError> {
        if tokenized.len() > max_batch_tokens {
            return Err(RuntimeError::Inference(format!(
                "sequence count {} exceeds n_batch {}",
                tokenized.len(),
                max_batch_tokens
            )));
        }

        // Prefill all but the last token per sequence without logits.
        let mut offsets = vec![0usize; tokenized.len()];
        let prefix_total: usize = tokenized
            .iter()
            .map(|tokens| tokens.len().saturating_sub(1))
            .sum();
        let mut remaining = prefix_total;
        let mut next_seq = 0usize;

        while remaining > 0 {
            let mut packed: Vec<(usize, ffi::llama_token, i32)> = Vec::new();

            while packed.len() < max_batch_tokens && remaining > 0 {
                let mut progressed = false;
                for _ in 0..tokenized.len() {
                    let seq_idx = next_seq;
                    next_seq = (next_seq + 1) % tokenized.len();

                    let prefix_end = tokenized[seq_idx].len().saturating_sub(1);
                    if offsets[seq_idx] >= prefix_end {
                        continue;
                    }

                    let pos = i32::try_from(offsets[seq_idx]).map_err(|_| {
                        RuntimeError::Inference("prompt token position exceeds i32".to_string())
                    })?;
                    let token = tokenized[seq_idx][offsets[seq_idx]];
                    offsets[seq_idx] = offsets[seq_idx].saturating_add(1);
                    remaining = remaining.saturating_sub(1);
                    packed.push((seq_idx, token, pos));
                    progressed = true;

                    if packed.len() >= max_batch_tokens || remaining == 0 {
                        break;
                    }
                }

                if !progressed {
                    break;
                }
            }

            if packed.is_empty() {
                break;
            }

            let mut batch_handle = BatchHandle::new(packed.len())?;
            let batch = batch_handle.as_mut();

            unsafe {
                // SAFETY: batch buffers are allocated by llama_batch_init and writable.
                for (idx, (seq_idx, token, pos)) in packed.iter().copied().enumerate() {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = pos;
                    *batch.n_seq_id.add(idx) = 1;

                    let seq_slot = *batch.seq_id.add(idx);
                    if seq_slot.is_null() {
                        return Err(RuntimeError::Inference(
                            "batch sequence slot allocation failed".to_string(),
                        ));
                    }

                    *seq_slot = i32::try_from(seq_idx).map_err(|_| {
                        RuntimeError::Inference("sequence id exceeds i32".to_string())
                    })?;
                    *batch.logits.add(idx) = 0;
                }

                batch.n_tokens = i32::try_from(packed.len()).map_err(|_| {
                    RuntimeError::Inference("batch token count exceeds i32".to_string())
                })?;
            }

            let status = unsafe {
                // SAFETY: context and batch are initialized and valid.
                ffi::llama_decode(self.ptr, *batch)
            };

            if status != 0 {
                return Err(RuntimeError::Inference(format!(
                    "llama multi-sequence prompt decode failed with status {status}"
                )));
            }
        }

        // Decode the final prompt token for each sequence in one pass so sampling can address
        // a stable output index for every active sequence.
        let mut batch_handle = BatchHandle::new(tokenized.len())?;
        let batch = batch_handle.as_mut();

        unsafe {
            // SAFETY: batch buffers are allocated by llama_batch_init and writable.
            for (idx, tokens) in tokenized.iter().enumerate() {
                let last_pos = tokens.len().saturating_sub(1);
                let token = tokens[last_pos];

                *batch.token.add(idx) = token;
                *batch.pos.add(idx) = i32::try_from(last_pos).map_err(|_| {
                    RuntimeError::Inference("prompt token position exceeds i32".to_string())
                })?;
                *batch.n_seq_id.add(idx) = 1;

                let seq_slot = *batch.seq_id.add(idx);
                if seq_slot.is_null() {
                    return Err(RuntimeError::Inference(
                        "batch sequence slot allocation failed".to_string(),
                    ));
                }
                *seq_slot = i32::try_from(idx)
                    .map_err(|_| RuntimeError::Inference("sequence id exceeds i32".to_string()))?;
                *batch.logits.add(idx) = 1;
            }

            batch.n_tokens = i32::try_from(tokenized.len()).map_err(|_| {
                RuntimeError::Inference("batch token count exceeds i32".to_string())
            })?;
        }

        let status = unsafe {
            // SAFETY: context and batch are initialized and valid.
            ffi::llama_decode(self.ptr, *batch)
        };
        if status != 0 {
            return Err(RuntimeError::Inference(format!(
                "llama multi-sequence prompt tail decode failed with status {status}"
            )));
        }

        Ok(())
    }

    fn decode_single_token(
        &mut self,
        token: ffi::llama_token,
        pos: i32,
        seq_id: ffi::llama_seq_id,
    ) -> Result<(), RuntimeError> {
        let mut batch_handle = BatchHandle::new(1)?;
        let batch = batch_handle.as_mut();

        unsafe {
            // SAFETY: single-token batch buffers are allocated and writable.
            *batch.token = token;
            *batch.pos = pos;
            *batch.n_seq_id = 1;

            if (*batch.seq_id).is_null() {
                return Err(RuntimeError::Inference(
                    "single-token seq slot allocation failed".to_string(),
                ));
            }

            *(*batch.seq_id) = seq_id;
            *batch.logits = 1;
            batch.n_tokens = 1;
        }

        let status = unsafe {
            // SAFETY: context and single-token batch are initialized and valid.
            ffi::llama_decode(self.ptr, *batch)
        };

        if status != 0 {
            return Err(RuntimeError::Inference(format!(
                "llama token decode failed with status {status}"
            )));
        }

        Ok(())
    }

    fn token_to_piece(&self, token: ffi::llama_token) -> Result<String, RuntimeError> {
        let mut buf = vec![0i8; 64];

        let mut written = unsafe {
            // SAFETY: vocab pointer is valid and output buffer is writable.
            ffi::llama_token_to_piece(
                self.model.vocab(),
                token,
                buf.as_mut_ptr(),
                i32::try_from(buf.len()).map_err(|_| {
                    RuntimeError::Inference("piece buffer size exceeds i32".to_string())
                })?,
                0,
                false,
            )
        };

        if written < 0 {
            let needed = written
                .checked_neg()
                .ok_or_else(|| RuntimeError::Inference("token piece size overflow".to_string()))?
                as usize;
            buf.resize(needed, 0);

            written = unsafe {
                // SAFETY: resized buffer provides requested capacity.
                ffi::llama_token_to_piece(
                    self.model.vocab(),
                    token,
                    buf.as_mut_ptr(),
                    i32::try_from(buf.len()).map_err(|_| {
                        RuntimeError::Inference("piece buffer size exceeds i32".to_string())
                    })?,
                    0,
                    false,
                )
            };
        }

        if written <= 0 {
            return Ok(String::new());
        }

        let bytes = unsafe {
            // SAFETY: `written` bytes were populated by llama_token_to_piece.
            slice::from_raw_parts(buf.as_ptr() as *const u8, written as usize)
        };

        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<ffi::llama_token>, RuntimeError> {
        let text_cstr = CString::new(text)
            .map_err(|_| RuntimeError::Inference("input text contains interior NUL".to_string()))?;

        let text_len = i32::try_from(text.len())
            .map_err(|_| RuntimeError::Inference("input text length exceeds i32".to_string()))?;

        let initial_cap = text.len().saturating_add(8);
        let initial_cap = i32::try_from(initial_cap)
            .map_err(|_| RuntimeError::Inference("token buffer size exceeds i32".to_string()))?;

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
            return Err(RuntimeError::Inference(
                "tokenization failed: input too large for int32 token count".to_string(),
            ));
        }

        if n_tokens < 0 {
            let required = n_tokens
                .checked_neg()
                .ok_or_else(|| RuntimeError::Inference("tokenization size overflow".to_string()))?;

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
                return Err(RuntimeError::Inference(format!(
                    "tokenization retry mismatch: expected {required}, got {check}"
                )));
            }

            n_tokens = check;
        }

        if n_tokens <= 0 {
            return Err(RuntimeError::Inference(
                "tokenization produced no tokens".to_string(),
            ));
        }

        tokens.truncate(n_tokens as usize);
        Ok(tokens)
    }
}

fn backend_inventory_summary(limit: usize) -> String {
    let mut parts = Vec::new();
    let count = unsafe {
        // SAFETY: pure backend metadata query.
        ffi::ggml_backend_dev_count()
    };
    let limit = limit.min(count);

    for idx in 0..limit {
        let dev = unsafe {
            // SAFETY: index is bounded by dev_count.
            ffi::ggml_backend_dev_get(idx)
        };
        if dev.is_null() {
            continue;
        }

        let dev_type = unsafe {
            // SAFETY: dev pointer comes from ggml backend registry.
            ffi::ggml_backend_dev_type(dev)
        };
        let name = unsafe {
            // SAFETY: ggml returns a NUL-terminated static/device-owned string.
            CStr::from_ptr(ffi::ggml_backend_dev_name(dev))
                .to_string_lossy()
                .into_owned()
        };
        let desc = unsafe {
            // SAFETY: ggml returns a NUL-terminated static/device-owned string.
            CStr::from_ptr(ffi::ggml_backend_dev_description(dev))
                .to_string_lossy()
                .into_owned()
        };
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            // SAFETY: dev pointer comes from ggml backend registry and out-ptrs are valid.
            ffi::ggml_backend_dev_memory(dev, &mut free, &mut total);
        }

        parts.push(format!(
            "#{idx}:type={dev_type},name={name},desc={desc},total_mib={},free_mib={}",
            total / (1024 * 1024),
            free / (1024 * 1024)
        ));
    }

    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("|")
    }
}

fn is_complete_json_object(text: &str) -> bool {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') {
        return false;
    }

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for ch in trimmed.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return true;
                }
            }
            _ => {}
        }
    }

    false
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
struct SamplerChain {
    vocab: *const ffi::llama_vocab,
    chain_ptr: *mut ffi::llama_sampler,
    grammar_ptr: Option<*mut ffi::llama_sampler>,
}

impl SamplerChain {
    fn new(vocab: *const ffi::llama_vocab, grammar: Option<&str>) -> Result<Self, RuntimeError> {
        let chain_ptr = unsafe {
            // SAFETY: default params are POD values from llama.
            let mut params = ffi::llama_sampler_chain_default_params();
            params.no_perf = true;
            ffi::llama_sampler_chain_init(params)
        };

        if chain_ptr.is_null() {
            return Err(RuntimeError::Inference(
                "failed to initialize sampler chain".to_string(),
            ));
        }

        let grammar_ptr = if let Some(grammar_src) = grammar {
            let grammar_c = CString::new(grammar_src).map_err(|_| {
                RuntimeError::Inference("grammar contains interior NUL".to_string())
            })?;
            let root_c = CString::new("root")
                .map_err(|_| RuntimeError::Inference("invalid grammar root".to_string()))?;
            let ptr = unsafe {
                // SAFETY: vocab and C strings are valid.
                ffi::llama_sampler_init_grammar(vocab, grammar_c.as_ptr(), root_c.as_ptr())
            };
            if ptr.is_null() {
                unsafe {
                    // SAFETY: chain pointer is owned by this constructor path.
                    ffi::llama_sampler_free(chain_ptr);
                }
                return Err(RuntimeError::Inference(
                    "failed to initialize grammar sampler".to_string(),
                ));
            }
            Some(ptr)
        } else {
            None
        };

        let mut chain = Self {
            vocab,
            chain_ptr,
            grammar_ptr,
        };

        chain.add_to_chain(unsafe { ffi::llama_sampler_init_top_k(40) })?;
        chain.add_to_chain(unsafe { ffi::llama_sampler_init_top_p(0.9, 1) })?;
        chain.add_to_chain(unsafe { ffi::llama_sampler_init_temp(0.2) })?;

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(42);
        chain.add_to_chain(unsafe { ffi::llama_sampler_init_dist(seed) })?;

        Ok(chain)
    }

    fn add_to_chain(&mut self, sampler: *mut ffi::llama_sampler) -> Result<(), RuntimeError> {
        if sampler.is_null() {
            return Err(RuntimeError::Inference(
                "failed to initialize sampler component".to_string(),
            ));
        }

        unsafe {
            // SAFETY: chain and sampler pointers are valid and chain takes ownership.
            ffi::llama_sampler_chain_add(self.chain_ptr, sampler);
        }
        Ok(())
    }

    fn sample(
        &mut self,
        ctx: *mut ffi::llama_context,
        idx: i32,
    ) -> Result<ffi::llama_token, RuntimeError> {
        let n_vocab = unsafe {
            // SAFETY: vocab pointer is valid for sampler chain lifetime.
            ffi::llama_vocab_n_tokens(self.vocab)
        };
        if n_vocab <= 0 {
            return Err(RuntimeError::Inference("vocab size is invalid".to_string()));
        }

        let logits = unsafe {
            // SAFETY: context pointer is valid during generation.
            ffi::llama_get_logits_ith(ctx, idx)
        };
        if logits.is_null() {
            return Err(RuntimeError::Inference(format!(
                "missing logits for output index {idx}"
            )));
        }

        let mut cur = logits_to_candidates(logits, n_vocab as usize);
        let mut cur_p = ffi::llama_token_data_array {
            data: cur.as_mut_ptr(),
            size: cur.len(),
            selected: -1,
            sorted: false,
        };

        unsafe {
            // SAFETY: sampler and candidate array are valid.
            ffi::llama_sampler_apply(self.chain_ptr, &mut cur_p);
        }
        let mut token = selected_token(&cur_p)?;

        if let Some(grammar_ptr) = self.grammar_ptr {
            // Check if sampled token satisfies grammar; if not, resample grammar-first.
            let mut single = ffi::llama_token_data {
                id: token,
                logit: 1.0,
                p: 0.0,
            };
            let mut single_arr = ffi::llama_token_data_array {
                data: std::ptr::addr_of_mut!(single),
                size: 1,
                selected: -1,
                sorted: false,
            };

            unsafe {
                // SAFETY: grammar sampler and token array are valid.
                ffi::llama_sampler_apply(grammar_ptr, &mut single_arr);
            }

            if single.logit == f32::NEG_INFINITY {
                let logits_retry = unsafe {
                    // SAFETY: context pointer is valid during generation.
                    ffi::llama_get_logits_ith(ctx, idx)
                };
                if logits_retry.is_null() {
                    return Err(RuntimeError::Inference(format!(
                        "missing logits for output index {idx} on grammar retry"
                    )));
                }

                let mut cur_retry = logits_to_candidates(logits_retry, n_vocab as usize);
                let mut cur_p_retry = ffi::llama_token_data_array {
                    data: cur_retry.as_mut_ptr(),
                    size: cur_retry.len(),
                    selected: -1,
                    sorted: false,
                };

                unsafe {
                    // SAFETY: samplers and candidate array are valid.
                    ffi::llama_sampler_apply(grammar_ptr, &mut cur_p_retry);
                    ffi::llama_sampler_apply(self.chain_ptr, &mut cur_p_retry);
                }
                token = selected_token(&cur_p_retry)?;
            }
        }

        Ok(token)
    }

    fn accept(&mut self, token: ffi::llama_token) {
        unsafe {
            // SAFETY: sampler pointers are valid for the chain lifetime.
            if let Some(grammar_ptr) = self.grammar_ptr {
                ffi::llama_sampler_accept(grammar_ptr, token);
            }
            ffi::llama_sampler_accept(self.chain_ptr, token);
        }
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: chain pointer is owned by this handle and dropped once.
            if let Some(grammar_ptr) = self.grammar_ptr.take() {
                ffi::llama_sampler_free(grammar_ptr);
            }
            if !self.chain_ptr.is_null() {
                ffi::llama_sampler_free(self.chain_ptr);
                self.chain_ptr = std::ptr::null_mut();
            }
        }
    }
}

#[derive(Debug)]
struct BatchHandle {
    batch: ffi::llama_batch,
}

fn logits_to_candidates(logits: *const f32, n_vocab: usize) -> Vec<ffi::llama_token_data> {
    let logits = unsafe {
        // SAFETY: caller guarantees `logits` points to `n_vocab` float values.
        slice::from_raw_parts(logits, n_vocab)
    };
    let mut out = Vec::with_capacity(n_vocab);
    for (id, &logit) in logits.iter().enumerate() {
        out.push(ffi::llama_token_data {
            id: id as ffi::llama_token,
            logit,
            p: 0.0,
        });
    }
    out
}

fn selected_token(cur_p: &ffi::llama_token_data_array) -> Result<ffi::llama_token, RuntimeError> {
    if cur_p.selected < 0 || (cur_p.selected as usize) >= cur_p.size {
        return Err(RuntimeError::Inference(
            "no selected token during sampling".to_string(),
        ));
    }
    let token = unsafe {
        // SAFETY: selected index was bounds-checked above.
        (*cur_p.data.add(cur_p.selected as usize)).id
    };
    Ok(token)
}

impl BatchHandle {
    fn new(n_tokens: usize) -> Result<Self, RuntimeError> {
        let n_tokens_i32 = i32::try_from(n_tokens)
            .map_err(|_| RuntimeError::Inference("batch token count exceeds i32".to_string()))?;

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
            return Err(RuntimeError::Inference(
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
