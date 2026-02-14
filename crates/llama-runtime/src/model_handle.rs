use std::ffi::{CStr, CString};
use std::path::Path;

use llama_sys::ffi;

use crate::error::RuntimeError;

const CHATML_TEMPLATE_FALLBACK: &str = concat!(
    "{%- for message in messages -%}\n",
    "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n",
    "{%- endfor -%}\n",
    "{%- if add_generation_prompt -%}\n",
    "  {{- '<|im_start|>assistant\\n' -}}\n",
    "{%- endif -%}"
);

#[derive(Debug)]
pub(crate) struct ModelHandle {
    ptr: *mut ffi::llama_model,
    vocab: *const ffi::llama_vocab,
    n_embd: usize,
    base_context_params: ffi::llama_context_params,
    has_encoder: bool,
    has_decoder: bool,
}

// SAFETY: ModelHandle wraps an immutable llama_model after construction. All mutations happen
// through context objects, and model lifetime is managed by Arc + Drop.
unsafe impl Send for ModelHandle {}
// SAFETY: read-only model access is thread-safe for shared ownership in this crate design.
unsafe impl Sync for ModelHandle {}

impl ModelHandle {
    pub(crate) fn load(model_path: &Path, cpu_only: bool) -> Result<Self, RuntimeError> {
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

        let mut mparams = unsafe {
            // SAFETY: pure default parameter initialization.
            ffi::llama_model_default_params()
        };
        let mut cparams = unsafe {
            // SAFETY: pure default parameter initialization.
            ffi::llama_context_default_params()
        };

        // Match llama.cpp common_params defaults for model/context setup.
        mparams.n_gpu_layers = -1;
        cparams.n_ctx = 0;
        cparams.n_seq_max = 1;
        cparams.n_batch = 2048;
        cparams.n_ubatch = 512;

        if cpu_only {
            mparams.n_gpu_layers = 0;
            mparams.split_mode = ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1;
            cparams.offload_kqv = false;
            cparams.op_offload = false;
            cparams.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        let mut requested_devices =
            resolve_requested_devices().map_err(|err| RuntimeError::Embed(err.to_string()))?;
        if let Some(devices) = requested_devices.as_mut() {
            mparams.devices = devices.as_mut_ptr();
        }

        let max_devices = unsafe {
            // SAFETY: pure metadata query.
            ffi::llama_max_devices()
        };
        let mut tensor_split = vec![0f32; max_devices.max(1)];
        let mut margins = vec![1024usize * 1024 * 1024; max_devices.max(1)];
        let max_overrides = unsafe {
            // SAFETY: pure metadata query.
            ffi::llama_max_tensor_buft_overrides()
        };
        let mut tensor_buft_overrides = vec![
            ffi::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            };
            max_overrides.max(1)
        ];

        mparams.tensor_split = tensor_split.as_ptr();
        mparams.tensor_buft_overrides = tensor_buft_overrides.as_ptr();

        let fit_status = unsafe {
            // SAFETY: pointers refer to valid writable buffers for the duration of this call.
            ffi::llama_params_fit(
                path_cstr.as_ptr(),
                &mut mparams,
                &mut cparams,
                tensor_split.as_mut_ptr(),
                tensor_buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                4096,
                ffi::ggml_log_level_GGML_LOG_LEVEL_ERROR,
            )
        };
        if fit_status != ffi::llama_params_fit_status_LLAMA_PARAMS_FIT_STATUS_SUCCESS
            && std::env::var("AUTOCOMMIT_LLAMA_LOG")
                .ok()
                .as_deref()
                .map(|v| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false)
        {
            eprintln!(
                "autocommit warning: llama_params_fit returned status={} for {}",
                fit_status,
                model_path.display()
            );
        }

        let ptr = unsafe {
            // SAFETY: FFI call with fitted params and a valid NUL-terminated path.
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
            base_context_params: cparams,
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

    pub(crate) fn context_params(&self) -> ffi::llama_context_params {
        self.base_context_params
    }

    pub(crate) fn apply_chat_template(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
    ) -> Option<String> {
        let template_ptr = unsafe {
            // SAFETY: model pointer is valid while ModelHandle owns it.
            ffi::llama_model_chat_template(self.ptr, std::ptr::null())
        };
        let template_override = CString::new(CHATML_TEMPLATE_FALLBACK).ok()?;
        let use_fallback = if template_ptr.is_null() {
            true
        } else {
            let template_name = unsafe {
                // SAFETY: template_ptr is non-null and points to a NUL-terminated string.
                CStr::from_ptr(template_ptr).to_string_lossy()
            };
            template_name == "chatml"
        };
        let template_ptr = if use_fallback {
            template_override.as_ptr()
        } else {
            template_ptr
        };

        let mut chat = Vec::with_capacity(2);
        let mut roles = Vec::with_capacity(2);
        let mut contents = Vec::with_capacity(2);

        if let Some(system_prompt) = system_prompt.filter(|s| !s.trim().is_empty()) {
            roles.push(CString::new("system").ok()?);
            contents.push(CString::new(system_prompt).ok()?);
            chat.push(ffi::llama_chat_message {
                role: roles[roles.len() - 1].as_ptr(),
                content: contents[contents.len() - 1].as_ptr(),
            });
        }

        roles.push(CString::new("user").ok()?);
        contents.push(CString::new(user_prompt).ok()?);
        chat.push(ffi::llama_chat_message {
            role: roles[roles.len() - 1].as_ptr(),
            content: contents[contents.len() - 1].as_ptr(),
        });

        let mut capacity = user_prompt
            .len()
            .saturating_add(system_prompt.map_or(0, str::len))
            .saturating_mul(4)
            .saturating_add(2048)
            .max(4096);
        for _ in 0..3 {
            let mut buf = vec![0u8; capacity];
            let written = unsafe {
                // SAFETY: pointers are valid for call duration and output buffer is writable.
                ffi::llama_chat_apply_template(
                    template_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    true,
                    buf.as_mut_ptr() as *mut i8,
                    i32::try_from(buf.len()).ok()?,
                )
            };

            if written <= 0 {
                return None;
            }

            let written_usize = usize::try_from(written).ok()?;
            if written_usize < buf.len() {
                buf.truncate(written_usize);
                return String::from_utf8(buf).ok();
            }

            capacity = written_usize.saturating_add(1);
        }

        None
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

fn resolve_requested_devices() -> Result<Option<Vec<ffi::ggml_backend_dev_t>>, RuntimeError> {
    let requested = std::env::var("LLAMA_ARG_DEVICE")
        .ok()
        .or_else(|| std::env::var("AUTOCOMMIT_LLAMA_DEVICE").ok());
    let Some(raw) = requested else {
        return Ok(None);
    };

    let entries = raw
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err(RuntimeError::Embed(
            "LLAMA_ARG_DEVICE/AUTOCOMMIT_LLAMA_DEVICE is empty".to_string(),
        ));
    }

    if entries.len() == 1 && entries[0].eq_ignore_ascii_case("none") {
        return Ok(Some(vec![std::ptr::null_mut()]));
    }

    let mut available = Vec::new();
    let count = unsafe {
        // SAFETY: pure backend metadata query.
        ffi::ggml_backend_dev_count()
    };
    for idx in 0..count {
        let dev = unsafe {
            // SAFETY: idx is bounded by ggml_backend_dev_count.
            ffi::ggml_backend_dev_get(idx)
        };
        if dev.is_null() {
            continue;
        }
        let name = unsafe {
            // SAFETY: ggml provides static NUL-terminated strings.
            CStr::from_ptr(ffi::ggml_backend_dev_name(dev))
                .to_string_lossy()
                .into_owned()
        };
        let desc = unsafe {
            // SAFETY: ggml provides static NUL-terminated strings.
            CStr::from_ptr(ffi::ggml_backend_dev_description(dev))
                .to_string_lossy()
                .into_owned()
        };
        available.push((name, desc, dev));
    }

    let mut out = Vec::with_capacity(entries.len() + 1);
    for target in entries {
        let matched = available.iter().find(|(name, desc, _)| {
            name.eq_ignore_ascii_case(target) || desc.eq_ignore_ascii_case(target)
        });
        let Some((_, _, dev)) = matched else {
            let choices = available
                .iter()
                .map(|(name, _, _)| {
                    if name.is_empty() {
                        "<unnamed>".to_string()
                    } else {
                        name.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            return Err(RuntimeError::Embed(format!(
                "invalid device '{target}' in LLAMA_ARG_DEVICE/AUTOCOMMIT_LLAMA_DEVICE (available: {choices})"
            )));
        };
        out.push(*dev);
    }
    out.push(std::ptr::null_mut());

    Ok(Some(out))
}
