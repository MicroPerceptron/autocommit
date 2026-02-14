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
    n_ctx_train: usize,
    has_gpu_devices: bool,
    max_gpu_mem_mib: usize,
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

        let mut selected_devices = select_model_devices(cpu_only);
        let (using_gpu_devices, max_gpu_mem_mib) = detect_gpu_devices(cpu_only);
        if llama_logs_enabled() {
            eprintln!("autocommit device inventory: {}", device_inventory_summary());
            if let Some(devs) = selected_devices.as_ref() {
                eprintln!("autocommit selected devices: {}", devs.len().saturating_sub(1));
            } else {
                eprintln!("autocommit selected devices: auto");
            }
            if !cpu_only && !using_gpu_devices {
                eprintln!(
                    "autocommit note: no usable GPU device metadata found; forcing CPU model placement"
                );
            }
        }

        let ptr = unsafe {
            // SAFETY: FFI call with default params and a valid NUL-terminated path.
            let mut mparams = ffi::llama_model_default_params();
            if let Some(devices) = selected_devices.as_mut() {
                mparams.devices = devices.as_mut_ptr();
            }
            if cpu_only || !using_gpu_devices {
                mparams.n_gpu_layers = 0;
                mparams.split_mode = ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE;
                mparams.main_gpu = -1;
                mparams.use_extra_bufts = false;
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

        let n_ctx_train = unsafe {
            // SAFETY: model pointer is valid while ModelHandle owns it.
            ffi::llama_model_n_ctx_train(ptr)
        };
        let n_ctx_train = if n_ctx_train > 0 {
            n_ctx_train as usize
        } else {
            4096
        };

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
            n_ctx_train,
            has_gpu_devices: using_gpu_devices,
            max_gpu_mem_mib,
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

    pub(crate) fn n_ctx_train(&self) -> usize {
        self.n_ctx_train
    }

    pub(crate) fn has_gpu_devices(&self) -> bool {
        self.has_gpu_devices
    }

    pub(crate) fn max_gpu_mem_mib(&self) -> usize {
        self.max_gpu_mem_mib
    }

    pub(crate) fn has_encoder(&self) -> bool {
        self.has_encoder
    }

    pub(crate) fn has_decoder(&self) -> bool {
        self.has_decoder
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

fn llama_logs_enabled() -> bool {
    std::env::var("AUTOCOMMIT_LLAMA_LOG")
        .ok()
        .as_deref()
        .map(|v| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn select_model_devices(cpu_only: bool) -> Option<Vec<ffi::ggml_backend_dev_t>> {
    if cpu_only {
        return Some(vec![std::ptr::null_mut()]);
    }

    // Keep auto backend selection behavior aligned with llama-cli.
    None
}

fn detect_gpu_devices(cpu_only: bool) -> (bool, usize) {
    if cpu_only {
        return (false, 0);
    }

    let count = unsafe {
        // SAFETY: pure backend metadata query.
        ffi::ggml_backend_dev_count()
    };

    let mut has_gpu = false;
    let mut max_gpu_mem_mib = 0usize;

    for idx in 0..count {
        let dev = unsafe {
            // SAFETY: index is bounded by device count.
            ffi::ggml_backend_dev_get(idx)
        };
        if dev.is_null() {
            continue;
        }

        let dev_type = unsafe {
            // SAFETY: device pointer comes from ggml registry.
            ffi::ggml_backend_dev_type(dev)
        };
        let is_gpu = dev_type == ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
            || dev_type == ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU;
        if !is_gpu {
            continue;
        }

        let (name, desc) = device_name_desc(dev);

        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            // SAFETY: device pointer is valid and output pointers are writable.
            ffi::ggml_backend_dev_memory(dev, &mut free, &mut total);
        }
        let usable = !name.trim().is_empty() || !desc.trim().is_empty() || total > 0 || free > 0;
        if !usable {
            continue;
        }

        has_gpu = true;
        max_gpu_mem_mib = max_gpu_mem_mib.max(total / (1024 * 1024));
    }

    (has_gpu, max_gpu_mem_mib)
}

fn device_name_desc(dev: ffi::ggml_backend_dev_t) -> (String, String) {
    let name = unsafe {
        // SAFETY: backend returns a NUL-terminated string; null ptr is handled.
        let ptr = ffi::ggml_backend_dev_name(dev);
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let desc = unsafe {
        // SAFETY: backend returns a NUL-terminated string; null ptr is handled.
        let ptr = ffi::ggml_backend_dev_description(dev);
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    (name, desc)
}

fn device_inventory_summary() -> String {
    let mut items = Vec::new();
    let count = unsafe {
        // SAFETY: pure backend metadata query.
        ffi::ggml_backend_dev_count()
    };
    for idx in 0..count {
        let dev = unsafe {
            // SAFETY: index is bounded by device count.
            ffi::ggml_backend_dev_get(idx)
        };
        if dev.is_null() {
            continue;
        }

        let dev_type = unsafe {
            // SAFETY: device pointer comes from ggml registry.
            ffi::ggml_backend_dev_type(dev)
        };
        let (name, desc) = device_name_desc(dev);
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            // SAFETY: device pointer is valid and output pointers are writable.
            ffi::ggml_backend_dev_memory(dev, &mut free, &mut total);
        }
        items.push(format!(
            "#{idx}:type={dev_type},name={name},desc={desc},free_mib={},total_mib={}",
            free / (1024 * 1024),
            total / (1024 * 1024)
        ));
    }

    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(" | ")
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
