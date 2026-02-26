use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use llama_sys::{bridge, ffi};

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
    context_shift_enabled: bool,
    context_shift_n_keep: i32,
    common: CommonParamsBridge,
}

// SAFETY: ModelHandle wraps an immutable llama_model after construction. All mutations happen
// through context objects, and model lifetime is managed by Arc + Drop.
unsafe impl Send for ModelHandle {}
// SAFETY: read-only model access is thread-safe for shared ownership in this crate design.
unsafe impl Sync for ModelHandle {}

impl ModelHandle {
    pub(crate) fn load(
        model_path: Option<&Path>,
        model_hf_repo: Option<&str>,
        model_cache_dir: Option<&Path>,
        cpu_only: bool,
    ) -> Result<Self, RuntimeError> {
        let mut common = CommonParamsBridge::new()?;
        common.set_n_parallel(1);
        common.apply_env()?;
        match model_path {
            Some(path) => common.set_model_path(path)?,
            None => {
                let repo = model_hf_repo.ok_or_else(|| {
                    RuntimeError::Embed(
                        "runtime model source is not configured (local path or HF repo required)"
                            .to_string(),
                    )
                })?;
                common.set_hf_repo(repo)?;
            }
        }
        if let Some(cache_dir) = model_cache_dir {
            common.set_cache_dir(cache_dir)?;
        }
        let model_path = common.resolve_model_path()?;
        if !model_path.exists() {
            return Err(RuntimeError::Embed(format!(
                "embedding model not found after resolution: {}",
                model_path.display()
            )));
        }
        let path_cstr = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::Embed(format!(
                "embedding model path contains interior NUL: {}",
                model_path.display()
            ))
        })?;
        let (mut mparams, mut cparams) = common.export_llama_params()?;

        if cpu_only {
            mparams.n_gpu_layers = 0;
            mparams.split_mode = ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1;
            cparams.offload_kqv = false;
            cparams.op_offload = false;
            cparams.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        let mut legacy_requested_devices = resolve_legacy_requested_devices()?;
        if let Some(devices) = legacy_requested_devices.as_mut() {
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

        common.fill_fit_buffers(&mut tensor_split, &mut tensor_buft_overrides, &mut margins)?;

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
            context_shift_enabled: common.context_shift_enabled(),
            context_shift_n_keep: legacy_keep_tokens_override(common.context_shift_n_keep()),
            common,
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

    pub(crate) fn context_shift_enabled(&self) -> bool {
        legacy_context_shift_override(self.context_shift_enabled)
    }

    pub(crate) fn context_shift_keep_tokens(
        &self,
        prompt_tokens: usize,
        n_ctx_seq: usize,
    ) -> usize {
        let keep = if self.context_shift_n_keep < 0 {
            prompt_tokens
        } else {
            usize::try_from(self.context_shift_n_keep)
                .unwrap_or(usize::MAX)
                .min(prompt_tokens)
        };
        keep.min(n_ctx_seq.saturating_sub(4))
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

    pub(crate) fn common_config_ptr(&self) -> *const std::ffi::c_void {
        self.common.as_ptr()
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

pub(crate) fn list_cached_models(
    cache_dir: Option<&Path>,
) -> Result<(PathBuf, Vec<String>), RuntimeError> {
    let mut common = CommonParamsBridge::new()?;
    if let Some(cache_dir) = cache_dir {
        common.set_cache_dir(cache_dir)?;
    }
    common.list_cached_models()
}

#[derive(Debug)]
struct CommonParamsBridge {
    ptr: *mut std::ffi::c_void,
}

impl CommonParamsBridge {
    fn new() -> Result<Self, RuntimeError> {
        let ptr = unsafe {
            // SAFETY: constructor is pure and returns a new owned bridge handle or null on failure.
            bridge::autocommit_common_config_new()
        };
        if ptr.is_null() {
            return Err(RuntimeError::Embed(
                "failed to allocate common_params bridge".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    fn set_model_path(&mut self, model_path: &Path) -> Result<(), RuntimeError> {
        let model_path = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::Embed(format!(
                "embedding model path contains interior NUL: {}",
                model_path.display()
            ))
        })?;
        unsafe {
            // SAFETY: bridge handle is valid and model_path is a live NUL-terminated string.
            bridge::autocommit_common_config_set_model_path(self.ptr, model_path.as_ptr());
        }
        Ok(())
    }

    fn set_hf_repo(&mut self, repo: &str) -> Result<(), RuntimeError> {
        let repo = CString::new(repo.as_bytes()).map_err(|_| {
            RuntimeError::Embed(format!("embedding HF repo contains interior NUL: {repo}"))
        })?;
        unsafe {
            // SAFETY: bridge handle is valid and repo is a live NUL-terminated string.
            bridge::autocommit_common_config_set_hf_repo(self.ptr, repo.as_ptr());
        }
        Ok(())
    }

    fn set_cache_dir(&mut self, cache_dir: &Path) -> Result<(), RuntimeError> {
        let cache_dir = CString::new(cache_dir.to_string_lossy().as_bytes()).map_err(|_| {
            RuntimeError::Embed(format!(
                "model cache dir contains interior NUL: {}",
                cache_dir.display()
            ))
        })?;
        unsafe {
            // SAFETY: bridge handle is valid and cache_dir is a live NUL-terminated string.
            bridge::autocommit_common_config_set_cache_dir(self.ptr, cache_dir.as_ptr());
        }
        Ok(())
    }

    fn resolve_model_path(&mut self) -> Result<std::path::PathBuf, RuntimeError> {
        let mut path_buf = vec![0i8; 1024];
        let mut err_buf = vec![0i8; 512];
        let ok = unsafe {
            // SAFETY: bridge handle and output buffers are valid.
            bridge::autocommit_common_config_resolve_model_path(
                self.ptr,
                path_buf.as_mut_ptr(),
                path_buf.len(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if ok == 0 {
            return Err(RuntimeError::Embed(format!(
                "failed to resolve model path: {}",
                err_buf_to_string(&err_buf)
            )));
        }
        let path = err_buf_to_string(&path_buf);
        if path.trim().is_empty() {
            return Err(RuntimeError::Embed(
                "common_params returned an empty model path".to_string(),
            ));
        }
        Ok(std::path::PathBuf::from(path))
    }

    fn list_cached_models(&mut self) -> Result<(PathBuf, Vec<String>), RuntimeError> {
        let mut models_buf = vec![0i8; 64 * 1024];
        let mut cache_dir_buf = vec![0i8; 2048];
        let mut err_buf = vec![0i8; 512];
        let ok = unsafe {
            // SAFETY: bridge handle and output buffers are valid.
            bridge::autocommit_common_config_list_cached_models(
                self.ptr,
                models_buf.as_mut_ptr(),
                models_buf.len(),
                cache_dir_buf.as_mut_ptr(),
                cache_dir_buf.len(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if ok == 0 {
            return Err(RuntimeError::Embed(format!(
                "failed to list cached models: {}",
                err_buf_to_string(&err_buf)
            )));
        }

        let cache_dir = err_buf_to_string(&cache_dir_buf);
        if cache_dir.trim().is_empty() {
            return Err(RuntimeError::Embed(
                "common_params returned an empty cache directory".to_string(),
            ));
        }

        let models = err_buf_to_string(&models_buf)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        Ok((PathBuf::from(cache_dir), models))
    }

    fn set_n_parallel(&mut self, n_parallel: i32) {
        unsafe {
            // SAFETY: bridge handle is valid and the setter is side-effect free outside the handle.
            bridge::autocommit_common_config_set_n_parallel(self.ptr, n_parallel);
        }
    }

    fn apply_env(&mut self) -> Result<(), RuntimeError> {
        let mut err_buf = vec![0i8; 512];
        let ok = unsafe {
            // SAFETY: bridge handle and output buffer are valid.
            bridge::autocommit_common_config_apply_env(
                self.ptr,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if ok == 0 {
            return Err(RuntimeError::Embed(format!(
                "failed to apply common_params env options: {}",
                err_buf_to_string(&err_buf)
            )));
        }
        Ok(())
    }

    fn export_llama_params(
        &mut self,
    ) -> Result<(ffi::llama_model_params, ffi::llama_context_params), RuntimeError> {
        let mut mparams = unsafe {
            // SAFETY: pure default parameter initialization.
            ffi::llama_model_default_params()
        };
        let mut cparams = unsafe {
            // SAFETY: pure default parameter initialization.
            ffi::llama_context_default_params()
        };
        let mut err_buf = vec![0i8; 512];
        let ok = unsafe {
            // SAFETY: bridge handle and output parameter pointers are valid.
            bridge::autocommit_common_config_export_llama_params(
                self.ptr,
                &mut mparams,
                &mut cparams,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if ok == 0 {
            return Err(RuntimeError::Embed(format!(
                "failed to export llama params from common_params: {}",
                err_buf_to_string(&err_buf)
            )));
        }
        Ok((mparams, cparams))
    }

    fn fill_fit_buffers(
        &mut self,
        tensor_split: &mut [f32],
        tensor_buft_overrides: &mut [ffi::llama_model_tensor_buft_override],
        margins: &mut [usize],
    ) -> Result<(), RuntimeError> {
        let mut err_buf = vec![0i8; 512];
        let ok = unsafe {
            // SAFETY: bridge handle and all mutable slices are valid buffers.
            bridge::autocommit_common_config_fill_fit_buffers(
                self.ptr,
                tensor_split.as_mut_ptr(),
                tensor_split.len(),
                tensor_buft_overrides.as_mut_ptr(),
                tensor_buft_overrides.len(),
                margins.as_mut_ptr(),
                margins.len(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if ok == 0 {
            return Err(RuntimeError::Embed(format!(
                "failed to prepare llama_params_fit buffers from common_params: {}",
                err_buf_to_string(&err_buf)
            )));
        }
        Ok(())
    }

    fn context_shift_enabled(&self) -> bool {
        let enabled = unsafe {
            // SAFETY: bridge handle is valid for the lifetime of this wrapper.
            bridge::autocommit_common_config_ctx_shift_enabled(self.ptr)
        };
        enabled != 0
    }

    fn context_shift_n_keep(&self) -> i32 {
        unsafe {
            // SAFETY: bridge handle is valid for the lifetime of this wrapper.
            bridge::autocommit_common_config_n_keep(self.ptr)
        }
    }
}

impl Drop for CommonParamsBridge {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: pointer is owned by this wrapper and must be freed exactly once.
            bridge::autocommit_common_config_free(self.ptr);
        }
    }
}

fn err_buf_to_string(err_buf: &[i8]) -> String {
    unsafe {
        // SAFETY: bridge writes a NUL-terminated C string into err_buf on failure.
        CStr::from_ptr(err_buf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

fn legacy_context_shift_override(common_default: bool) -> bool {
    match std::env::var("AUTOCOMMIT_CTX_SHIFT")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => true,
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => false,
        _ => common_default,
    }
}

fn legacy_keep_tokens_override(common_default: i32) -> i32 {
    std::env::var("AUTOCOMMIT_CTX_KEEP")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(common_default)
}

fn resolve_legacy_requested_devices() -> Result<Option<Vec<ffi::ggml_backend_dev_t>>, RuntimeError>
{
    if std::env::var("LLAMA_ARG_DEVICE").is_ok() {
        return Ok(None);
    }
    let requested = std::env::var("AUTOCOMMIT_LLAMA_DEVICE").ok();
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
            "AUTOCOMMIT_LLAMA_DEVICE is empty".to_string(),
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
                "invalid device '{target}' in AUTOCOMMIT_LLAMA_DEVICE (available: {choices})"
            )));
        };
        out.push(*dev);
    }
    out.push(std::ptr::null_mut());

    Ok(Some(out))
}
