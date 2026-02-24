#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod bridge {
    use std::ffi::{c_char, c_int, c_void};

    use crate::ffi;

    unsafe extern "C" {
        pub fn autocommit_common_config_new() -> *mut c_void;
        pub fn autocommit_common_config_free(cfg: *mut c_void);

        pub fn autocommit_common_config_set_model_path(cfg: *mut c_void, path: *const c_char);
        pub fn autocommit_common_config_set_hf_repo(cfg: *mut c_void, repo: *const c_char);
        pub fn autocommit_common_config_set_cache_dir(cfg: *mut c_void, dir: *const c_char);
        pub fn autocommit_common_config_set_n_parallel(cfg: *mut c_void, n_parallel: i32);
        pub fn autocommit_common_config_resolve_model_path(
            cfg: *mut c_void,
            out_path: *mut c_char,
            out_path_len: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;
        pub fn autocommit_common_config_list_cached_models(
            cfg: *mut c_void,
            out_models: *mut c_char,
            out_models_len: usize,
            out_cache_dir: *mut c_char,
            out_cache_dir_len: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;

        pub fn autocommit_common_config_apply_env(
            cfg: *mut c_void,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;

        pub fn autocommit_common_config_export_llama_params(
            cfg: *mut c_void,
            mparams: *mut ffi::llama_model_params,
            cparams: *mut ffi::llama_context_params,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;

        pub fn autocommit_common_config_fill_fit_buffers(
            cfg: *mut c_void,
            tensor_split: *mut f32,
            tensor_split_len: usize,
            tensor_buft_overrides: *mut ffi::llama_model_tensor_buft_override,
            tensor_buft_overrides_len: usize,
            margins: *mut usize,
            margins_len: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;

        pub fn autocommit_common_config_ctx_shift_enabled(cfg: *const c_void) -> c_int;
        pub fn autocommit_common_config_n_keep(cfg: *const c_void) -> i32;

        pub fn autocommit_common_sampler_new(
            cfg: *const c_void,
            model: *mut ffi::llama_model,
            grammar: *const c_char,
            grammar_lazy: c_int,
        ) -> *mut c_void;
        pub fn autocommit_common_sampler_clone(sampler: *mut c_void) -> *mut c_void;
        pub fn autocommit_common_sampler_free(sampler: *mut c_void);
        pub fn autocommit_common_sampler_sample(
            sampler: *mut c_void,
            ctx: *mut ffi::llama_context,
            idx: c_int,
            grammar_first: c_int,
        ) -> ffi::llama_token;
        pub fn autocommit_common_sampler_accept(
            sampler: *mut c_void,
            token: ffi::llama_token,
            accept_grammar: c_int,
        );
        pub fn autocommit_common_sampler_reset(sampler: *mut c_void);

        pub fn autocommit_cosine_similarity(a: *const f32, b: *const f32, n: c_int) -> f32;

        pub fn autocommit_common_log_set_verbosity(verbosity: c_int);
    }
}
