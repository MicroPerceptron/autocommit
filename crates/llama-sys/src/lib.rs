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
        pub fn autocommit_common_config_set_n_parallel(cfg: *mut c_void, n_parallel: i32);

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
    }
}
