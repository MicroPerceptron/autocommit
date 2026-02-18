use std::env;
use std::path::PathBuf;

pub const EMBEDDING_MODEL_ENV: &str = "AUTOCOMMIT_EMBED_MODEL";
pub const FALLBACK_MODEL_ENV: &str = "LLAMA_MODEL_PATH";
pub const EMBEDDING_HF_REPO_ENV: &str = "AUTOCOMMIT_EMBED_HF_REPO";
pub const FALLBACK_HF_REPO_ENV: &str = "LLAMA_ARG_HF_REPO";
pub const LLAMA_CACHE_ENV: &str = "LLAMA_CACHE";
pub const DEFAULT_HF_REPO: &str = "ggml-org/gemma-3n-E2B-it-GGUF:Q8_0";

pub fn resolve_embedding_model_path() -> Option<PathBuf> {
    env::var_os(EMBEDDING_MODEL_ENV)
        .or_else(|| env::var_os(FALLBACK_MODEL_ENV))
        .map(PathBuf::from)
}

pub fn resolve_embedding_hf_repo() -> Option<String> {
    env::var(EMBEDDING_HF_REPO_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var(FALLBACK_HF_REPO_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

pub fn resolve_llama_cache_dir() -> Option<PathBuf> {
    env::var_os(LLAMA_CACHE_ENV).map(PathBuf::from)
}
