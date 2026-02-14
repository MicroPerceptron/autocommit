use std::env;
use std::path::PathBuf;

pub const EMBEDDING_MODEL_ENV: &str = "AUTOCOMMIT_EMBED_MODEL";
pub const FALLBACK_MODEL_ENV: &str = "LLAMA_MODEL_PATH";

pub fn resolve_embedding_model_path() -> Option<PathBuf> {
    env::var_os(EMBEDDING_MODEL_ENV)
        .or_else(|| env::var_os(FALLBACK_MODEL_ENV))
        .map(PathBuf::from)
}
