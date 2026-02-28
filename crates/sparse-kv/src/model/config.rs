use crate::error::InferenceError;
use crate::gguf::GgufMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnType {
    SwiGLU,
    GeGLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_layer: usize,
    pub n_ctx_max: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub ffn_type: FfnType,
    pub norm_type: NormType,
}

impl ModelConfig {
    pub fn from_gguf_metadata(meta: &GgufMetadata) -> Result<Self, InferenceError> {
        let n_embd = meta
            .get_u32("llama.embedding_length")
            .or_else(|| meta.get_u32("gpt2.embedding_length"))
            .ok_or_else(|| {
                InferenceError::Load("missing embedding_length in GGUF metadata".into())
            })? as usize;

        let n_head = meta
            .get_u32("llama.attention.head_count")
            .or_else(|| meta.get_u32("gpt2.attention.head_count"))
            .ok_or_else(|| {
                InferenceError::Load("missing attention.head_count in GGUF metadata".into())
            })? as usize;

        let n_kv_head = meta
            .get_u32("llama.attention.head_count_kv")
            .or_else(|| meta.get_u32("gpt2.attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;

        let n_layer = meta
            .get_u32("llama.block_count")
            .or_else(|| meta.get_u32("gpt2.block_count"))
            .ok_or_else(|| {
                InferenceError::Load("missing block_count in GGUF metadata".into())
            })? as usize;

        let n_vocab = meta
            .get_u32("llama.vocab_size")
            .or_else(|| meta.get_u32("gpt2.vocab_size"))
            .unwrap_or(32000) as usize;

        let n_ctx_max = meta
            .get_u32("llama.context_length")
            .or_else(|| meta.get_u32("gpt2.context_length"))
            .unwrap_or(2048) as usize;

        if n_embd % n_head != 0 {
            return Err(InferenceError::Load(format!(
                "n_embd ({n_embd}) is not divisible by n_head ({n_head})"
            )));
        }
        let head_dim = n_embd / n_head;

        let ffn_hidden = meta
            .get_u32("llama.feed_forward_length")
            .or_else(|| meta.get_u32("gpt2.feed_forward_length"))
            .unwrap_or((n_embd as u32) * 4) as usize;

        let rope_theta = meta.get_f32("llama.rope.freq_base").unwrap_or(10000.0);

        let norm_eps = meta
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_kv_head,
            n_layer,
            n_ctx_max,
            head_dim,
            ffn_hidden,
            rope_theta,
            norm_eps,
            ffn_type: FfnType::SwiGLU,
            norm_type: NormType::RmsNorm,
        })
    }
}
