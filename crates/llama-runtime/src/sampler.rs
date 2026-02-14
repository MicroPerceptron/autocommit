#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.9,
            top_k: 40,
        }
    }
}
