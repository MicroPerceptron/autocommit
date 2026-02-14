use autocommit_core::types::DiffChunk;

#[derive(Debug, Clone)]
pub struct DiffBatch {
    pub chunks: Vec<DiffChunk>,
}

impl DiffBatch {
    pub fn from_chunks(chunks: &[DiffChunk]) -> Self {
        Self {
            chunks: chunks.to_vec(),
        }
    }
}
