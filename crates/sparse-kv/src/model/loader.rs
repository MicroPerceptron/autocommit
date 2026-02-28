use std::path::Path;

use crate::error::InferenceError;
use crate::gguf::GgufReader;
use crate::model::ModelConfig;

pub fn load(path: &Path) -> Result<(ModelConfig, GgufReader), InferenceError> {
    let reader = GgufReader::open(path)?;
    let config = ModelConfig::from_gguf_metadata(&reader.metadata)?;
    Ok((config, reader))
}
