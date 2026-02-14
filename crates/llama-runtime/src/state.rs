use std::fs;
use std::path::Path;

use crate::error::RuntimeError;

pub fn save_state(path: &Path, bytes: &[u8]) -> Result<(), RuntimeError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| RuntimeError::State(err.to_string()))?;
    }
    fs::write(path, bytes).map_err(|err| RuntimeError::State(err.to_string()))
}

pub fn load_state(path: &Path) -> Result<Vec<u8>, RuntimeError> {
    fs::read(path).map_err(|err| RuntimeError::State(err.to_string()))
}
