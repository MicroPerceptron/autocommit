use std::sync::{Arc, Mutex};

use crate::CoreError;
use crate::llm::traits::LlmEngine;
use crate::types::{DiffChunk, PartialReport};

pub fn analyze_chunks(
    engine: &dyn LlmEngine,
    chunks: &[DiffChunk],
) -> Result<Vec<PartialReport>, CoreError> {
    if let Some(result) = engine.analyze_chunks_batched(chunks) {
        return result;
    }

    if chunks.len() <= 1 {
        return chunks
            .iter()
            .map(|chunk| engine.analyze_chunk(chunk))
            .collect();
    }

    let results: Arc<Mutex<Vec<Option<Result<PartialReport, CoreError>>>>> =
        Arc::new(Mutex::new((0..chunks.len()).map(|_| None).collect()));

    std::thread::scope(|scope| {
        for (idx, chunk) in chunks.iter().enumerate() {
            let results = Arc::clone(&results);
            scope.spawn(move || {
                let result = engine.analyze_chunk(chunk);
                if let Ok(mut guard) = results.lock() {
                    guard[idx] = Some(result);
                }
            });
        }
    });

    let mut out = Vec::with_capacity(chunks.len());
    let mut locked = results
        .lock()
        .map_err(|_| CoreError::Engine("fanout lock poisoned".to_string()))?;
    for (idx, slot) in locked.iter_mut().enumerate() {
        let value = slot
            .take()
            .ok_or_else(|| CoreError::Engine(format!("missing fanout result at index {idx}")))??;
        out.push(value);
    }

    Ok(out)
}
