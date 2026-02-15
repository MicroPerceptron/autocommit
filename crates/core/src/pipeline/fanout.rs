use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::CoreError;
use crate::llm::traits::LlmEngine;
use crate::types::{DiffChunk, PartialReport};

pub fn analyze_chunks(
    engine: &dyn LlmEngine,
    chunks: &[DiffChunk],
) -> Result<Vec<PartialReport>, CoreError> {
    let cache = PartialCache::new_from_env();
    if let Some(cache) = cache.as_ref() {
        let mut cached = Vec::with_capacity(chunks.len());
        let mut missing = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            if let Some(report) = cache.load(chunk) {
                cached.push(Some(report));
            } else {
                cached.push(None);
                missing.push(idx);
            }
        }
        if missing.is_empty() {
            return Ok(cached.into_iter().map(|r| r.expect("cached")).collect());
        }

        let analyzed = if let Some(result) = engine.analyze_chunks_batched(
            &missing
                .iter()
                .map(|&idx| &chunks[idx])
                .cloned()
                .collect::<Vec<_>>(),
        ) {
            result?
        } else {
            missing
                .iter()
                .map(|&idx| engine.analyze_chunk(&chunks[idx]))
                .collect::<Result<Vec<_>, _>>()?
        };

        for (slot, report) in missing.into_iter().zip(analyzed.into_iter()) {
            cached[slot] = Some(report.clone());
            cache.store(&chunks[slot], &report);
        }

        return Ok(cached.into_iter().map(|r| r.expect("filled")).collect());
    }

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

struct PartialCache {
    root: PathBuf,
}

impl PartialCache {
    fn new_from_env() -> Option<Self> {
        let root = std::env::var("AUTOCOMMIT_PARTIAL_CACHE_DIR").ok()?;
        if root.trim().is_empty() {
            return None;
        }
        Some(Self {
            root: PathBuf::from(root),
        })
    }

    fn load(&self, chunk: &DiffChunk) -> Option<PartialReport> {
        let path = self.entry_path(chunk);
        let bytes = std::fs::read(path).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    fn store(&self, chunk: &DiffChunk, report: &PartialReport) {
        let path = self.entry_path(chunk);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(report) {
            let _ = std::fs::write(path, bytes);
        }
    }

    fn entry_path(&self, chunk: &DiffChunk) -> PathBuf {
        let key = chunk_cache_key(chunk);
        self.root.join(format!("{key}.json"))
    }
}

fn chunk_cache_key(chunk: &DiffChunk) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    "v1".hash(&mut hasher);
    chunk.path.hash(&mut hasher);
    chunk.text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
