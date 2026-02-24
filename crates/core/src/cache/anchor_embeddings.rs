use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llm::traits::EmbeddingVector;

#[derive(Debug, Serialize, Deserialize)]
struct AnchorCacheEntry {
    model_fingerprint: String,
    draft_anchor_text_hash: u64,
    full_anchor_text_hash: u64,
    draft_embedding: EmbeddingVector,
    full_embedding: EmbeddingVector,
}

#[derive(Debug)]
pub struct AnchorEmbeddingCache {
    path: PathBuf,
}

pub struct CachedAnchors {
    pub draft: EmbeddingVector,
    pub full: EmbeddingVector,
}

impl AnchorEmbeddingCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            path: dir.into().join("anchor_embeddings.json"),
        }
    }

    pub fn load(
        &self,
        model_fingerprint: &str,
        draft_anchor_text: &str,
        full_anchor_text: &str,
    ) -> Option<CachedAnchors> {
        let bytes = fs::read(&self.path).ok()?;
        let entry: AnchorCacheEntry = serde_json::from_slice(&bytes).ok()?;

        if entry.model_fingerprint != model_fingerprint
            || entry.draft_anchor_text_hash != text_hash(draft_anchor_text)
            || entry.full_anchor_text_hash != text_hash(full_anchor_text)
        {
            return None;
        }

        if entry.draft_embedding.is_empty() || entry.full_embedding.is_empty() {
            return None;
        }

        Some(CachedAnchors {
            draft: entry.draft_embedding,
            full: entry.full_embedding,
        })
    }

    pub fn store(
        &self,
        model_fingerprint: &str,
        draft_anchor_text: &str,
        full_anchor_text: &str,
        draft_embedding: &EmbeddingVector,
        full_embedding: &EmbeddingVector,
    ) {
        let entry = AnchorCacheEntry {
            model_fingerprint: model_fingerprint.to_string(),
            draft_anchor_text_hash: text_hash(draft_anchor_text),
            full_anchor_text_hash: text_hash(full_anchor_text),
            draft_embedding: draft_embedding.clone(),
            full_embedding: full_embedding.clone(),
        };

        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let _ = fs::write(&self.path, bytes);
        }
    }
}

pub fn anchor_cache_from_env() -> Option<AnchorEmbeddingCache> {
    let dir = std::env::var("AUTOCOMMIT_ANCHOR_CACHE_DIR").ok()?;
    if dir.trim().is_empty() {
        return None;
    }
    Some(AnchorEmbeddingCache::new(dir))
}

pub fn anchor_cache_from_path(path: &Path) -> AnchorEmbeddingCache {
    AnchorEmbeddingCache::new(path)
}

fn text_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn round_trip_cache() {
        let dir = std::env::temp_dir().join("autocommit_anchor_test");
        let _ = fs::remove_dir_all(&dir);
        let cache = AnchorEmbeddingCache::new(&dir);

        let draft = vec![0.1, 0.2, 0.3];
        let full = vec![0.4, 0.5, 0.6];

        cache.store("model_abc", "draft text", "full text", &draft, &full);

        let loaded = cache
            .load("model_abc", "draft text", "full text")
            .expect("should load");
        assert_eq!(loaded.draft, draft);
        assert_eq!(loaded.full, full);

        // Different model fingerprint should miss.
        assert!(cache.load("model_xyz", "draft text", "full text").is_none());

        // Different anchor text should miss.
        assert!(cache.load("model_abc", "changed", "full text").is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
