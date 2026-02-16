use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use autocommit_core::AnalysisReport;
use serde::{Deserialize, Serialize};

const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_KEY_VERSION: &str = "2.3";
const CACHE_DIR: &str = "autocommit/kv";
const REPORT_CACHE_FILE: &str = "report_cache.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReportCache {
    schema_version: u32,
    updated_unix_secs: u64,
    entries: Vec<ReportCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReportCacheEntry {
    cache_key: String,
    report: AnalysisReport,
    created_unix_secs: u64,
}

pub(crate) fn cache_key(scope: &str, model_profile: &str, diff_hash: &str) -> String {
    format!("{scope}|{model_profile}|{CACHE_KEY_VERSION}|{diff_hash}")
}

pub(crate) fn read_cached_report(cache_path: &Path, cache_key: &str) -> Option<AnalysisReport> {
    let bytes = fs::read(cache_path).ok()?;
    let mut cache = serde_json::from_slice::<ReportCache>(&bytes).ok()?;
    if cache.schema_version != CACHE_SCHEMA_VERSION {
        return None;
    }
    cache.entries.retain(|entry| entry.cache_key == cache_key);
    cache.entries.first().map(|entry| entry.report.clone())
}

pub(crate) fn write_cached_report(
    cache_path: &Path,
    cache_key: &str,
    report: &AnalysisReport,
) -> Result<(), std::io::Error> {
    let mut cache = load_cache(cache_path);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(0);

    cache.entries.retain(|entry| entry.cache_key != cache_key);
    cache.entries.push(ReportCacheEntry {
        cache_key: cache_key.to_string(),
        report: report.clone(),
        created_unix_secs: now,
    });
    cache.updated_unix_secs = now;

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_vec_pretty(&cache)
        .map_err(|err| std::io::Error::other(format!("serialize report cache failed: {err}")))?;
    fs::write(cache_path, payload)
}

pub(crate) fn cache_path(git_dir: &Path) -> PathBuf {
    git_dir.join(CACHE_DIR).join(REPORT_CACHE_FILE)
}

pub(crate) fn diff_hash(diff_text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    diff_text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn load_cache(path: &Path) -> ReportCache {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => {
            return ReportCache {
                schema_version: CACHE_SCHEMA_VERSION,
                updated_unix_secs: 0,
                entries: Vec::new(),
            };
        }
    };

    let cache = serde_json::from_slice::<ReportCache>(&bytes).unwrap_or_else(|_| ReportCache {
        schema_version: CACHE_SCHEMA_VERSION,
        updated_unix_secs: 0,
        entries: Vec::new(),
    });
    if cache.schema_version == CACHE_SCHEMA_VERSION {
        cache
    } else {
        ReportCache {
            schema_version: CACHE_SCHEMA_VERSION,
            updated_unix_secs: 0,
            entries: Vec::new(),
        }
    }
}
