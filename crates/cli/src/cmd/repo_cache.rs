use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use autocommit_core::CoreError;
use serde::{Deserialize, Serialize};

const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_DIR: &str = "autocommit/kv";
const GENERATION_STATE_FILE: &str = "generation.session";
const METADATA_FILE: &str = "metadata.json";

#[derive(Debug, Clone)]
pub(crate) struct RepoKvPaths {
    pub(crate) repo_root: PathBuf,
    pub(crate) git_dir: PathBuf,
    pub(crate) generation_state: PathBuf,
    pub(crate) metadata: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RepoKvMetadata {
    pub(crate) version: u32,
    pub(crate) profile: String,
    pub(crate) model_path: Option<String>,
    pub(crate) created_unix_secs: u64,
}

impl RepoKvMetadata {
    pub(crate) fn new(profile: &str, model_path: Option<&Path>) -> Self {
        let created_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|dur| dur.as_secs())
            .unwrap_or(0);

        Self {
            version: CACHE_SCHEMA_VERSION,
            profile: profile.to_string(),
            model_path: model_path.map(|path| path.to_string_lossy().into_owned()),
            created_unix_secs,
        }
    }
}

pub(crate) fn discover_repo_kv_paths() -> Result<RepoKvPaths, CoreError> {
    let repo_root = PathBuf::from(run_git(&["rev-parse", "--show-toplevel"])?);
    if repo_root.as_os_str().is_empty() {
        return Err(CoreError::Io(
            "failed to resolve repository root path".to_string(),
        ));
    }

    let git_dir_raw = PathBuf::from(run_git(&["rev-parse", "--git-common-dir"])?);
    let git_dir = if git_dir_raw.is_absolute() {
        git_dir_raw
    } else {
        repo_root.join(git_dir_raw)
    };
    let cache_dir = git_dir.join(CACHE_DIR);

    Ok(RepoKvPaths {
        repo_root,
        git_dir,
        generation_state: cache_dir.join(GENERATION_STATE_FILE),
        metadata: cache_dir.join(METADATA_FILE),
    })
}

pub(crate) fn maybe_discover_repo_kv_paths() -> Option<RepoKvPaths> {
    discover_repo_kv_paths().ok()
}

pub(crate) fn ensure_cache_dir(paths: &RepoKvPaths) -> Result<(), CoreError> {
    let cache_dir = paths
        .generation_state
        .parent()
        .ok_or_else(|| CoreError::Io("invalid generation cache path".to_string()))?;
    fs::create_dir_all(cache_dir)?;
    Ok(())
}

pub(crate) fn write_metadata(
    paths: &RepoKvPaths,
    metadata: &RepoKvMetadata,
) -> Result<(), CoreError> {
    if let Some(parent) = paths.metadata.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(metadata)?;
    fs::write(&paths.metadata, bytes)?;
    Ok(())
}

pub(crate) fn read_metadata(paths: &RepoKvPaths) -> Option<RepoKvMetadata> {
    let bytes = fs::read(&paths.metadata).ok()?;
    let metadata = serde_json::from_slice::<RepoKvMetadata>(&bytes).ok()?;
    if metadata.version == CACHE_SCHEMA_VERSION {
        Some(metadata)
    } else {
        None
    }
}

fn run_git(args: &[&str]) -> Result<String, CoreError> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|err| CoreError::Io(format!("failed to run git {}: {err}", args.join(" "))))?;

    if !output.status.success() {
        return Err(CoreError::Io(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
