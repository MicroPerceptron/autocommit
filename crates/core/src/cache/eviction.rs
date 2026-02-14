use std::fs;
use std::path::Path;

pub fn dir_size_bytes(path: &Path) -> u64 {
    let mut size = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                size = size.saturating_add(dir_size_bytes(&p));
            } else if let Ok(meta) = entry.metadata() {
                size = size.saturating_add(meta.len());
            }
        }
    }
    size
}
