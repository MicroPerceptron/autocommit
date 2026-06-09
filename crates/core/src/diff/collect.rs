use std::collections::BTreeMap;

use crate::diff::parse_unified;
use crate::types::DiffChunk;

pub fn collect(diff_text: &str) -> Vec<DiffChunk> {
    let mut chunks = Vec::new();
    let mut current_path = String::new();
    let mut current = String::new();

    for line in diff_text.lines() {
        if line.starts_with("diff --git ") {
            flush_chunk(&mut chunks, &current_path, &current);
            current.clear();
            current_path = parse_path(line).unwrap_or_else(|| "unknown".to_string());
        }

        current.push_str(line);
        current.push('\n');
    }

    flush_chunk(&mut chunks, &current_path, &current);

    if chunks.is_empty() && !diff_text.trim().is_empty() {
        chunks.push(DiffChunk {
            path: "unknown".to_string(),
            text: diff_text.to_string(),
            ranges: parse_unified::extract_ranges(diff_text),
            estimated_tokens: estimate_tokens(diff_text),
        });
    }

    chunks
}

/// Merge chunks that share the same directory scope, reducing fanout call count.
/// Preserves `diff --git` headers within merged chunks so the model can still see
/// per-file boundaries. Chunks whose merged estimated_tokens would exceed
/// `max_tokens_per_chunk` are left separate.
pub fn merge_by_scope(chunks: Vec<DiffChunk>, max_tokens_per_chunk: usize) -> Vec<DiffChunk> {
    if chunks.len() <= 1 {
        return chunks;
    }

    let mut groups: BTreeMap<String, Vec<DiffChunk>> = BTreeMap::new();
    for chunk in chunks {
        let scope = scope_key(&chunk.path);
        groups.entry(scope).or_default().push(chunk);
    }

    let mut result = Vec::with_capacity(groups.len());
    for (_scope, group) in groups {
        if group.len() == 1 {
            result.extend(group);
            continue;
        }

        // Try to merge all chunks in this scope group.
        let mut merged_text = String::new();
        let mut merged_ranges = Vec::new();
        let mut merged_tokens = 0usize;
        let mut paths: Vec<String> = Vec::new();
        let first_path = group.first().map(|c| c.path.clone()).unwrap_or_default();

        for chunk in group {
            if !merged_text.is_empty()
                && merged_tokens + chunk.estimated_tokens > max_tokens_per_chunk
            {
                // Flush current merge.
                let flush_path = if paths.len() == 1 {
                    paths[0].clone()
                } else {
                    first_path.clone()
                };
                result.push(DiffChunk {
                    path: flush_path,
                    text: std::mem::take(&mut merged_text),
                    ranges: std::mem::take(&mut merged_ranges),
                    estimated_tokens: merged_tokens,
                });
                merged_tokens = 0;
                paths.clear();
            }

            if !merged_text.is_empty() {
                merged_text.push('\n');
            }
            if paths.is_empty() || paths.last().map(|p| p.as_str()) != Some(&chunk.path) {
                paths.push(chunk.path.clone());
            }
            merged_text.push_str(&chunk.text);
            merged_ranges.extend(chunk.ranges);
            merged_tokens += chunk.estimated_tokens;
        }

        if !merged_text.is_empty() {
            let path = if paths.len() == 1 {
                paths.into_iter().next().unwrap()
            } else {
                first_path
            };
            result.push(DiffChunk {
                path,
                text: merged_text,
                ranges: merged_ranges,
                estimated_tokens: merged_tokens,
            });
        }
    }

    result
}

fn scope_key(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 2 {
        return path.to_string();
    }
    match parts[0] {
        "crates" | "src" | "packages" | "libs" => parts
            .get(1)
            .map(|s| format!("{}/{}", parts[0], s))
            .unwrap_or_else(|| parts[0].to_string()),
        "tests" | "test" => "tests".to_string(),
        "docs" => "docs".to_string(),
        "third_party" | "vendor" => "third_party".to_string(),
        _ => parts[0].to_string(),
    }
}

fn flush_chunk(chunks: &mut Vec<DiffChunk>, path: &str, text: &str) {
    if text.trim().is_empty() {
        return;
    }

    let clean_path = if path.is_empty() { "unknown" } else { path };
    chunks.push(DiffChunk {
        path: clean_path.to_string(),
        text: text.to_string(),
        ranges: parse_unified::extract_ranges(text),
        estimated_tokens: estimate_tokens(text),
    });
}

fn parse_path(header: &str) -> Option<String> {
    let mut parts = header.split_whitespace();
    let _ = parts.next()?;
    let _ = parts.next()?;
    let _old = parts.next()?;
    let new_path = parts.next()?;
    Some(new_path.trim_start_matches("b/").to_string())
}

fn estimate_tokens(text: &str) -> usize {
    if text.trim().is_empty() {
        return 1;
    }

    let whitespace_tokens = text.split_whitespace().count();
    let punctuation_tokens = text
        .bytes()
        .filter(|byte| matches!(*byte, b'{' | b'}' | b'(' | b')' | b'[' | b']' | b':' | b';'))
        .count()
        / 3;
    let char_tokens = text.len() / 10;

    whitespace_tokens
        .max(char_tokens)
        .saturating_add(punctuation_tokens)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_diff_by_file_header() {
        let diff = "diff --git a/src/a.rs b/src/a.rs\n@@ -1 +1 @@\n-x\n+y\ndiff --git a/src/b.rs b/src/b.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let chunks = collect(diff);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].path, "src/a.rs");
        assert_eq!(chunks[1].path, "src/b.rs");
    }

    #[test]
    fn merge_by_scope_groups_same_directory() {
        let diff = "\
diff --git a/crates/core/src/a.rs b/crates/core/src/a.rs\n@@ -1 +1 @@\n-x\n+y\n\
diff --git a/crates/core/src/b.rs b/crates/core/src/b.rs\n@@ -1 +1 @@\n-a\n+b\n\
diff --git a/crates/cli/src/c.rs b/crates/cli/src/c.rs\n@@ -1 +1 @@\n-m\n+n\n";
        let chunks = collect(diff);
        assert_eq!(chunks.len(), 3);

        let merged = merge_by_scope(chunks, 10_000);
        // core/a.rs and core/b.rs share scope "crates/core", cli/c.rs is separate.
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_respects_token_cap() {
        let diff = "\
diff --git a/src/a.rs b/src/a.rs\n@@ -1 +1 @@\n-x\n+y\n\
diff --git a/src/b.rs b/src/b.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let chunks = collect(diff);
        // Set cap so low that merging is prevented.
        let merged = merge_by_scope(chunks, 1);
        assert_eq!(merged.len(), 2);
    }
}
