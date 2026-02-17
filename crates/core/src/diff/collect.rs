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
}
