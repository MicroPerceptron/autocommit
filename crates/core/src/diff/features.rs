use crate::types::DiffChunk;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffFeatures {
    pub files_changed: usize,
    pub lines_changed: usize,
    pub hunks: usize,
    pub binary_files: usize,
    pub risky_paths: usize,
    pub whitespace_only_lines: usize,
}

pub fn extract(chunks: &[DiffChunk]) -> DiffFeatures {
    let mut features = DiffFeatures {
        files_changed: chunks.len(),
        ..Default::default()
    };

    for chunk in chunks {
        features.hunks += chunk.ranges.len();
        for line in chunk.text.lines() {
            if line.starts_with('+') || line.starts_with('-') {
                features.lines_changed += 1;
                if line[1..].bytes().all(|b| b.is_ascii_whitespace()) {
                    features.whitespace_only_lines += 1;
                }
            }
        }

        if chunk.text.contains("Binary files") {
            features.binary_files += 1;
        }

        let path = chunk.path.as_str();
        if path.contains("Cargo.toml")
            || path.contains("package.json")
            || path.contains("migrations")
            || path.contains(".github/workflows")
        {
            features.risky_paths += 1;
        }
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LineRange;

    fn chunk(path: &str, text: &str) -> DiffChunk {
        DiffChunk {
            path: path.to_string(),
            text: text.to_string(),
            ranges: vec![LineRange {
                old_start: 1,
                old_count: 1,
                new_start: 1,
                new_count: 1,
            }],
            estimated_tokens: text.len() / 4,
        }
    }

    #[test]
    fn whitespace_only_blank_lines() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    @@ -1,3 +1,5 @@\n \
                    fn main() {\n\
                    +\n\
                    +\n\
                     println!(\"hello\");\n\
                     }";
        let features = extract(&[chunk("src/lib.rs", diff)]);
        assert_eq!(features.lines_changed, 2);
        assert_eq!(features.whitespace_only_lines, 2);
    }

    #[test]
    fn whitespace_only_indentation_changes() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    @@ -1,2 +1,2 @@\n\
                    -    old_indent\n\
                    +        new_indent\n";
        let features = extract(&[chunk("src/lib.rs", diff)]);
        assert_eq!(features.lines_changed, 2);
        // These have non-whitespace content after the prefix, so NOT whitespace-only
        assert_eq!(features.whitespace_only_lines, 0);
    }

    #[test]
    fn whitespace_only_mixed_diff() {
        // Mix of whitespace-only and content changes
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    @@ -1,4 +1,4 @@\n\
                    -\n\
                    +\n\
                    -    let x = 1;\n\
                    +    let x = 2;\n";
        let features = extract(&[chunk("src/lib.rs", diff)]);
        assert_eq!(features.lines_changed, 4);
        assert_eq!(features.whitespace_only_lines, 2);
    }

    #[test]
    fn whitespace_only_tabs_and_spaces() {
        let diff = "diff --git a/f.py b/f.py\n\
                    @@ -1,2 +1,2 @@\n\
                    -\t \t\n\
                    + \t \n";
        let features = extract(&[chunk("f.py", diff)]);
        assert_eq!(features.lines_changed, 2);
        assert_eq!(features.whitespace_only_lines, 2);
    }

    #[test]
    fn no_whitespace_only_in_functional_diff() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    @@ -1,2 +1,2 @@\n\
                    -fn old() {}\n\
                    +fn new() {}\n";
        let features = extract(&[chunk("src/lib.rs", diff)]);
        assert_eq!(features.lines_changed, 2);
        assert_eq!(features.whitespace_only_lines, 0);
    }
}
