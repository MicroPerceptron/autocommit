use crate::types::DiffChunk;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffFeatures {
    pub files_changed: usize,
    pub lines_changed: usize,
    pub hunks: usize,
    pub binary_files: usize,
    pub risky_paths: usize,
}

pub fn extract(chunks: &[DiffChunk]) -> DiffFeatures {
    let mut features = DiffFeatures::default();
    features.files_changed = chunks.len();

    for chunk in chunks {
        features.hunks += chunk.ranges.len();
        features.lines_changed += chunk
            .text
            .lines()
            .filter(|line| line.starts_with('+') || line.starts_with('-'))
            .count();

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
