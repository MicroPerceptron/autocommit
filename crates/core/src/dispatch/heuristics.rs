use crate::diff::features::DiffFeatures;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeuristicScore {
    pub complexity: f32,
    pub risky: bool,
    pub borderline: bool,
}

pub fn score(features: &DiffFeatures) -> HeuristicScore {
    let complexity = features.files_changed as f32 * 0.35
        + features.lines_changed as f32 * 0.005
        + features.hunks as f32 * 0.2
        + features.risky_paths as f32 * 0.7;

    let risky = features.risky_paths > 0 || features.binary_files > 0;
    let borderline = (150..=900).contains(&features.lines_changed);

    HeuristicScore {
        complexity,
        risky,
        borderline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_risky_config_changes() {
        let features = DiffFeatures {
            files_changed: 1,
            lines_changed: 50,
            hunks: 2,
            binary_files: 0,
            risky_paths: 1,
        };
        let score = score(&features);
        assert!(score.risky);
        assert!(score.complexity > 0.0);
    }
}
