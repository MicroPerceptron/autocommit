/// Classifies file paths by their semantic importance to a commit message.
///
/// `classify_path` returns `Some(tier)` for cases where the path alone is
/// decisive, and `None` for ambiguous paths that need embedding-based
/// classification (e.g., manifests that could be adding a core dependency
/// or just bumping a dev tool version).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImportanceTier {
    /// Core application/library code — defines the commit's purpose.
    Primary,
    /// Tests, documentation, build scripts — validates or supports primary code.
    Secondary,
    /// Config, manifests, lock files, dotfiles — accompanies code changes.
    Supporting,
}

/// Classify a file path into an importance tier.
///
/// Returns `Some(tier)` when the path unambiguously maps to a tier,
/// `None` when the path is ambiguous and needs embedding-based classification.
pub fn classify_path(path: &str) -> Option<ImportanceTier> {
    let path_lower = path.to_ascii_lowercase();
    let filename = filename_of(&path_lower);

    if is_lock_file(filename) || is_generated(filename, &path_lower) {
        return Some(ImportanceTier::Supporting);
    }

    // Ambiguous check before dotfile — CI workflows (.github/workflows/*)
    // and other paths starting with '.' may need embedding-based classification.
    if is_ambiguous_file(filename, &path_lower) {
        return None;
    }

    if is_dotfile(&path_lower) {
        return Some(ImportanceTier::Supporting);
    }

    if is_test_file(filename, &path_lower) {
        return Some(ImportanceTier::Secondary);
    }

    if is_docs_file(filename, &path_lower) {
        return Some(ImportanceTier::Secondary);
    }

    Some(ImportanceTier::Primary)
}

fn is_lock_file(filename: &str) -> bool {
    matches!(
        filename,
        "cargo.lock"
            | "package-lock.json"
            | "yarn.lock"
            | "pnpm-lock.yaml"
            | "poetry.lock"
            | "pipfile.lock"
            | "gemfile.lock"
            | "composer.lock"
            | "go.sum"
            | "flake.lock"
    )
}

fn is_generated(filename: &str, path: &str) -> bool {
    filename == "generated.rs"
        || filename.ends_with(".generated.ts")
        || filename.ends_with(".generated.js")
        || filename.ends_with(".pb.go")
        || filename.ends_with(".pb.rs")
        || path.contains("/generated/")
        || path.contains("/__generated__/")
}

fn is_dotfile(path: &str) -> bool {
    // Files or directories starting with a dot (hidden).
    path.starts_with('.') || path.contains("/.")
}

fn is_ambiguous_file(filename: &str, path: &str) -> bool {
    // Manifests: could be meaningful dependency changes or trivial version bumps.
    let is_manifest = matches!(
        filename,
        "cargo.toml"
            | "package.json"
            | "pyproject.toml"
            | "setup.cfg"
            | "setup.py"
            | "go.mod"
            | "gemfile"
            | "composer.json"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
    );
    if is_manifest {
        return true;
    }

    // Build scripts — could be core build logic or trivial config.
    if filename == "build.rs"
        || filename == "makefile"
        || filename == "justfile"
        || filename == "cmakelists.txt"
    {
        return true;
    }

    // CI workflows — could be meaningful pipeline changes or trivial tweaks.
    if path.starts_with(".github/workflows/")
        || path.starts_with(".gitlab-ci")
        || path.starts_with(".circleci/")
    {
        return true;
    }

    // Docker files — could be meaningful infra or trivial base image bumps.
    if filename == "dockerfile"
        || filename == "docker-compose.yml"
        || filename == "docker-compose.yaml"
    {
        return true;
    }

    false
}

fn is_test_file(filename: &str, path: &str) -> bool {
    if path.starts_with("tests/")
        || path.starts_with("test/")
        || path.contains("/tests/")
        || path.contains("/test/")
        || path.contains("/__tests__/")
    {
        return true;
    }
    filename.ends_with("_test.rs")
        || filename.ends_with("_test.go")
        || filename.ends_with("_test.py")
        || filename.ends_with(".test.ts")
        || filename.ends_with(".test.tsx")
        || filename.ends_with(".test.js")
        || filename.ends_with(".test.jsx")
        || filename.ends_with(".spec.ts")
        || filename.ends_with(".spec.tsx")
        || filename.ends_with(".spec.js")
        || filename.ends_with(".spec.jsx")
        || filename.starts_with("test_")
}

fn is_docs_file(filename: &str, path: &str) -> bool {
    if path.starts_with("docs/") || path.starts_with("doc/") {
        return true;
    }
    matches!(
        filename,
        "readme.md" | "changelog.md" | "contributing.md" | "license" | "license.md"
    ) || (filename.ends_with(".md") && !path.contains("/src/"))
}

fn filename_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_code_is_primary() {
        assert_eq!(classify_path("crates/core/src/lib.rs"), Some(ImportanceTier::Primary));
        assert_eq!(classify_path("src/main.py"), Some(ImportanceTier::Primary));
        assert_eq!(classify_path("lib/handler.ts"), Some(ImportanceTier::Primary));
        assert_eq!(classify_path("cmd/server/main.go"), Some(ImportanceTier::Primary));
    }

    #[test]
    fn lock_files_are_supporting() {
        assert_eq!(classify_path("Cargo.lock"), Some(ImportanceTier::Supporting));
        assert_eq!(classify_path("package-lock.json"), Some(ImportanceTier::Supporting));
        assert_eq!(classify_path("yarn.lock"), Some(ImportanceTier::Supporting));
    }

    #[test]
    fn dotfiles_are_supporting() {
        assert_eq!(classify_path(".claude/settings.json"), Some(ImportanceTier::Supporting));
        assert_eq!(classify_path(".mcp.json"), Some(ImportanceTier::Supporting));
        assert_eq!(classify_path(".gitignore"), Some(ImportanceTier::Supporting));
    }

    #[test]
    fn tests_are_secondary() {
        assert_eq!(classify_path("tests/integration.rs"), Some(ImportanceTier::Secondary));
        assert_eq!(classify_path("src/handler.test.ts"), Some(ImportanceTier::Secondary));
        assert_eq!(
            classify_path("crates/core/tests/fanout.rs"),
            Some(ImportanceTier::Secondary)
        );
    }

    #[test]
    fn docs_are_secondary() {
        assert_eq!(classify_path("docs/api.md"), Some(ImportanceTier::Secondary));
        assert_eq!(classify_path("README.md"), Some(ImportanceTier::Secondary));
    }

    #[test]
    fn manifests_are_ambiguous() {
        assert_eq!(classify_path("Cargo.toml"), None);
        assert_eq!(classify_path("crates/core/Cargo.toml"), None);
        assert_eq!(classify_path("package.json"), None);
        assert_eq!(classify_path("pyproject.toml"), None);
    }

    #[test]
    fn build_scripts_are_ambiguous() {
        assert_eq!(classify_path("build.rs"), None);
        assert_eq!(classify_path("Makefile"), None);
    }

    #[test]
    fn ci_workflows_are_ambiguous() {
        assert_eq!(classify_path(".github/workflows/ci.yml"), None);
    }

    #[test]
    fn generated_files_are_supporting() {
        assert_eq!(classify_path("generated.rs"), Some(ImportanceTier::Supporting));
        assert_eq!(
            classify_path("src/api.generated.ts"),
            Some(ImportanceTier::Supporting)
        );
        assert_eq!(
            classify_path("proto/service.pb.go"),
            Some(ImportanceTier::Supporting)
        );
    }
}
