use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use autocommit_core::AnalysisReport;
use autocommit_core::types::TypeTag;
use serde::{Deserialize, Serialize};

use crate::cmd::git;

const VERSION_CONTEXT_SCHEMA: u32 = 1;
const CACHE_DIR: &str = "autocommit/kv";
const VERSION_CONTEXT_FILE: &str = "version_context.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BumpLevel {
    Patch,
    Minor,
    Major,
}

impl BumpLevel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Patch => "patch",
            Self::Minor => "minor",
            Self::Major => "major",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VersionRecommendation {
    pub(crate) manifest_path: String,
    pub(crate) ecosystem: &'static str,
    pub(crate) tool: &'static str,
    pub(crate) current_version: Option<String>,
    pub(crate) suggested_version: Option<String>,
    pub(crate) level: BumpLevel,
    pub(crate) reason: String,
    kind: ManifestKind,
}

#[cfg(test)]
pub(crate) fn test_recommendation(
    manifest_path: &str,
    ecosystem: &'static str,
    tool: &'static str,
    current_version: Option<&str>,
    suggested_version: Option<&str>,
    level: BumpLevel,
    reason: &str,
) -> VersionRecommendation {
    VersionRecommendation {
        manifest_path: manifest_path.to_string(),
        ecosystem,
        tool,
        current_version: current_version.map(ToOwned::to_owned),
        suggested_version: suggested_version.map(ToOwned::to_owned),
        level,
        reason: reason.to_string(),
        kind: ManifestKind::CargoToml,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestKind {
    CargoToml,
    PackageJson,
    PyprojectToml,
    SetupCfg,
    SetupPy,
    GoMod,
    PomXml,
    Gradle,
    ComposerJson,
    Gemfile,
    Gemspec,
    MixExs,
    PackageSwift,
    Csproj,
    PubspecYaml,
}

impl ManifestKind {
    fn ecosystem(self) -> &'static str {
        match self {
            Self::CargoToml => "Rust",
            Self::PackageJson => "JavaScript/TypeScript",
            Self::PyprojectToml | Self::SetupCfg | Self::SetupPy => "Python",
            Self::GoMod => "Go",
            Self::PomXml | Self::Gradle => "Java/Kotlin",
            Self::ComposerJson => "PHP",
            Self::Gemfile | Self::Gemspec => "Ruby",
            Self::MixExs => "Elixir",
            Self::PackageSwift => "Swift",
            Self::Csproj => ".NET",
            Self::PubspecYaml => "Dart/Flutter",
        }
    }

    fn tool(self) -> &'static str {
        match self {
            Self::CargoToml => "Cargo",
            Self::PackageJson => "npm/yarn/pnpm",
            Self::PyprojectToml => "pyproject",
            Self::SetupCfg => "setuptools",
            Self::SetupPy => "setuptools",
            Self::GoMod => "go mod",
            Self::PomXml => "Maven",
            Self::Gradle => "Gradle",
            Self::ComposerJson => "Composer",
            Self::Gemfile => "Bundler",
            Self::Gemspec => "RubyGems",
            Self::MixExs => "Mix",
            Self::PackageSwift => "SwiftPM",
            Self::Csproj => "MSBuild/NuGet",
            Self::PubspecYaml => "pub",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct VersionContext {
    schema_version: u32,
    updated_unix_secs: u64,
    manifests: BTreeMap<String, ManifestSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ManifestSnapshot {
    kind: String,
    version: Option<String>,
    last_recommended: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Semver {
    major: u64,
    minor: u64,
    patch: u64,
}

#[derive(Debug, Clone)]
struct GoModInfo {
    major: u64,
}

pub(crate) fn recommend(
    repo: &git::Repo,
    diff_text: &str,
    report: &AnalysisReport,
    embedding_level: Option<BumpLevel>,
) -> Vec<VersionRecommendation> {
    recommend_inner(repo, diff_text, report, embedding_level).unwrap_or_default()
}

fn recommend_inner(
    repo: &git::Repo,
    diff_text: &str,
    report: &AnalysisReport,
    embedding_level: Option<BumpLevel>,
) -> Result<Vec<VersionRecommendation>, std::io::Error> {
    let repo_root = repo.repo_root();
    let context_path = repo
        .common_git_dir()
        .join(CACHE_DIR)
        .join(VERSION_CONTEXT_FILE);

    let changed_paths = parse_changed_paths(diff_text);
    let source_changed = changed_paths
        .iter()
        .any(|path| !is_manifest_path(path) && !is_lockfile_path(path));
    let heuristic_level = suggested_level(report);
    let recommended_level = combine_recommended_level(heuristic_level, embedding_level);

    let mut context = load_context(&context_path);
    let candidates = collect_manifest_candidates(&repo_root, &changed_paths);

    let mut recommendations = Vec::new();
    for (manifest_path, kind) in candidates {
        let absolute = repo_root.join(&manifest_path);
        let mut current_version = read_manifest_version(&absolute, kind);
        let go_info = if kind == ManifestKind::GoMod {
            let info = read_go_mod_info(&absolute);
            current_version = info
                .as_ref()
                .map(|go| format!("v{}", go.major))
                .or(current_version);
            info
        } else {
            None
        };
        let previous_version = context
            .manifests
            .get(&manifest_path)
            .and_then(|snapshot| snapshot.version.clone());
        let manifest_changed = changed_paths.contains(&manifest_path);
        let should_evaluate = source_changed || manifest_changed;

        if should_evaluate {
            if kind == ManifestKind::GoMod {
                if let Some(rec) = build_go_mod_recommendation(
                    &manifest_path,
                    go_info.as_ref(),
                    previous_version.as_deref(),
                    manifest_changed,
                    recommended_level,
                ) {
                    recommendations.push(rec);
                }
            } else if let Some(rec) = build_recommendation(
                &manifest_path,
                kind,
                current_version.as_deref(),
                previous_version.as_deref(),
                manifest_changed,
                recommended_level,
            ) {
                recommendations.push(rec);
            }
        }

        let snapshot = ManifestSnapshot {
            kind: format!("{kind:?}"),
            version: current_version.clone(),
            last_recommended: recommendations
                .iter()
                .rev()
                .find(|rec| rec.manifest_path == manifest_path)
                .and_then(|rec| rec.suggested_version.clone()),
        };
        context.manifests.insert(manifest_path, snapshot);
    }

    persist_context(&context_path, &context)?;
    recommendations.sort_by(|a, b| a.manifest_path.cmp(&b.manifest_path));
    Ok(recommendations)
}

fn combine_recommended_level(heuristic: BumpLevel, embedding: Option<BumpLevel>) -> BumpLevel {
    // Embeddings are currently good at catching obvious breaking-change signals, but
    // too noisy for patch-vs-minor distinctions. Keep patch/minor heuristic-driven.
    match embedding {
        Some(BumpLevel::Major) => BumpLevel::Major.max(heuristic),
        _ => heuristic,
    }
}

fn build_recommendation(
    manifest_path: &str,
    kind: ManifestKind,
    current_version: Option<&str>,
    previous_version: Option<&str>,
    manifest_changed: bool,
    level: BumpLevel,
) -> Option<VersionRecommendation> {
    let (current_semver, previous_semver) = (
        current_version.and_then(parse_semver),
        previous_version.and_then(parse_semver),
    );

    if let (Some(previous), Some(current)) = (previous_semver, current_semver) {
        if current > previous {
            let actual = bump_distance(previous, current);
            if let Some(actual) = actual {
                if actual >= level {
                    return None;
                }
            }
            let suggested = previous.bump(level).to_string();
            let actual_level = actual.unwrap_or(BumpLevel::Patch);
            return Some(VersionRecommendation {
                manifest_path: manifest_path.to_string(),
                ecosystem: kind.ecosystem(),
                tool: kind.tool(),
                current_version: current_version.map(ToOwned::to_owned),
                suggested_version: Some(suggested),
                level,
                reason: format!(
                    "version bumped by {}, but detected changes suggest at least a {} bump",
                    actual_level.as_str(),
                    level.as_str(),
                ),
                kind,
            });
        }
    }

    if let Some(current) = current_semver {
        let suggested = current.bump(level).to_string();
        let reason = if manifest_changed {
            format!(
                "manifest changed without an explicit project version bump; suggest {} bump",
                level.as_str()
            )
        } else {
            format!(
                "code changes detected; suggest a {} project version bump",
                level.as_str()
            )
        };
        return Some(VersionRecommendation {
            manifest_path: manifest_path.to_string(),
            ecosystem: kind.ecosystem(),
            tool: kind.tool(),
            current_version: current_version.map(ToOwned::to_owned),
            suggested_version: Some(suggested),
            level,
            reason,
            kind,
        });
    }
    None
}

fn build_go_mod_recommendation(
    manifest_path: &str,
    info: Option<&GoModInfo>,
    previous_version: Option<&str>,
    manifest_changed: bool,
    level: BumpLevel,
) -> Option<VersionRecommendation> {
    let info = info?;
    if level != BumpLevel::Major {
        return None;
    }

    let previous_major = previous_version
        .and_then(parse_semver)
        .map(|semver| semver.major);
    if let Some(prev) = previous_major {
        if info.major > prev {
            return None;
        }
    }

    let target_major = if info.major < 2 { 2 } else { info.major + 1 };
    let suggested = format!("v{target_major}");
    let current_version = format!("v{}", info.major);
    let reason = if manifest_changed {
        format!("go.mod changed; breaking changes suggest module path bump to /{suggested}")
    } else {
        format!("breaking changes suggest Go module path bump to /{suggested}")
    };

    Some(VersionRecommendation {
        manifest_path: manifest_path.to_string(),
        ecosystem: ManifestKind::GoMod.ecosystem(),
        tool: ManifestKind::GoMod.tool(),
        current_version: Some(current_version),
        suggested_version: Some(suggested),
        level,
        reason,
        kind: ManifestKind::GoMod,
    })
}

pub(crate) fn apply(
    repo: &git::Repo,
    recommendations: &[VersionRecommendation],
) -> Result<Vec<String>, String> {
    if recommendations.is_empty() {
        return Ok(Vec::new());
    }

    let root = repo.repo_root();
    let mut touched = Vec::new();

    for rec in recommendations {
        let suggested = rec.suggested_version.as_deref().ok_or_else(|| {
            format!(
                "missing suggested version for {}, cannot apply bump",
                rec.manifest_path
            )
        })?;
        let manifest_path = root.join(&rec.manifest_path);
        apply_manifest_bump(&manifest_path, rec.kind, suggested).map_err(|err| {
            format!(
                "failed to bump {} to {}: {}",
                rec.manifest_path, suggested, err
            )
        })?;
        touched.push(rec.manifest_path.clone());
    }

    touched.sort();
    touched.dedup();
    Ok(touched)
}

fn suggested_level(report: &AnalysisReport) -> BumpLevel {
    let subject = report.commit_message.trim().to_ascii_lowercase();
    let summary = report.summary.trim().to_ascii_lowercase();
    let risk_notes = report.risk.notes.join(" ").to_ascii_lowercase();

    let has_breaking = contains_breaking_signal(&subject)
        || contains_breaking_signal(&summary)
        || contains_breaking_signal(&risk_notes)
        || report.items.iter().any(|item| {
            let text = format!(
                "{} {}",
                item.title.to_ascii_lowercase(),
                item.intent.to_ascii_lowercase()
            );
            contains_breaking_signal(&text)
        });
    if has_breaking {
        return BumpLevel::Major;
    }

    let subject_feature = subject.starts_with("feat(")
        || subject.starts_with("feat:")
        || subject.starts_with("feature(")
        || subject.starts_with("feature:");
    let subject_patchy = subject.starts_with("fix(")
        || subject.starts_with("fix:")
        || subject.starts_with("refactor(")
        || subject.starts_with("refactor:")
        || subject.starts_with("chore(")
        || subject.starts_with("chore:")
        || subject.starts_with("docs(")
        || subject.starts_with("docs:")
        || subject.starts_with("test(")
        || subject.starts_with("test:");

    let total = report.items.len();
    let feature_like = report
        .items
        .iter()
        .filter(|item| {
            matches!(item.type_tag, TypeTag::Feat)
                || matches!(item.bucket, autocommit_core::types::ChangeBucket::Feature)
        })
        .count();
    let strong_feature_like = report
        .items
        .iter()
        .filter(|item| item.confidence >= 0.78)
        .filter(|item| {
            matches!(item.type_tag, TypeTag::Feat)
                || matches!(item.bucket, autocommit_core::types::ChangeBucket::Feature)
        })
        .count();
    let strong_feature_phrase_hits = report
        .items
        .iter()
        .filter(|item| {
            let text = format!(
                "{} {}",
                item.title.to_ascii_lowercase(),
                item.intent.to_ascii_lowercase()
            );
            contains_strong_feature_signal(&text)
        })
        .count();
    let summary_or_risk_feature = contains_strong_feature_signal(&summary)
        || contains_strong_feature_signal(&risk_notes)
        || contains_feature_signal(&summary)
        || contains_feature_signal(&risk_notes);
    let multi_item_feature =
        feature_like >= 2 && total > 0 && feature_like.saturating_mul(2) >= total;
    let high_conf_multi_item_feature =
        strong_feature_like >= 2 && total > 0 && strong_feature_like.saturating_mul(2) >= total;

    // Conservative default: patch unless there is clear and repeated feature evidence.
    if subject_patchy && !subject_feature {
        if high_conf_multi_item_feature
            && (strong_feature_phrase_hits > 0 || summary_or_risk_feature)
        {
            return BumpLevel::Minor;
        }
        return BumpLevel::Patch;
    }

    if subject_feature {
        if strong_feature_like >= 1 && (strong_feature_phrase_hits > 0 || summary_or_risk_feature) {
            return BumpLevel::Minor;
        }
        if multi_item_feature && strong_feature_phrase_hits > 0 {
            return BumpLevel::Minor;
        }
        return BumpLevel::Patch;
    }

    if high_conf_multi_item_feature && (strong_feature_phrase_hits > 0 || summary_or_risk_feature) {
        return BumpLevel::Minor;
    }

    BumpLevel::Patch
}

fn contains_breaking_signal(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "breaking change",
        "breaking:",
        "breaks compatibility",
        "incompatible",
        "backward incompatible",
        "backwards incompatible",
        "remove ",
        "removed ",
        "deprecat",
        "migration",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn contains_feature_signal(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "add ",
        "adds ",
        "added ",
        "new ",
        "introduce",
        "support ",
        "enable ",
        "implements ",
        "implement ",
        "expose ",
        "allow ",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn contains_strong_feature_signal(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "new command",
        "new subcommand",
        "new api",
        "new endpoint",
        "new flag",
        "new option",
        "new workflow",
        "add command",
        "add subcommand",
        "add api",
        "add endpoint",
        "add flag",
        "add option",
        "add support for",
        "introduce command",
        "introduce api",
        "introduce endpoint",
        "introduce feature",
        "support for ",
        "implements command",
        "implements api",
        "expose api",
        "expose endpoint",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn parse_changed_paths(diff_text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in diff_text.lines() {
        if !line.starts_with("diff --git ") {
            continue;
        }
        let mut parts = line.split_whitespace();
        let _ = parts.next();
        let _ = parts.next();
        let _old = parts.next();
        let Some(new_path) = parts.next() else {
            continue;
        };
        out.insert(new_path.trim_start_matches("b/").to_string());
    }
    out
}

fn collect_manifest_candidates(
    repo_root: &Path,
    changed_paths: &BTreeSet<String>,
) -> BTreeMap<String, ManifestKind> {
    let discovered = discover_repo_manifests(repo_root);
    let by_dir = manifests_by_dir(&discovered);
    let mut out = BTreeMap::new();
    let source_changed = changed_paths
        .iter()
        .any(|path| !is_manifest_path(path) && !is_lockfile_path(path));

    for path in changed_paths {
        if let Some(kind) = discovered.get(path).copied() {
            out.insert(path.clone(), kind);
            continue;
        }

        for manifest_path in nearest_manifest_paths(path, &by_dir) {
            if let Some(kind) = discovered.get(&manifest_path).copied() {
                out.entry(manifest_path).or_insert(kind);
            }
        }
    }

    if out.is_empty() && source_changed {
        for (path, kind) in root_level_manifests(&discovered) {
            out.insert(path, kind);
        }
    }

    out
}

fn discover_repo_manifests(repo_root: &Path) -> BTreeMap<String, ManifestKind> {
    let mut out = BTreeMap::new();
    discover_repo_manifests_recursive(repo_root, repo_root, &mut out);
    out
}

fn discover_repo_manifests_recursive(
    repo_root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, ManifestKind>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };

        if metadata.is_dir() {
            let name = entry.file_name();
            if should_skip_walk_dir(name.to_string_lossy().as_ref()) {
                continue;
            }
            discover_repo_manifests_recursive(repo_root, &path, out);
            continue;
        }

        if !metadata.is_file() {
            continue;
        }

        let relative = match path.strip_prefix(repo_root) {
            Ok(rel) => rel,
            Err(_) => continue,
        };
        let relative = normalize_rel_path(relative);
        if let Some(kind) = detect_manifest_kind(&relative) {
            out.insert(relative, kind);
        }
    }
}

fn should_skip_walk_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".venv"
            | "venv"
            | "__pycache__"
            | "dist"
            | "build"
            | ".next"
            | ".turbo"
            | ".idea"
    )
}

fn manifests_by_dir(manifests: &BTreeMap<String, ManifestKind>) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::<String, Vec<String>>::new();
    for path in manifests.keys() {
        let dir = manifest_parent_dir(path);
        out.entry(dir).or_default().push(path.clone());
    }

    for entries in out.values_mut() {
        entries.sort();
    }
    out
}

fn nearest_manifest_paths(path: &str, by_dir: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut dir = std::path::Path::new(path).parent();
    while let Some(parent) = dir {
        let key = normalize_rel_path(parent);
        if let Some(paths) = by_dir.get(&key) {
            return paths.clone();
        }

        if key.is_empty() {
            break;
        }
        dir = parent.parent();
    }

    Vec::new()
}

fn root_level_manifests(
    manifests: &BTreeMap<String, ManifestKind>,
) -> BTreeMap<String, ManifestKind> {
    manifests
        .iter()
        .filter(|(path, _)| !path.contains('/'))
        .map(|(path, kind)| (path.clone(), *kind))
        .collect()
}

fn manifest_parent_dir(path: &str) -> String {
    normalize_rel_path(
        std::path::Path::new(path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("")),
    )
}

fn normalize_rel_path(path: &Path) -> String {
    let raw = path.to_string_lossy().replace('\\', "/");
    if raw == "." {
        String::new()
    } else {
        raw.trim_matches('/').to_string()
    }
}

fn detect_manifest_kind(path: &str) -> Option<ManifestKind> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with("cargo.toml") {
        Some(ManifestKind::CargoToml)
    } else if lower.ends_with("package.json") {
        Some(ManifestKind::PackageJson)
    } else if lower.ends_with("pyproject.toml") {
        Some(ManifestKind::PyprojectToml)
    } else if lower.ends_with("setup.cfg") {
        Some(ManifestKind::SetupCfg)
    } else if lower.ends_with("setup.py") {
        Some(ManifestKind::SetupPy)
    } else if lower.ends_with("go.mod") {
        Some(ManifestKind::GoMod)
    } else if lower.ends_with("pom.xml") {
        Some(ManifestKind::PomXml)
    } else if lower.ends_with("build.gradle") || lower.ends_with("build.gradle.kts") {
        Some(ManifestKind::Gradle)
    } else if lower.ends_with("composer.json") {
        Some(ManifestKind::ComposerJson)
    } else if lower.ends_with("gemfile") {
        Some(ManifestKind::Gemfile)
    } else if lower.ends_with(".gemspec") {
        Some(ManifestKind::Gemspec)
    } else if lower.ends_with("mix.exs") {
        Some(ManifestKind::MixExs)
    } else if lower.ends_with("package.swift") {
        Some(ManifestKind::PackageSwift)
    } else if lower.ends_with(".csproj") {
        Some(ManifestKind::Csproj)
    } else if lower.ends_with("pubspec.yaml") {
        Some(ManifestKind::PubspecYaml)
    } else {
        None
    }
}

fn is_manifest_path(path: &str) -> bool {
    detect_manifest_kind(path).is_some()
}

fn is_lockfile_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with("cargo.lock")
        || lower.ends_with("package-lock.json")
        || lower.ends_with("yarn.lock")
        || lower.ends_with("pnpm-lock.yaml")
        || lower.ends_with("poetry.lock")
        || lower.ends_with("pipfile.lock")
        || lower.ends_with("go.sum")
        || lower.ends_with("composer.lock")
        || lower.ends_with("gemfile.lock")
}

fn read_manifest_version(path: &Path, kind: ManifestKind) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    match kind {
        ManifestKind::CargoToml => read_cargo_version(&content),
        ManifestKind::PackageJson | ManifestKind::ComposerJson => read_json_version(&content),
        ManifestKind::PyprojectToml => read_toml_section_value(&content, "project", "version")
            .or_else(|| read_toml_section_value(&content, "tool.poetry", "version")),
        ManifestKind::SetupCfg => read_ini_section_value(&content, "metadata", "version"),
        ManifestKind::SetupPy => read_python_setup_version(&content),
        ManifestKind::GoMod => None,
        ManifestKind::PomXml => read_xml_tag_value(&content, "version"),
        ManifestKind::Gradle => read_gradle_version(&content),
        ManifestKind::Gemfile => None,
        ManifestKind::Gemspec => read_gemspec_version(&content),
        ManifestKind::MixExs => read_mix_version(&content),
        ManifestKind::PackageSwift => None,
        ManifestKind::Csproj => read_xml_tag_value(&content, "Version"),
        ManifestKind::PubspecYaml => read_yaml_key_value(&content, "version"),
    }
}

fn read_cargo_version(content: &str) -> Option<String> {
    let mut in_package = false;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(value) = parse_key_value_line(line, "version") {
            return Some(value);
        }
    }
    None
}

fn read_json_version(content: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    value
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn read_toml_section_value(content: &str, section: &str, key: &str) -> Option<String> {
    let mut current_section: Option<String> = None;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            let section_name = line.trim_start_matches('[').trim_end_matches(']');
            current_section = Some(section_name.to_string());
            continue;
        }
        if current_section.as_deref() != Some(section) {
            continue;
        }
        if let Some(value) = parse_key_value_line(line, key) {
            return Some(value);
        }
    }
    None
}

fn read_ini_section_value(content: &str, section: &str, key: &str) -> Option<String> {
    let mut current_section: Option<String> = None;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            let section_name = line.trim_start_matches('[').trim_end_matches(']');
            current_section = Some(section_name.to_string());
            continue;
        }
        if current_section.as_deref() != Some(section) {
            continue;
        }
        if let Some(value) = parse_key_value_line(line, key) {
            return Some(value);
        }
    }
    None
}

fn read_xml_tag_value(content: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)?;
    let rest = &content[start + open.len()..];
    let end = rest.find(&close)?;
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn read_gradle_version(content: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if !line.starts_with("version") {
            continue;
        }
        let (_, value) = line.split_once('=')?;
        let value = value.trim().trim_matches('"').trim_matches('\'').trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn read_mix_version(content: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        let Some(pos) = line.find("version:") else {
            continue;
        };
        let value = line[pos + "version:".len()..]
            .trim()
            .trim_matches(',')
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn read_gemspec_version(content: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if !line.contains(".version") || !line.contains('=') {
            continue;
        }
        let (_, value) = line.split_once('=')?;
        let value = value.trim().trim_matches('"').trim_matches('\'').trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn read_python_setup_version(content: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        let Some(pos) = line.find("version=") else {
            continue;
        };
        let value = line[pos + "version=".len()..]
            .trim()
            .trim_matches(',')
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn read_yaml_key_value(content: &str, key: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if let Some(value) = parse_yaml_key_value_line(line, key) {
            return Some(value);
        }
    }
    None
}

fn parse_key_value_line(line: &str, key: &str) -> Option<String> {
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (lhs, rhs) = line.split_once('=')?;
    if lhs.trim() != key {
        return None;
    }

    let value = rhs
        .trim()
        .trim_end_matches(',')
        .trim_matches('"')
        .trim_matches('\'')
        .trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn parse_yaml_key_value_line(line: &str, key: &str) -> Option<String> {
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (lhs, rhs) = line.split_once(':')?;
    if lhs.trim() != key {
        return None;
    }

    let value = rhs
        .trim()
        .trim_end_matches(',')
        .trim_matches('"')
        .trim_matches('\'')
        .trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn apply_manifest_bump(
    manifest_path: &Path,
    kind: ManifestKind,
    suggested_version: &str,
) -> Result<(), std::io::Error> {
    let content = fs::read_to_string(manifest_path)?;
    let updated = match kind {
        ManifestKind::CargoToml => replace_toml_key_in_section(
            &content,
            "package",
            "version",
            suggested_version,
            AssignmentStyle::Toml,
        ),
        ManifestKind::PackageJson | ManifestKind::ComposerJson => {
            replace_json_version(&content, suggested_version)
        }
        ManifestKind::PyprojectToml => replace_toml_key_in_section(
            &content,
            "project",
            "version",
            suggested_version,
            AssignmentStyle::Toml,
        )
        .or_else(|| {
            replace_toml_key_in_section(
                &content,
                "tool.poetry",
                "version",
                suggested_version,
                AssignmentStyle::Toml,
            )
        }),
        ManifestKind::SetupCfg => replace_toml_key_in_section(
            &content,
            "metadata",
            "version",
            suggested_version,
            AssignmentStyle::Ini,
        ),
        ManifestKind::SetupPy => replace_setup_py_version(&content, suggested_version),
        ManifestKind::GoMod => {
            let suggested_major = parse_go_mod_major(suggested_version)
                .ok_or_else(|| std::io::Error::other("invalid Go module version format"))?;
            replace_go_mod_module_path(&content, suggested_major)
        }
        ManifestKind::PomXml => replace_xml_tag_value(&content, "version", suggested_version),
        ManifestKind::Gradle => replace_gradle_version(&content, suggested_version),
        ManifestKind::Gemfile => None,
        ManifestKind::Gemspec => replace_gemspec_version(&content, suggested_version),
        ManifestKind::MixExs => replace_mix_version(&content, suggested_version),
        ManifestKind::PackageSwift => None,
        ManifestKind::Csproj => replace_xml_tag_value(&content, "Version", suggested_version),
        ManifestKind::PubspecYaml => replace_yaml_key_value(&content, "version", suggested_version),
    }
    .ok_or_else(|| {
        std::io::Error::other("could not locate a writable in-file version field for this manifest")
    })?;

    fs::write(manifest_path, updated)
}

#[derive(Clone, Copy)]
enum AssignmentStyle {
    Toml,
    Ini,
}

fn replace_toml_key_in_section(
    content: &str,
    section: &str,
    key: &str,
    suggested_version: &str,
    style: AssignmentStyle,
) -> Option<String> {
    let mut current_section: Option<String> = None;
    let mut changed = false;
    let mut lines = Vec::new();
    let trailing_newline = content.ends_with('\n');

    for raw_line in content.lines() {
        let line = raw_line;
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section_name = trimmed.trim_start_matches('[').trim_end_matches(']');
            current_section = Some(section_name.to_string());
            lines.push(line.to_string());
            continue;
        }

        if current_section.as_deref() == Some(section) {
            if let Some(next) = replace_assignment_line(line, key, suggested_version, style) {
                lines.push(next);
                changed = true;
                continue;
            }
        }
        lines.push(line.to_string());
    }

    if !changed {
        return None;
    }

    Some(join_lines(lines, trailing_newline))
}

fn replace_assignment_line(
    line: &str,
    key: &str,
    suggested_version: &str,
    style: AssignmentStyle,
) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') || trimmed.starts_with(';') {
        return None;
    }
    if !trimmed.starts_with(key) {
        return None;
    }

    let rest = &trimmed[key.len()..];
    let mut rest_chars = rest.chars();
    let first = rest_chars.next()?;
    let valid_sep = first.is_ascii_whitespace()
        || matches!(
            (style, first),
            (AssignmentStyle::Toml, '=') | (AssignmentStyle::Ini, '=')
        );
    if !valid_sep {
        return None;
    }

    let indent_len = line.len().saturating_sub(trimmed.len());
    let indent = &line[..indent_len];
    Some(format!("{indent}{key} = \"{suggested_version}\""))
}

fn replace_json_version(content: &str, suggested_version: &str) -> Option<String> {
    let mut value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let object = value.as_object_mut()?;
    object.insert(
        "version".to_string(),
        serde_json::Value::String(suggested_version.to_string()),
    );
    serde_json::to_string_pretty(&value).ok()
}

fn replace_xml_tag_value(content: &str, tag: &str, suggested_version: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)?;
    let value_start = start + open.len();
    let rest = &content[value_start..];
    let value_end_rel = rest.find(&close)?;
    let value_end = value_start + value_end_rel;

    let mut updated = String::with_capacity(content.len() + suggested_version.len());
    updated.push_str(&content[..value_start]);
    updated.push_str(suggested_version);
    updated.push_str(&content[value_end..]);
    Some(updated)
}

fn replace_gradle_version(content: &str, suggested_version: &str) -> Option<String> {
    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }

            let trimmed = line.trim_start();
            if trimmed.starts_with("version") && trimmed.contains('=') {
                let indent_len = line.len().saturating_sub(trimmed.len());
                let indent = &line[..indent_len];
                changed = true;
                format!("{indent}version = \"{suggested_version}\"")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn replace_gemspec_version(content: &str, suggested_version: &str) -> Option<String> {
    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }
            let trimmed = line.trim_start();
            if !trimmed.contains(".version") || !trimmed.contains('=') {
                return line.to_string();
            }

            let indent_len = line.len().saturating_sub(trimmed.len());
            let indent = &line[..indent_len];
            let lhs = trimmed
                .split_once('=')
                .map(|(lhs, _)| lhs.trim())
                .unwrap_or("");
            if lhs.is_empty() {
                return line.to_string();
            }
            changed = true;
            format!("{indent}{lhs} = \"{suggested_version}\"")
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn replace_mix_version(content: &str, suggested_version: &str) -> Option<String> {
    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }
            let Some(marker) = line.find("version:") else {
                return line.to_string();
            };

            let prefix = &line[..marker];
            let rest = &line[marker + "version:".len()..];
            let suffix = rest.find(',').map(|idx| &rest[idx..]).unwrap_or("");
            changed = true;
            format!("{prefix}version: \"{suggested_version}\"{suffix}")
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn replace_setup_py_version(content: &str, suggested_version: &str) -> Option<String> {
    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }
            let Some(marker) = line.find("version=") else {
                return line.to_string();
            };

            let prefix = &line[..marker + "version=".len()];
            let rest = &line[marker + "version=".len()..];
            let suffix_start = rest
                .find(',')
                .or_else(|| rest.find(')'))
                .unwrap_or(rest.len());
            let suffix = &rest[suffix_start..];
            changed = true;
            format!("{prefix}\"{suggested_version}\"{suffix}")
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn replace_yaml_key_value(content: &str, key: &str, suggested_version: &str) -> Option<String> {
    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                return line.to_string();
            }
            let Some((lhs, rhs)) = trimmed.split_once(':') else {
                return line.to_string();
            };
            if lhs.trim() != key {
                return line.to_string();
            }

            let indent_len = line.len().saturating_sub(trimmed.len());
            let indent = &line[..indent_len];
            let comment = rhs.find('#').map(|idx| rhs[idx..].trim_end()).unwrap_or("");
            changed = true;
            if comment.is_empty() {
                format!("{indent}{key}: {suggested_version}")
            } else {
                format!("{indent}{key}: {suggested_version} {comment}")
            }
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn read_go_mod_info(path: &Path) -> Option<GoModInfo> {
    let content = fs::read_to_string(path).ok()?;
    parse_go_mod_info(&content)
}

fn parse_go_mod_info(content: &str) -> Option<GoModInfo> {
    for raw_line in content.lines() {
        let line = raw_line.trim_start();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !line.starts_with("module ") {
            continue;
        }
        let rest = line.trim_start_matches("module ").trim();
        let (module_path, _) = split_inline_comment(rest);
        let module_path = module_path.trim();
        if module_path.is_empty() {
            return None;
        }
        let (_, major, _) = parse_go_mod_path_major(module_path);
        let major = major.unwrap_or(1);
        return Some(GoModInfo { major });
    }
    None
}

fn parse_go_mod_path_major(module_path: &str) -> (&str, Option<u64>, bool) {
    if let Some((base, suffix)) = module_path.rsplit_once("/v") {
        if let Ok(major) = suffix.parse::<u64>() {
            if major >= 2 {
                return (base, Some(major), true);
            }
        }
    }
    (module_path, None, false)
}

fn parse_go_mod_major(value: &str) -> Option<u64> {
    let trimmed = value.trim().trim_start_matches('v');
    trimmed.parse::<u64>().ok()
}

fn replace_go_mod_module_path(content: &str, suggested_major: u64) -> Option<String> {
    if suggested_major < 2 {
        return None;
    }

    let mut changed = false;
    let trailing_newline = content.ends_with('\n');
    let lines = content
        .lines()
        .map(|line| {
            if changed {
                return line.to_string();
            }
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                return line.to_string();
            }
            if !trimmed.starts_with("module ") {
                return line.to_string();
            }

            let indent_len = line.len().saturating_sub(trimmed.len());
            let indent = &line[..indent_len];
            let rest = trimmed.trim_start_matches("module ").trim();
            let (module_path, comment) = split_inline_comment(rest);
            let module_path = module_path.trim();
            let (base, _, has_suffix) = parse_go_mod_path_major(module_path);
            let new_path = if has_suffix {
                format!("{base}/v{suggested_major}")
            } else {
                format!("{module_path}/v{suggested_major}")
            };

            changed = true;
            if comment.is_empty() {
                format!("{indent}module {new_path}")
            } else {
                format!("{indent}module {new_path} {comment}")
            }
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }
    Some(join_lines(lines, trailing_newline))
}

fn split_inline_comment(value: &str) -> (&str, &str) {
    if let Some(idx) = value.find("//") {
        let (left, right) = value.split_at(idx);
        (left.trim_end(), right.trim_end())
    } else {
        (value, "")
    }
}

fn join_lines(lines: Vec<String>, trailing_newline: bool) -> String {
    let mut out = lines.join("\n");
    if trailing_newline {
        out.push('\n');
    }
    out
}

fn load_context(path: &Path) -> VersionContext {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => {
            return VersionContext {
                schema_version: VERSION_CONTEXT_SCHEMA,
                ..VersionContext::default()
            };
        }
    };

    let context = serde_json::from_slice::<VersionContext>(&bytes).unwrap_or_default();
    if context.schema_version == VERSION_CONTEXT_SCHEMA {
        context
    } else {
        VersionContext {
            schema_version: VERSION_CONTEXT_SCHEMA,
            ..VersionContext::default()
        }
    }
}

fn persist_context(path: &Path, context: &VersionContext) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut context = context.clone();
    context.schema_version = VERSION_CONTEXT_SCHEMA;
    context.updated_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(0);

    let payload = serde_json::to_vec_pretty(&context)
        .map_err(|err| std::io::Error::other(format!("serialize version context failed: {err}")))?;
    fs::write(path, payload)
}

fn parse_semver(raw: &str) -> Option<Semver> {
    let trimmed = raw.trim().trim_start_matches('v');
    let token = trimmed
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect::<String>();
    if token.is_empty() {
        return None;
    }

    let mut parts = token.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    Some(Semver {
        major,
        minor,
        patch,
    })
}

fn bump_distance(previous: Semver, current: Semver) -> Option<BumpLevel> {
    if current.major > previous.major {
        Some(BumpLevel::Major)
    } else if current.minor > previous.minor {
        Some(BumpLevel::Minor)
    } else if current.patch > previous.patch {
        Some(BumpLevel::Patch)
    } else {
        None
    }
}

impl Semver {
    fn bump(self, level: BumpLevel) -> Self {
        match level {
            BumpLevel::Patch => Self {
                major: self.major,
                minor: self.minor,
                patch: self.patch.saturating_add(1),
            },
            BumpLevel::Minor => Self {
                major: self.major,
                minor: self.minor.saturating_add(1),
                patch: 0,
            },
            BumpLevel::Major => Self {
                major: self.major.saturating_add(1),
                minor: 0,
                patch: 0,
            },
        }
    }
}

impl std::fmt::Display for Semver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn parse_changed_paths_extracts_new_file_side() {
        let diff = "\
diff --git a/crates/core/src/lib.rs b/crates/core/src/lib.rs\n\
@@ -1 +1 @@\n\
-a\n\
+b\n\
diff --git a/Cargo.toml b/Cargo.toml\n";

        let paths = parse_changed_paths(diff);
        assert!(paths.contains("crates/core/src/lib.rs"));
        assert!(paths.contains("Cargo.toml"));
    }

    #[test]
    fn semver_parse_and_bump_work_for_standard_values() {
        let parsed = parse_semver("v1.2.3").expect("semver");
        assert_eq!(
            parsed.bump(BumpLevel::Minor).to_string(),
            "1.3.0".to_string()
        );
    }

    #[test]
    fn suggested_level_prefers_minor_for_features() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat(core): add feature".to_string(),
            summary: "Add feature".to_string(),
            items: vec![autocommit_core::types::ChangeItem {
                id: "x".to_string(),
                bucket: autocommit_core::types::ChangeBucket::Feature,
                type_tag: TypeTag::Feat,
                title: "Add feature".to_string(),
                intent: "Add feature".to_string(),
                files: Vec::new(),
                confidence: 0.8,
            }],
            risk: autocommit_core::types::RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: autocommit_core::types::DiffStats::default(),
            dispatch: autocommit_core::types::DispatchDecision {
                route: autocommit_core::types::DispatchRoute::DraftOnly,
                reason_codes: Vec::new(),
                estimated_cost_tokens: 0,
            },
        };

        assert_eq!(suggested_level(&report), BumpLevel::Minor);
    }

    #[test]
    fn suggested_level_keeps_patch_for_refactor_subject_with_single_noisy_feature_item() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "refactor(core): simplify parser".to_string(),
            summary: "Refactor parser".to_string(),
            items: vec![autocommit_core::types::ChangeItem {
                id: "x".to_string(),
                bucket: autocommit_core::types::ChangeBucket::Patch,
                type_tag: TypeTag::Feat,
                title: "Simplify parser".to_string(),
                intent: "Refactor parsing".to_string(),
                files: Vec::new(),
                confidence: 0.65,
            }],
            risk: autocommit_core::types::RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: autocommit_core::types::DiffStats::default(),
            dispatch: autocommit_core::types::DispatchDecision {
                route: autocommit_core::types::DispatchRoute::DraftOnly,
                reason_codes: Vec::new(),
                estimated_cost_tokens: 0,
            },
        };

        assert_eq!(suggested_level(&report), BumpLevel::Patch);
    }

    #[test]
    fn suggested_level_keeps_patch_for_feat_subject_without_real_feature_signals() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat(core): refactor internals".to_string(),
            summary: "Refactor internals".to_string(),
            items: vec![autocommit_core::types::ChangeItem {
                id: "x".to_string(),
                bucket: autocommit_core::types::ChangeBucket::Patch,
                type_tag: TypeTag::Refactor,
                title: "Refactor internals".to_string(),
                intent: "Reorganize implementation".to_string(),
                files: Vec::new(),
                confidence: 0.82,
            }],
            risk: autocommit_core::types::RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: autocommit_core::types::DiffStats {
                files_changed: 2,
                lines_changed: 40,
                hunks: 3,
                binary_files: 0,
            },
            dispatch: autocommit_core::types::DispatchDecision {
                route: autocommit_core::types::DispatchRoute::DraftOnly,
                reason_codes: Vec::new(),
                estimated_cost_tokens: 0,
            },
        };

        assert_eq!(suggested_level(&report), BumpLevel::Patch);
    }

    #[test]
    fn suggested_level_returns_minor_for_multi_signal_feature_change() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "feat(core): add workspace config command".to_string(),
            summary: "Add a new command to configure workspace defaults".to_string(),
            items: vec![
                autocommit_core::types::ChangeItem {
                    id: "a".to_string(),
                    bucket: autocommit_core::types::ChangeBucket::Feature,
                    type_tag: TypeTag::Feat,
                    title: "Add `config` command".to_string(),
                    intent: "Add interactive config workflow".to_string(),
                    files: Vec::new(),
                    confidence: 0.91,
                },
                autocommit_core::types::ChangeItem {
                    id: "b".to_string(),
                    bucket: autocommit_core::types::ChangeBucket::Feature,
                    type_tag: TypeTag::Feat,
                    title: "Support persisted defaults".to_string(),
                    intent: "Enable per-repo defaults".to_string(),
                    files: Vec::new(),
                    confidence: 0.86,
                },
            ],
            risk: autocommit_core::types::RiskReport {
                level: "low".to_string(),
                notes: vec!["adds new CLI workflow".to_string()],
            },
            stats: autocommit_core::types::DiffStats {
                files_changed: 4,
                lines_changed: 120,
                hunks: 8,
                binary_files: 0,
            },
            dispatch: autocommit_core::types::DispatchDecision {
                route: autocommit_core::types::DispatchRoute::DraftOnly,
                reason_codes: Vec::new(),
                estimated_cost_tokens: 0,
            },
        };

        assert_eq!(suggested_level(&report), BumpLevel::Minor);
    }

    #[test]
    fn suggested_level_keeps_patch_for_single_noisy_feature_signal() {
        let report = AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "refactor(cli): simplify argument parsing".to_string(),
            summary: "Add support for cleaner parsing internals".to_string(),
            items: vec![autocommit_core::types::ChangeItem {
                id: "x".to_string(),
                bucket: autocommit_core::types::ChangeBucket::Feature,
                type_tag: TypeTag::Feat,
                title: "Support cleaner parsing".to_string(),
                intent: "Refactor parser internals".to_string(),
                files: Vec::new(),
                confidence: 0.92,
            }],
            risk: autocommit_core::types::RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: autocommit_core::types::DiffStats {
                files_changed: 1,
                lines_changed: 42,
                hunks: 3,
                binary_files: 0,
            },
            dispatch: autocommit_core::types::DispatchDecision {
                route: autocommit_core::types::DispatchRoute::DraftOnly,
                reason_codes: Vec::new(),
                estimated_cost_tokens: 0,
            },
        };

        assert_eq!(suggested_level(&report), BumpLevel::Patch);
    }

    #[test]
    fn combine_recommended_level_uses_embedding_only_for_major_escalation() {
        assert_eq!(
            combine_recommended_level(BumpLevel::Patch, Some(BumpLevel::Minor)),
            BumpLevel::Patch
        );
        assert_eq!(
            combine_recommended_level(BumpLevel::Minor, Some(BumpLevel::Patch)),
            BumpLevel::Minor
        );
        assert_eq!(
            combine_recommended_level(BumpLevel::Patch, Some(BumpLevel::Major)),
            BumpLevel::Major
        );
    }

    #[test]
    fn build_recommendation_reports_actual_vs_suggested_bump_distance() {
        let rec = build_recommendation(
            "crates/cli/Cargo.toml",
            ManifestKind::CargoToml,
            Some("1.0.1"),
            Some("1.0.0"),
            true,
            BumpLevel::Minor,
        )
        .expect("recommendation should be produced");

        assert_eq!(
            rec.reason,
            "version bumped by patch, but detected changes suggest at least a minor bump"
        );
        assert_eq!(rec.suggested_version.as_deref(), Some("1.1.0"));
    }

    #[test]
    fn collect_manifest_candidates_maps_to_nearest_subproject_manifest() {
        let root = create_temp_tree();
        write_file(
            &root,
            "Cargo.toml",
            "[workspace]\nmembers = [\"apps/web\", \"crates/core\"]\n",
        );
        write_file(
            &root,
            "apps/web/package.json",
            "{ \"name\": \"web\", \"version\": \"1.2.3\" }\n",
        );
        write_file(
            &root,
            "crates/core/Cargo.toml",
            "[package]\nname = \"core\"\nversion = \"0.4.0\"\n",
        );

        let changed = BTreeSet::from([
            "apps/web/src/index.ts".to_string(),
            "crates/core/src/lib.rs".to_string(),
        ]);

        let candidates = collect_manifest_candidates(&root, &changed);
        assert!(candidates.contains_key("apps/web/package.json"));
        assert!(candidates.contains_key("crates/core/Cargo.toml"));
        assert!(!candidates.contains_key("Cargo.toml"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_manifest_bump_updates_cargo_toml_version() {
        let root = create_temp_tree();
        let manifest = root.join("Cargo.toml");
        write_file(
            &root,
            "Cargo.toml",
            "[package]\nname = \"demo\"\nversion = \"0.4.0\"\n",
        );

        apply_manifest_bump(&manifest, ManifestKind::CargoToml, "0.5.0").expect("cargo bump apply");
        let next = fs::read_to_string(&manifest).expect("read cargo");
        assert!(next.contains("version = \"0.5.0\""));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_manifest_bump_updates_package_json_version() {
        let root = create_temp_tree();
        let manifest = root.join("package.json");
        write_file(
            &root,
            "package.json",
            "{\n  \"name\": \"demo\",\n  \"version\": \"1.2.3\"\n}\n",
        );

        apply_manifest_bump(&manifest, ManifestKind::PackageJson, "1.3.0")
            .expect("package bump apply");
        let next = fs::read_to_string(&manifest).expect("read package");
        assert!(next.contains("\"version\": \"1.3.0\""));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_manifest_bump_updates_go_mod_module_path() {
        let root = create_temp_tree();
        let manifest = root.join("go.mod");
        write_file(&root, "go.mod", "module example.com/demo\n\ngo 1.22\n");

        apply_manifest_bump(&manifest, ManifestKind::GoMod, "v2").expect("go mod bump apply");
        let next = fs::read_to_string(&manifest).expect("read go.mod");
        assert!(next.contains("module example.com/demo/v2"));

        let _ = fs::remove_dir_all(&root);
    }

    fn create_temp_tree() -> PathBuf {
        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("autocommit-version-bump-test-{stamp}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp tree");
        path
    }

    fn write_file(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, contents).expect("write file");
    }
}
