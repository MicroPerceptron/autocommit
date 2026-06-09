use std::io::Write;

use dialoguer::console::Term;
use dialoguer::{Confirm, Select, theme::ColorfulTheme};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CommitPolicy {
    #[serde(default)]
    pub(crate) sign_commits: bool,
    #[serde(default)]
    pub(crate) add_signoff: bool,
    #[serde(default)]
    pub(crate) enforce_conventional_subject: bool,
    #[serde(default)]
    pub(crate) enforce_subject_limit: bool,
    #[serde(default = "default_subject_limit")]
    pub(crate) subject_limit: u16,
}

impl Default for CommitPolicy {
    fn default() -> Self {
        Self {
            sign_commits: false,
            add_signoff: false,
            enforce_conventional_subject: false,
            enforce_subject_limit: false,
            subject_limit: default_subject_limit(),
        }
    }
}

const fn default_subject_limit() -> u16 {
    72
}

pub(crate) struct PolicyPromptOutcome {
    pub(crate) policy: CommitPolicy,
}

pub(crate) fn prompt_commit_policy_setup(
    existing: Option<&CommitPolicy>,
    rich: bool,
) -> Result<PolicyPromptOutcome, String> {
    if rich {
        prompt_commit_policy_setup_rich(existing)
    } else {
        prompt_commit_policy_setup_basic(existing)
    }
}

#[cfg(feature = "llama-native")]
pub(crate) fn policy_summary(policy: &CommitPolicy) -> String {
    let mut parts = Vec::new();
    if policy.sign_commits {
        parts.push("signed commits");
    }
    if policy.add_signoff {
        parts.push("DCO signoff");
    }
    if policy.enforce_conventional_subject {
        parts.push("conventional subject");
    }
    if policy.enforce_subject_limit {
        parts.push("subject length limit");
    }
    if parts.is_empty() {
        "defaults (no branch-rule extras)".to_string()
    } else {
        format!(
            "{}{}",
            parts.join(", "),
            if policy.enforce_subject_limit {
                format!("={}", policy.subject_limit)
            } else {
                String::new()
            }
        )
    }
}

pub(crate) fn append_signoff_trailer(message: &str, identity: &str) -> String {
    let trimmed_identity = identity.trim();
    if trimmed_identity.is_empty() {
        return message.trim().to_string();
    }

    let trailer = format!("Signed-off-by: {trimmed_identity}");
    if has_case_insensitive_line(message, &trailer) {
        return message.trim().to_string();
    }

    let mut out = message.trim_end().to_string();
    if out.is_empty() {
        return trailer;
    }
    out.push_str("\n\n");
    out.push_str(&trailer);
    out
}

pub(crate) fn validate_commit_message(message: &str, policy: &CommitPolicy) -> Result<(), String> {
    let subject = message.lines().next().unwrap_or_default().trim();
    if subject.is_empty() {
        return Err("generated commit subject is empty".to_string());
    }

    if policy.enforce_subject_limit {
        let max_len = usize::from(policy.subject_limit.max(1));
        let len = subject.chars().count();
        if len > max_len {
            return Err(format!(
                "commit subject exceeds configured limit ({len} > {max_len})"
            ));
        }
    }

    if policy.enforce_conventional_subject && !is_conventional_subject(subject) {
        return Err(
            "commit subject must follow Conventional Commits (`type(scope): description`)"
                .to_string(),
        );
    }

    Ok(())
}

fn prompt_commit_policy_setup_rich(
    existing: Option<&CommitPolicy>,
) -> Result<PolicyPromptOutcome, String> {
    let term = Term::stderr();
    let theme = ColorfulTheme::default();
    let mut policy = existing.cloned().unwrap_or_default();

    let configure_now = Confirm::with_theme(&theme)
        .with_prompt("Configure one-time commit policy for this repo?")
        .default(existing.is_none())
        .interact_on(&term)
        .map_err(|err| format!("failed to read commit policy confirmation: {err}"))?;
    if !configure_now {
        return Ok(PolicyPromptOutcome { policy });
    }

    policy.sign_commits = Confirm::with_theme(&theme)
        .with_prompt("Require cryptographically signed commits (`git -S`)?")
        .default(policy.sign_commits)
        .interact_on(&term)
        .map_err(|err| format!("failed to read signed-commit policy: {err}"))?;

    policy.add_signoff = Confirm::with_theme(&theme)
        .with_prompt("Auto-add DCO signoff trailer (`Signed-off-by`)?")
        .default(policy.add_signoff)
        .interact_on(&term)
        .map_err(|err| format!("failed to read signoff policy: {err}"))?;

    policy.enforce_conventional_subject = Confirm::with_theme(&theme)
        .with_prompt("Enforce Conventional Commit subject format?")
        .default(policy.enforce_conventional_subject)
        .interact_on(&term)
        .map_err(|err| format!("failed to read conventional-subject policy: {err}"))?;

    policy.enforce_subject_limit = Confirm::with_theme(&theme)
        .with_prompt("Enforce a commit subject length limit?")
        .default(policy.enforce_subject_limit)
        .interact_on(&term)
        .map_err(|err| format!("failed to read subject-length policy: {err}"))?;

    if policy.enforce_subject_limit {
        let limits = [50u16, 72u16, 100u16];
        let default_idx = limits
            .iter()
            .position(|limit| *limit == policy.subject_limit)
            .unwrap_or(1);
        let labels = limits
            .iter()
            .map(|value| format!("{value} chars"))
            .collect::<Vec<_>>();
        let selection = Select::with_theme(&theme)
            .with_prompt("Subject length limit")
            .items(&labels)
            .default(default_idx)
            .interact_on_opt(&term)
            .map_err(|err| format!("failed to read subject length limit: {err}"))?;
        if let Some(index) = selection {
            policy.subject_limit = limits[index];
        }
    }

    Ok(PolicyPromptOutcome { policy })
}

fn prompt_commit_policy_setup_basic(
    existing: Option<&CommitPolicy>,
) -> Result<PolicyPromptOutcome, String> {
    let mut policy = existing.cloned().unwrap_or_default();

    if !prompt_yes_no_basic(
        "Configure one-time commit policy for this repo? [Y/n]: ",
        existing.is_none(),
    )? {
        return Ok(PolicyPromptOutcome { policy });
    }

    policy.sign_commits = prompt_yes_no_basic(
        "Require cryptographically signed commits (`git -S`)? [y/N]: ",
        policy.sign_commits,
    )?;
    policy.add_signoff = prompt_yes_no_basic(
        "Auto-add DCO signoff trailer (`Signed-off-by`)? [y/N]: ",
        policy.add_signoff,
    )?;
    policy.enforce_conventional_subject = prompt_yes_no_basic(
        "Enforce Conventional Commit subject format? [y/N]: ",
        policy.enforce_conventional_subject,
    )?;
    policy.enforce_subject_limit = prompt_yes_no_basic(
        "Enforce a commit subject length limit? [y/N]: ",
        policy.enforce_subject_limit,
    )?;
    if policy.enforce_subject_limit {
        policy.subject_limit = prompt_subject_limit_basic(policy.subject_limit)?;
    }

    Ok(PolicyPromptOutcome { policy })
}

fn prompt_yes_no_basic(prompt: &str, default: bool) -> Result<bool, String> {
    loop {
        print!("{prompt}");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(default);
        }
        match value.to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("invalid choice, enter y or n"),
        }
    }
}

fn prompt_subject_limit_basic(default: u16) -> Result<u16, String> {
    loop {
        print!("Subject length limit [50/72/100] (default {default}): ");
        std::io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush prompt output: {err}"))?;
        let value = read_line_trimmed()?;
        if value.is_empty() {
            return Ok(default);
        }
        match value.parse::<u16>() {
            Ok(parsed @ 50) | Ok(parsed @ 72) | Ok(parsed @ 100) => return Ok(parsed),
            _ => println!("invalid choice, enter 50, 72, or 100"),
        }
    }
}

fn read_line_trimmed() -> Result<String, String> {
    let mut buffer = String::new();
    let read = std::io::stdin()
        .read_line(&mut buffer)
        .map_err(|err| format!("failed to read prompt input: {err}"))?;
    if read == 0 {
        return Err("interactive input was closed".to_string());
    }
    Ok(buffer.trim().to_string())
}

fn has_case_insensitive_line(message: &str, needle: &str) -> bool {
    message
        .lines()
        .any(|line| line.trim().eq_ignore_ascii_case(needle.trim()))
}

fn is_conventional_subject(subject: &str) -> bool {
    let Some((prefix, description)) = subject.split_once(':') else {
        return false;
    };
    if !description.starts_with(' ') || description.trim().is_empty() {
        return false;
    }

    let mut prefix = prefix.trim();
    if prefix.ends_with('!') {
        prefix = prefix[..prefix.len() - 1].trim_end();
    }
    if prefix.is_empty() {
        return false;
    }

    let (type_part, scope_part) = if let Some(open_idx) = prefix.find('(') {
        let Some(close_idx) = prefix.rfind(')') else {
            return false;
        };
        if close_idx + 1 != prefix.len() || close_idx <= open_idx + 1 {
            return false;
        }
        let type_part = &prefix[..open_idx];
        let scope = &prefix[open_idx + 1..close_idx];
        (type_part, Some(scope))
    } else {
        (prefix, None)
    };

    if !is_valid_type(type_part) {
        return false;
    }
    if let Some(scope) = scope_part
        && !is_valid_scope(scope)
    {
        return false;
    }
    true
}

fn is_valid_type(value: &str) -> bool {
    let value = value.trim();
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

fn is_valid_scope(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '/' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_signoff_only_once() {
        let message = "feat(cli): add prompt";
        let identity = "Example User <example@user.dev>";
        let once = append_signoff_trailer(message, identity);
        let twice = append_signoff_trailer(&once, identity);
        assert_eq!(once, twice);
        assert!(once.contains("Signed-off-by: Example User <example@user.dev>"));
    }

    #[test]
    fn conventional_subject_validation() {
        assert!(is_conventional_subject("feat(cli): add prompt"));
        assert!(is_conventional_subject("fix(core)!: remove old behavior"));
        assert!(!is_conventional_subject("Refactor CLI output"));
        assert!(!is_conventional_subject("feat:"));
    }
}
