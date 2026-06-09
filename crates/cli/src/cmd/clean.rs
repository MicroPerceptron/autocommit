#[cfg(feature = "llama-native")]
use std::io::IsTerminal;
#[cfg(feature = "llama-native")]
use std::path::Path;

use clap::Parser;
#[cfg(feature = "llama-native")]
use dialoguer::Confirm;
#[cfg(feature = "llama-native")]
use dialoguer::console::Term;
#[cfg(feature = "llama-native")]
use dialoguer::theme::ColorfulTheme;

#[cfg(feature = "llama-native")]
use crate::cmd::repo_cache;

pub fn run(args: &[String]) -> Result<String, String> {
    let parsed = match CleanArgs::parse_from(args)? {
        ParseOutcome::Continue(parsed) => parsed,
        ParseOutcome::EarlyExit(text) => return Ok(text),
    };
    let assume_yes = parsed.yes;

    #[cfg(feature = "llama-native")]
    {
        run_native(assume_yes)
    }

    #[cfg(not(feature = "llama-native"))]
    {
        let _ = assume_yes;
        Err("clean requires llama-native feature at build time".to_string())
    }
}

#[cfg(feature = "llama-native")]
fn run_native(assume_yes: bool) -> Result<String, String> {
    let paths = repo_cache::discover_repo_kv_paths()
        .map_err(|err| format!("failed to resolve repository paths: {err}"))?;

    let state_path = paths.generation_state;
    if !state_path.is_file() {
        return Ok(format!(
            "no persisted KV cache found at `{}`\n",
            state_path.display()
        ));
    }

    let bytes = file_size(&state_path).map_err(|err| {
        format!(
            "failed to read cache size for `{}`: {err}",
            state_path.display()
        )
    })?;

    let should_delete = if assume_yes {
        true
    } else {
        if !std::io::stdin().is_terminal() || !Term::stderr().is_term() {
            return Err(
                "clean requires an interactive terminal for confirmation; use `--yes` to bypass"
                    .to_string(),
            );
        }

        Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Erase persisted KV cache at `{}` ({}). Continue?",
                state_path.display(),
                human_bytes(bytes)
            ))
            .default(false)
            .interact_on(&Term::stderr())
            .map_err(|err| format!("failed to read confirmation: {err}"))?
    };

    if !should_delete {
        return Ok("canceled; KV cache was not deleted\n".to_string());
    }

    std::fs::remove_file(&state_path)
        .map_err(|err| format!("failed to delete `{}`: {err}", state_path.display()))?;

    Ok(format!(
        "deleted persisted KV cache `{}` (freed {})\n",
        state_path.display(),
        human_bytes(bytes)
    ))
}

#[cfg(feature = "llama-native")]
fn file_size(path: &Path) -> Result<u64, std::io::Error> {
    Ok(std::fs::metadata(path)?.len())
}

#[cfg(feature = "llama-native")]
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0usize;

    while value >= 1024.0 && unit_idx + 1 < UNITS.len() {
        value /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{value:.2} {}", UNITS[unit_idx])
    }
}

enum ParseOutcome<T> {
    Continue(T),
    EarlyExit(String),
}

#[derive(Parser, Debug)]
#[command(
    name = "autocommit clean",
    about = "Delete persisted per-repo KV generation cache"
)]
struct CleanArgs {
    /// Skip confirmation and delete cache immediately
    #[arg(long, short = 'y')]
    yes: bool,
}

impl CleanArgs {
    fn parse_from(args: &[String]) -> Result<ParseOutcome<Self>, String> {
        let argv = std::iter::once("autocommit clean".to_string()).chain(args.iter().cloned());
        match Self::try_parse_from(argv) {
            Ok(parsed) => Ok(ParseOutcome::Continue(parsed)),
            Err(err) => {
                use clap::error::ErrorKind;
                match err.kind() {
                    ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                        Ok(ParseOutcome::EarlyExit(err.to_string()))
                    }
                    _ => Err(err.to_string()),
                }
            }
        }
    }
}
