mod cmd;
mod output;
mod path_util;

use clap::{CommandFactory, Parser, ValueEnum};

#[derive(Parser)]
#[command(
    name = "autocommit",
    version,
    about = "AI-assisted commit and pull request generation for local git repositories",
    disable_help_flag = true,
    disable_version_flag = true,
    after_help = "Use `autocommit <command> --help` for command-specific options.\nGlobal flags supported: `--help`, `-h`, `--version`, `-V`."
)]
struct Cli {
    #[arg(value_name = "COMMAND", value_enum)]
    command: Option<CommandName>,
    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    args: Vec<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CommandName {
    /// Analyze git diff and emit a structured report
    Analyze,
    /// Generate and create a commit message from local changes
    Commit,
    /// Remove persisted per-repo KV generation cache
    Clean,
    /// View and edit per-repo runtime and policy config
    Config,
    /// Explain dispatch routing for a sample change profile
    ExplainDispatch,
    /// Initialize per-repo runtime cache and settings
    Init,
    /// Generate and optionally create/update a pull request
    Pr,
}

fn main() {
    if let Err(err) = run(std::env::args().skip(1).collect()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if let Some(first) = args.first()
        && first == "help" {
            if args.len() == 1 {
                let mut cmd = Cli::command();
                cmd.print_help()
                    .map_err(|err| format!("failed to render help output: {err}"))?;
                println!();
                return Ok(());
            }
            let mut forwarded = vec![args[1].clone(), "--help".to_string()];
            forwarded.extend(args.iter().skip(2).cloned());
            return run(forwarded);
        }

    if args.is_empty() || matches!(args.as_slice(), [flag] if flag == "-h" || flag == "--help") {
        let mut cmd = Cli::command();
        cmd.print_help()
            .map_err(|err| format!("failed to render help output: {err}"))?;
        println!();
        return Ok(());
    }
    if matches!(args.as_slice(), [flag] if flag == "-V" || flag == "--version") {
        println!("autocommit {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let argv = std::iter::once("autocommit".to_string()).chain(args);
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(err) => {
            use clap::error::ErrorKind;
            return match err.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    print!("{err}");
                    Ok(())
                }
                _ => Err(err.to_string()),
            };
        }
    };

    match cli.command {
        Some(CommandName::Analyze) => {
            let output = cmd::analyze::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        Some(CommandName::Commit) => {
            let output = cmd::commit::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        Some(CommandName::Clean) => {
            let output = cmd::clean::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        Some(CommandName::Config) => {
            let output = cmd::config::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        Some(CommandName::Pr) => {
            let output = cmd::pr::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        Some(CommandName::ExplainDispatch) => {
            let output = cmd::explain_dispatch::run(&cli.args);
            print!("{output}");
            Ok(())
        }
        Some(CommandName::Init) => {
            let output = cmd::init::run(&cli.args)?;
            print!("{output}");
            Ok(())
        }
        None => Err("missing command; run `autocommit --help`".to_string()),
    }
}
