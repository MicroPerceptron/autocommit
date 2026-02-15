use autocommit_core::diff::features::DiffFeatures;
use autocommit_core::dispatch::policy;
use clap::Parser;

pub fn run(args: &[String]) -> String {
    match ExplainDispatchArgs::parse_from(args) {
        Ok(ParseOutcome::Continue(_)) => {}
        Ok(ParseOutcome::EarlyExit(text)) => return text,
        Err(err) => return err,
    }

    let sample = DiffFeatures {
        files_changed: 4,
        lines_changed: 220,
        hunks: 8,
        binary_files: 0,
        risky_paths: 0,
    };

    let decision = policy::decide(&sample, Some(0.81));
    format!(
        "route={:?} reasons={} estimated_tokens={}\n",
        decision.route,
        decision.reason_codes.join(","),
        decision.estimated_cost_tokens
    )
}

enum ParseOutcome<T> {
    Continue(T),
    EarlyExit(String),
}

#[derive(Parser, Debug)]
#[command(
    name = "autocommit-cli explain-dispatch",
    about = "Show dispatch policy routing for a representative sample"
)]
struct ExplainDispatchArgs {}

impl ExplainDispatchArgs {
    fn parse_from(args: &[String]) -> Result<ParseOutcome<Self>, String> {
        let argv = std::iter::once("autocommit-cli explain-dispatch".to_string())
            .chain(args.iter().cloned());
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
