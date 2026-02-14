mod cmd;
mod output;

fn main() {
    if let Err(err) = run(std::env::args().skip(1).collect()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("analyze") => {
            let output = cmd::analyze::run(&args[1..])?;
            print!("{output}");
            Ok(())
        }
        Some("commit") => {
            let output = cmd::commit::run(&args[1..])?;
            print!("{output}");
            Ok(())
        }
        Some("explain-dispatch") => {
            let output = cmd::explain_dispatch::run(&args[1..]);
            print!("{output}");
            Ok(())
        }
        Some("init") => {
            let output = cmd::init::run(&args[1..])?;
            print!("{output}");
            Ok(())
        }
        Some("--help") | Some("-h") | None => {
            println!("autocommit-cli <analyze|commit|explain-dispatch|init> [options]");
            Ok(())
        }
        Some(other) => Err(format!("unknown command: {other}")),
    }
}
