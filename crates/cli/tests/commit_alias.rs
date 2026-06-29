use std::process::Command;

fn run_autocommit(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_autocommit"))
        .args(args)
        .output()
        .expect("run autocommit")
}

#[test]
fn short_commit_alias_forwards_to_commit_help() {
    let commit = run_autocommit(&["commit", "--help"]);
    let alias = run_autocommit(&["c", "--help"]);

    assert!(
        commit.status.success(),
        "commit --help stderr: {}",
        String::from_utf8_lossy(&commit.stderr)
    );
    assert!(
        alias.status.success(),
        "c --help stderr: {}",
        String::from_utf8_lossy(&alias.stderr)
    );
    assert_eq!(commit.stdout, alias.stdout);
}

#[test]
fn top_level_help_documents_short_commit_alias() {
    let output = run_autocommit(&["--help"]);

    assert!(
        output.status.success(),
        "--help stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("commit"), "help output: {stdout}");
    assert!(stdout.contains("c"), "help output: {stdout}");
    assert!(
        stdout.contains("Short aliases: `autocommit c` runs `autocommit commit`."),
        "help output: {stdout}"
    );
}
