use std::fs;
use std::process::Command;

#[test]
fn analyze_json_roundtrip() {
    let diff = "diff --git a/src/lib.rs b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n";
    let path = std::env::temp_dir().join(format!(
        "autocommit-cli-test-{}-{}.diff",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_millis()
    ));
    fs::write(&path, diff).expect("write diff fixture");

    let output = Command::new(env!("CARGO_BIN_EXE_autocommit-cli"))
        .args([
            "analyze",
            "--json",
            "--diff-file",
            path.to_str().expect("path"),
        ])
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("valid json output");
    assert_eq!(parsed["schema_version"], "1.0");
    assert!(parsed["items"].is_array());

    let _ = fs::remove_file(path);
}
