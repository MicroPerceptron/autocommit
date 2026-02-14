use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn core_never_imports_llama_sys() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);

    for file in files {
        let body = fs::read_to_string(&file).expect("read source file");
        assert!(
            !body.contains("llama_sys"),
            "core must not import llama_sys directly: {}",
            file.display()
        );
    }
}

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
}
