use crate::types::DiffChunk;

pub fn build_analyze_prompt(chunk: &DiffChunk) -> String {
    format!(
        "Analyze this diff chunk and summarize intent.\\nPath: {}\\n\\n{}",
        chunk.path, chunk.text
    )
}

pub fn build_reduce_prompt(partial_count: usize) -> String {
    format!("Reduce {partial_count} partial analyses into one structured commit report.")
}

pub fn build_embedding_prompt(chunks: &[DiffChunk], max_chars: usize) -> String {
    let mut out = String::new();
    for chunk in chunks {
        if out.len() >= max_chars {
            break;
        }
        out.push_str("Path: ");
        out.push_str(&chunk.path);
        out.push('\n');
        for line in chunk.text.lines().take(10) {
            out.push_str(line);
            out.push('\n');
            if out.len() >= max_chars {
                break;
            }
        }
    }
    out
}
