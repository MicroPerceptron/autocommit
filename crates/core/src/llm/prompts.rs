use crate::types::DiffChunk;

pub const SYSTEM_PROMPT: &str = r#"You are a git commit message generator. Given a diff, output a descriptive commit message.

Format:
- First line: type(scope): short description
- Types: feat, fix, refactor, docs, test, chore, perf, style
- Scope: primary module/file affected (optional)
- Blank line, then bullet points for details if non-trivial

Guidelines:
- Describe WHAT changed and WHY, not HOW (the diff itself shows how)
- Infer intent from: log messages, variable names, comments, control flow patterns
- For bug fixes: "fix: prevent X when Y" not "fix: add null check"
- For features: "feat: add X for Y" describing capability, not implementation
- For refactors: "refactor: extract/simplify/reorganize X" with motivation if clear
- For diagnostics/debugging: "fix/chore: add diagnostic for X"
- Multiple related changes: summarize the theme, then list key parts
- Unrelated changes: indicate mixed scope clearly
"#;

pub fn build_analyze_prompt(chunk: &DiffChunk) -> String {
    format!(
        "Task: Analyze one diff chunk.\n\
Return ONLY JSON with keys: summary, bucket, type_tag, title, intent.\n\
Allowed bucket values: Feature, Patch, Addition, Other.\n\
Allowed type_tag values: Feat, Fix, Refactor, Docs, Test, Chore, Perf, Style, Mixed.\n\
Path: {}\n\
Diff:\n```diff\n{}\n```",
        chunk.path, chunk.text
    )
}

pub fn build_reduce_prompt(partial_count: usize) -> String {
    format!(
        "/no_think\n\
Task: Produce final commit metadata from {partial_count} chunk summaries.\n\
Return ONLY JSON with this exact shape:\n\
{{\"commit_message\":\"...\",\"summary\":\"...\",\"risk_level\":\"low|medium|high\",\"risk_notes\":[\"...\"]}}\n\
Rules:\n\
- commit_message must be a single conventional commit header that describes concrete code changes\n\
- commit_message must not mention analysis process words like: reduce, reducer, analysis, analyses, report, chunk, or partial\n\
- summary must be one sentence about the code change outcome\n\
- risk_level must be low, medium, or high\n\
- risk_notes should be concise and concrete\n\
- absolutely no explanations, no markdown, no <think> tags"
    )
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
