/// Extracts key symbol declarations from diff `+` lines.
///
/// Scans added lines for common declaration patterns (functions, structs,
/// traits, classes, etc.) across multiple languages. Returns up to 10
/// deduplicated symbol names that can be used in the reduce prompt to
/// give the LLM concrete nouns/verbs for the commit title.

use crate::types::DiffChunk;

/// Maximum number of symbols to return.
const MAX_SYMBOLS: usize = 10;

/// Extract key symbol names from added lines in diff chunks.
pub fn extract_key_symbols(chunks: &[DiffChunk]) -> Vec<String> {
    let mut symbols = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for chunk in chunks {
        for line in chunk.text.lines() {
            // Only scan added lines (starting with +, but not +++ header).
            let trimmed = if let Some(rest) = line.strip_prefix('+') {
                if rest.starts_with("++") {
                    continue;
                }
                rest.trim()
            } else {
                continue;
            };

            if trimmed.is_empty() {
                continue;
            }

            if let Some(name) = extract_symbol(trimmed) {
                if !name.is_empty() && seen.insert(name.clone()) {
                    symbols.push(name);
                    if symbols.len() >= MAX_SYMBOLS {
                        return symbols;
                    }
                }
            }
        }
    }

    symbols
}

/// Try to extract a symbol name from a single added line.
fn extract_symbol(line: &str) -> Option<String> {
    // Rust: pub fn, pub struct, pub trait, pub enum, pub mod, impl
    if let Some(rest) = line
        .strip_prefix("pub fn ")
        .or_else(|| line.strip_prefix("pub(crate) fn "))
        .or_else(|| line.strip_prefix("fn "))
    {
        return extract_identifier(rest);
    }
    if let Some(rest) = line
        .strip_prefix("pub struct ")
        .or_else(|| line.strip_prefix("pub(crate) struct "))
        .or_else(|| line.strip_prefix("struct "))
    {
        return extract_identifier(rest);
    }
    if let Some(rest) = line
        .strip_prefix("pub trait ")
        .or_else(|| line.strip_prefix("pub(crate) trait "))
        .or_else(|| line.strip_prefix("trait "))
    {
        return extract_identifier(rest);
    }
    if let Some(rest) = line
        .strip_prefix("pub enum ")
        .or_else(|| line.strip_prefix("pub(crate) enum "))
        .or_else(|| line.strip_prefix("enum "))
    {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("pub mod ").or_else(|| line.strip_prefix("mod ")) {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("impl ") {
        // Skip lifetime/generic prefix to get the type name.
        let rest = rest.trim_start_matches(|c: char| c == '<' || c == '\'');
        return extract_identifier(rest);
    }

    // Python: def, class
    if let Some(rest) = line.strip_prefix("def ") {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("class ") {
        return extract_identifier(rest);
    }
    // Async Python
    if let Some(rest) = line.strip_prefix("async def ") {
        return extract_identifier(rest);
    }

    // JavaScript/TypeScript: export function, export class, export const
    if let Some(rest) = line
        .strip_prefix("export function ")
        .or_else(|| line.strip_prefix("export async function "))
    {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("export class ") {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("export const ") {
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("export default function ") {
        return extract_identifier(rest);
    }
    // Plain function/class (non-exported)
    if let Some(rest) = line
        .strip_prefix("function ")
        .or_else(|| line.strip_prefix("async function "))
    {
        return extract_identifier(rest);
    }

    // Go: func, type ... struct/interface
    if let Some(rest) = line.strip_prefix("func ") {
        // Skip receiver: func (s *Server) MethodName(...)
        let rest = if rest.starts_with('(') {
            // Find closing paren, then skip whitespace.
            rest.find(')')
                .and_then(|i| rest.get(i + 1..))
                .map(|s| s.trim())
                .unwrap_or(rest)
        } else {
            rest
        };
        return extract_identifier(rest);
    }
    if let Some(rest) = line.strip_prefix("type ") {
        // "type Foo struct" or "type Bar interface"
        if rest.contains(" struct") || rest.contains(" interface") {
            return extract_identifier(rest);
        }
    }

    None
}

/// Extract an identifier from the start of a string.
/// Stops at the first non-alphanumeric, non-underscore character.
fn extract_identifier(s: &str) -> Option<String> {
    let name: String = s
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();

    if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LineRange;

    fn chunk_with_diff(text: &str) -> DiffChunk {
        DiffChunk {
            path: "test.rs".to_string(),
            text: text.to_string(),
            ranges: vec![LineRange {
                old_start: 1,
                old_count: 1,
                new_start: 1,
                new_count: 1,
            }],
            estimated_tokens: 100,
        }
    }

    #[test]
    fn extracts_rust_declarations() {
        let diff = "\
+pub fn validate_input(s: &str) -> bool {
+pub struct McpServer {
+pub trait Handler {
+pub enum Status {
+mod internal;
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(
            symbols,
            vec!["validate_input", "McpServer", "Handler", "Status", "internal"]
        );
    }

    #[test]
    fn extracts_python_declarations() {
        let diff = "\
+def process_request(data):
+class RequestHandler:
+async def fetch_data(url):
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["process_request", "RequestHandler", "fetch_data"]);
    }

    #[test]
    fn extracts_js_ts_declarations() {
        let diff = "\
+export function createServer(config) {
+export class ApiClient {
+export const DEFAULT_TIMEOUT = 5000;
+function helperFn() {
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(
            symbols,
            vec!["createServer", "ApiClient", "DEFAULT_TIMEOUT", "helperFn"]
        );
    }

    #[test]
    fn extracts_go_declarations() {
        let diff = "\
+func NewServer(cfg Config) *Server {
+func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
+type Router struct {
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["NewServer", "HandleRequest", "Router"]);
    }

    #[test]
    fn skips_diff_header_lines() {
        let diff = "\
+++ b/src/lib.rs
+pub fn real_function() {
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["real_function"]);
    }

    #[test]
    fn deduplicates_symbols() {
        let diff = "\
+pub fn process() {
+pub fn process() {
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["process"]);
    }

    #[test]
    fn caps_at_max_symbols() {
        let mut lines = String::new();
        for i in 0..15 {
            lines.push_str(&format!("+pub fn func_{i}() {{\n"));
        }
        let symbols = extract_key_symbols(&[chunk_with_diff(&lines)]);
        assert_eq!(symbols.len(), MAX_SYMBOLS);
    }

    #[test]
    fn ignores_removed_lines() {
        let diff = "\
-pub fn old_function() {
+pub fn new_function() {
";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["new_function"]);
    }

    #[test]
    fn handles_impl_blocks() {
        let diff = "+impl McpServer {\n";
        let symbols = extract_key_symbols(&[chunk_with_diff(diff)]);
        assert_eq!(symbols, vec!["McpServer"]);
    }
}
