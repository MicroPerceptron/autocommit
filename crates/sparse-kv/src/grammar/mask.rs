/// Trie-accelerated token mask computation.
///
/// Walks the token ByteTrie and grammar state in lockstep. When a byte
/// is rejected by the grammar, the entire trie subtree is pruned —
/// potentially skipping thousands of tokens in one check.

use crate::tokenizer::trie::ByteTrie;

use super::parser::CompiledGrammar;
use super::state::GrammarState;
use super::TokenMask;

/// Build a token validity mask by walking the trie with the grammar state.
///
/// For each trie node:
/// - If it has a token ID and the grammar is at an accepting boundary → set bit
/// - For each child byte: try to advance the grammar. If it rejects → prune subtree.
pub fn build_mask(
    grammar: &CompiledGrammar,
    state: &GrammarState,
    trie: &ByteTrie,
    n_vocab: usize,
) -> TokenMask {
    let mut mask = TokenMask::new(n_vocab);
    walk(grammar, state, trie, 0, &mut mask);
    mask
}

fn walk(
    grammar: &CompiledGrammar,
    state: &GrammarState,
    trie: &ByteTrie,
    node_idx: u32,
    mask: &mut TokenMask,
) {
    let node = &trie.nodes[node_idx as usize];

    // If this trie node has a token ID, the token is valid: all its bytes
    // were accepted by the grammar (or we wouldn't have reached this node).
    // We do NOT require is_accepting here — a valid token just needs its
    // bytes to be consumable. Whether generation can STOP is determined
    // separately via is_complete()/is_accepting().
    if let Some(token_id) = node.value {
        mask.set(token_id);
    }

    // Try each child byte. If the grammar can accept it, recurse.
    // If not, the entire subtree is pruned.
    for &(byte, child_idx) in &node.children {
        if let Some(next_state) = state.advance_byte(byte, grammar) {
            walk(grammar, &next_state, trie, child_idx, mask);
        }
        // Grammar rejected this byte → skip all tokens in this subtree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parser::parse_gbnf;

    fn make_test_trie() -> ByteTrie {
        let mut trie = ByteTrie::new();
        // Token 0: "hello"
        trie.insert(b"hello", 0);
        // Token 1: "hi"
        trie.insert(b"hi", 1);
        // Token 2: "123"
        trie.insert(b"123", 2);
        // Token 3: "a"
        trie.insert(b"a", 3);
        // Token 4: "ab"
        trie.insert(b"ab", 4);
        // Token 5: "abc"
        trie.insert(b"abc", 5);
        // Token 6: " " (space)
        trie.insert(b" ", 6);
        // Token 7: "{"
        trie.insert(b"{", 7);
        // Token 8: "}"
        trie.insert(b"}", 8);
        // Token 9: "1"
        trie.insert(b"1", 9);
        trie
    }

    #[test]
    fn mask_exact_literal() {
        let g = parse_gbnf(r#"root ::= "hello""#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let mask = build_mask(&g, &state, &trie, 10);
        // Only token 0 ("hello") should match
        assert!(mask.is_set(0));
        assert!(!mask.is_set(1));
        assert!(!mask.is_set(2));
        assert_eq!(mask.count_set(), 1);
    }

    #[test]
    fn mask_alternatives() {
        let g = parse_gbnf(r#"root ::= "hello" | "hi""#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let mask = build_mask(&g, &state, &trie, 10);
        assert!(mask.is_set(0)); // "hello"
        assert!(mask.is_set(1)); // "hi"
        assert!(!mask.is_set(2)); // "123"
        assert_eq!(mask.count_set(), 2);
    }

    #[test]
    fn mask_char_class_digits() {
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let mask = build_mask(&g, &state, &trie, 10);
        // Token 2 ("123") and token 9 ("1") start with digits
        assert!(mask.is_set(2)); // "123"
        assert!(mask.is_set(9)); // "1"
        assert!(!mask.is_set(0)); // "hello"
        assert!(!mask.is_set(3)); // "a"
    }

    #[test]
    fn mask_letter_class() {
        let g = parse_gbnf(r#"root ::= [a-z]+"#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let mask = build_mask(&g, &state, &trie, 10);
        assert!(mask.is_set(0)); // "hello"
        assert!(mask.is_set(1)); // "hi"
        assert!(mask.is_set(3)); // "a"
        assert!(mask.is_set(4)); // "ab"
        assert!(mask.is_set(5)); // "abc"
        assert!(!mask.is_set(2)); // "123"
        assert!(!mask.is_set(6)); // " "
    }

    #[test]
    fn mask_after_partial_match() {
        let g = parse_gbnf(r#"root ::= "a" [a-z]*"#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        // At initial state, tokens starting with 'a' are valid
        let mask = build_mask(&g, &state, &trie, 10);
        assert!(mask.is_set(3)); // "a"
        assert!(mask.is_set(4)); // "ab"
        assert!(mask.is_set(5)); // "abc"
        assert!(!mask.is_set(0)); // "hello" — doesn't start with 'a'

        // After accepting "a", we're in the [a-z]* part
        let state2 = state.advance_byte(b'a', &g).unwrap();
        let mask2 = build_mask(&g, &state2, &trie, 10);
        // Now any lowercase letter token is valid (continuing the word)
        // Plus tokens that represent "accepting" (empty [a-z]*)
        // Since [a-z]* accepts empty, all valid-prefix tokens count
        assert!(mask2.is_set(0)); // "hello" — all lowercase
        assert!(mask2.is_set(3)); // "a"
        assert!(mask2.is_set(4)); // "ab"
    }

    #[test]
    fn mask_empty_grammar_state() {
        // After accepting all required characters, grammar should accept
        let g = parse_gbnf(r#"root ::= "a""#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let state2 = state.advance_byte(b'a', &g).unwrap();
        // Grammar is complete — no more tokens should be valid
        // (except tokens that represent empty string, which our trie doesn't have)
        let mask = build_mask(&g, &state2, &trie, 10);
        assert_eq!(mask.count_set(), 0);
    }

    #[test]
    fn mask_json_like() {
        let g = parse_gbnf(r#"root ::= "{" "}" "#).unwrap();
        let state = GrammarState::initial(&g);
        let trie = make_test_trie();

        let mask = build_mask(&g, &state, &trie, 10);
        // Only "{" token should be valid initially
        assert!(mask.is_set(7)); // "{"
        assert!(!mask.is_set(8)); // "}" — not yet

        // After accepting "{"
        let state2 = state.advance_byte(b'{', &g).unwrap();
        let mask2 = build_mask(&g, &state2, &trie, 10);
        assert!(mask2.is_set(8)); // "}" — now valid
        assert!(!mask2.is_set(7)); // "{" — not valid
    }

    // --- Real vocab trie tests ---
    //
    // Load actual llama.cpp GGUF vocabs and verify mask building at scale.
    // This validates that trie-accelerated masking produces correct results
    // with 128K+ token vocabularies, not just our 10-token test trie.

    #[test]
    fn mask_real_vocab_digit_grammar() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..");
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
        if !gguf_path.exists() {
            eprintln!("SKIP: real vocab not found at {gguf_path:?}");
            return;
        }

        let reader = crate::gguf::GgufReader::open(&gguf_path).unwrap();
        let vocab = crate::tokenizer::vocab::Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();
        let n_vocab = vocab.n_vocab();

        // Build token trie from vocab
        let trie = &vocab.token_trie;

        // Grammar: [0-9]+ (one or more digits)
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        let state = GrammarState::initial(&g);

        let mask = build_mask(&g, &state, trie, n_vocab);
        let count = mask.count_set();
        eprintln!("digit grammar [0-9]+ on llama-bpe vocab ({n_vocab} tokens): {count} valid tokens");

        // Sanity: there should be tokens for "0"-"9", "00"-"99", "000"-"999", etc.
        // Expect at least 10 (single digits) and less than n_vocab
        assert!(count >= 10, "expected at least 10 digit tokens, got {count}");
        assert!(count < n_vocab, "grammar should restrict vocab");

        // Verify known tokens by looking them up
        // Token "0" should be valid
        if let Some(id) = trie.get(b"0") {
            assert!(mask.is_set(id), "token '0' (id={id}) should be valid for [0-9]+");
        }
        // Token "hello" should NOT be valid
        if let Some(id) = trie.get(b"hello") {
            assert!(!mask.is_set(id), "token 'hello' (id={id}) should not be valid for [0-9]+");
        }
    }

    #[test]
    fn mask_real_vocab_literal_grammar() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..");
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
        if !gguf_path.exists() {
            return;
        }

        let reader = crate::gguf::GgufReader::open(&gguf_path).unwrap();
        let vocab = crate::tokenizer::vocab::Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();
        let n_vocab = vocab.n_vocab();
        let trie = &vocab.token_trie;

        // Grammar: exactly "true"
        let g = parse_gbnf(r#"root ::= "true""#).unwrap();
        let state = GrammarState::initial(&g);
        let mask = build_mask(&g, &state, trie, n_vocab);
        let count = mask.count_set();
        eprintln!("literal \"true\" on llama-bpe vocab: {count} valid tokens");

        // Only tokens that are prefixes of "true" should be valid:
        // "t", "tr", "tru", "true" (and any merged versions)
        assert!(count >= 1, "at least 'true' token should be valid");
        assert!(count < 20, "only a handful of prefix tokens, got {count}");

        // After consuming "tru", only tokens starting with "e" are valid
        let mut s = state.clone();
        for &b in b"tru" {
            s = s.advance_byte(b, &g).unwrap();
        }
        let mask2 = build_mask(&g, &s, trie, n_vocab);
        let count2 = mask2.count_set();
        eprintln!("after 'tru', valid tokens: {count2}");
        assert!(count2 >= 1, "token 'e' should be valid after 'tru'");
    }

    #[test]
    fn mask_real_vocab_key_value_grammar() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..");
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
        if !gguf_path.exists() {
            return;
        }

        let reader = crate::gguf::GgufReader::open(&gguf_path).unwrap();
        let vocab = crate::tokenizer::vocab::Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();
        let n_vocab = vocab.n_vocab();
        let trie = &vocab.token_trie;

        // Multi-rule grammar
        let g = parse_gbnf(
            "root ::= key \":\" value\n\
             key ::= [a-z]+\n\
             value ::= [0-9]+"
        ).unwrap();
        let state = GrammarState::initial(&g);

        let mask = build_mask(&g, &state, trie, n_vocab);
        let count = mask.count_set();
        eprintln!("key:value grammar initial state: {count} valid tokens");

        // Initially: lowercase letter tokens and lowercase-starting multi-byte tokens
        assert!(count > 20, "many lowercase tokens expected, got {count}");

        // After "key:", only digit tokens should be valid
        let mut s = state.clone();
        for &b in b"abc:" {
            s = s.advance_byte(b, &g).unwrap();
        }
        let mask2 = build_mask(&g, &s, trie, n_vocab);
        let count2 = mask2.count_set();
        eprintln!("after 'abc:', valid tokens: {count2}");
        assert!(count2 >= 10, "digit tokens should be valid after ':'");

        // Verify a letter token is NOT valid after ':'
        if let Some(id) = trie.get(b"a") {
            assert!(!mask2.is_set(id), "'a' should not be valid in value position");
        }
    }

    #[test]
    fn mask_real_vocab_trie_pruning_efficiency() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..");
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
        if !gguf_path.exists() {
            return;
        }

        let reader = crate::gguf::GgufReader::open(&gguf_path).unwrap();
        let vocab = crate::tokenizer::vocab::Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();
        let n_vocab = vocab.n_vocab();
        let trie = &vocab.token_trie;

        // Restrictive grammar: only digits
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        let state = GrammarState::initial(&g);

        let mask = build_mask(&g, &state, trie, n_vocab);
        let valid = mask.count_set();
        let reject_ratio = 1.0 - (valid as f64 / n_vocab as f64);
        eprintln!(
            "trie pruning: {valid}/{n_vocab} valid ({:.1}% rejected)",
            reject_ratio * 100.0
        );

        // With [0-9]+, the vast majority of 128K tokens should be rejected.
        // Expect >95% rejection (only digit-starting tokens pass).
        assert!(
            reject_ratio > 0.95,
            "expected >95% rejection for digit-only grammar, got {:.1}%",
            reject_ratio * 100.0
        );
    }
}
