/// Grammar-constrained sampling with index masking.
///
/// Parses GBNF grammars, tracks parse state via a pushdown automaton,
/// and computes token validity masks by walking the token trie in lockstep
/// with the grammar — pruning entire subtrees when a byte is rejected.

pub mod parser;
pub mod state;
pub mod mask;

use std::collections::HashMap;

use crate::error::InferenceError;
use crate::tokenizer::trie::ByteTrie;

use parser::CompiledGrammar;
use state::GrammarState;

/// Bitmask over the token vocabulary. Bit i = 1 means token i is allowed.
#[derive(Clone)]
pub struct TokenMask {
    bits: Vec<u64>,
    n_vocab: usize,
}

impl TokenMask {
    /// All tokens disallowed.
    pub fn new(n_vocab: usize) -> Self {
        Self {
            bits: vec![0u64; (n_vocab + 63) / 64],
            n_vocab,
        }
    }

    /// All tokens allowed.
    pub fn allow_all(n_vocab: usize) -> Self {
        let full_words = n_vocab / 64;
        let remainder = n_vocab % 64;
        let mut bits = vec![u64::MAX; full_words];
        if remainder > 0 {
            bits.push((1u64 << remainder) - 1);
        }
        Self { bits, n_vocab }
    }

    pub fn set(&mut self, id: u32) {
        let word = id as usize / 64;
        let bit = id as usize % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    pub fn is_set(&self, id: u32) -> bool {
        let word = id as usize / 64;
        let bit = id as usize % 64;
        word < self.bits.len() && (self.bits[word] >> bit) & 1 == 1
    }

    /// Set disallowed tokens to -inf in the logits array.
    pub fn mask_logits(&self, logits: &mut [f32]) {
        for (i, &word) in self.bits.iter().enumerate() {
            if word == u64::MAX {
                continue; // all 64 tokens valid, skip
            }
            let base = i * 64;
            // Process each bit in this word
            for bit in 0..64 {
                let idx = base + bit;
                if idx >= logits.len() {
                    return;
                }
                if (word >> bit) & 1 == 0 {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Bitwise AND: intersection of two masks.
    pub fn and(&self, other: &Self) -> Self {
        let len = self.bits.len().min(other.bits.len());
        let mut bits = Vec::with_capacity(len);
        for i in 0..len {
            bits.push(self.bits[i] & other.bits[i]);
        }
        Self {
            bits,
            n_vocab: self.n_vocab.min(other.n_vocab),
        }
    }

    /// Number of allowed tokens.
    pub fn count_set(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// Grammar engine: parses GBNF, tracks state, computes cached token masks.
pub struct GrammarEngine {
    grammar: CompiledGrammar,
    state: GrammarState,
    cache: HashMap<u64, TokenMask>,
}

impl GrammarEngine {
    /// Compile a GBNF grammar string.
    pub fn new(gbnf: &str) -> Result<Self, InferenceError> {
        let grammar = parser::parse_gbnf(gbnf)?;
        let state = GrammarState::initial(&grammar);
        Ok(Self {
            grammar,
            state,
            cache: HashMap::new(),
        })
    }

    /// Get the token mask for the current grammar state.
    /// Caches results by state hash.
    pub fn token_mask(&mut self, token_trie: &ByteTrie, n_vocab: usize) -> &TokenMask {
        let hash = self.state.hash();
        if !self.cache.contains_key(&hash) {
            let m = mask::build_mask(&self.grammar, &self.state, token_trie, n_vocab);
            self.cache.insert(hash, m);
        }
        &self.cache[&hash]
    }

    /// Advance the grammar state by accepting a token's bytes.
    pub fn accept_token(&mut self, token_bytes: &[u8]) -> Result<(), InferenceError> {
        for &byte in token_bytes {
            self.state = self.state.advance_byte(byte, &self.grammar).ok_or_else(|| {
                InferenceError::Tokenize(format!(
                    "grammar rejected byte 0x{byte:02X}"
                ))
            })?;
        }
        Ok(())
    }

    /// Check if the grammar is in an accepting state.
    pub fn is_complete(&self) -> bool {
        self.state.is_accepting(&self.grammar)
    }

    /// Reset to initial state (reuse with same grammar).
    pub fn reset(&mut self) {
        self.state = GrammarState::initial(&self.grammar);
    }

    /// Access the compiled grammar (for testing).
    pub fn grammar(&self) -> &CompiledGrammar {
        &self.grammar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_new_all_zero() {
        let mask = TokenMask::new(100);
        for i in 0..100 {
            assert!(!mask.is_set(i));
        }
        assert_eq!(mask.count_set(), 0);
    }

    #[test]
    fn mask_allow_all() {
        let mask = TokenMask::allow_all(100);
        for i in 0..100 {
            assert!(mask.is_set(i));
        }
        assert!(!mask.is_set(100));
        assert_eq!(mask.count_set(), 100);
    }

    #[test]
    fn mask_set_and_check() {
        let mut mask = TokenMask::new(256);
        mask.set(0);
        mask.set(63);
        mask.set(64);
        mask.set(255);
        assert!(mask.is_set(0));
        assert!(mask.is_set(63));
        assert!(mask.is_set(64));
        assert!(mask.is_set(255));
        assert!(!mask.is_set(1));
        assert!(!mask.is_set(128));
        assert_eq!(mask.count_set(), 4);
    }

    #[test]
    fn mask_logits_application() {
        let mut mask = TokenMask::new(5);
        mask.set(1);
        mask.set(3);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mask.mask_logits(&mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn mask_and_composable() {
        let mut a = TokenMask::new(128);
        let mut b = TokenMask::new(128);
        a.set(1);
        a.set(2);
        a.set(3);
        b.set(2);
        b.set(3);
        b.set(4);
        let c = a.and(&b);
        assert!(!c.is_set(1));
        assert!(c.is_set(2));
        assert!(c.is_set(3));
        assert!(!c.is_set(4));
        assert_eq!(c.count_set(), 2);
    }

    #[test]
    fn mask_boundary_64() {
        // Test exactly at u64 boundary
        let mut mask = TokenMask::new(64);
        for i in 0..64 {
            mask.set(i);
        }
        assert_eq!(mask.count_set(), 64);
        assert_eq!(mask.bits.len(), 1);
        assert_eq!(mask.bits[0], u64::MAX);
    }

    #[test]
    fn mask_allow_all_boundary() {
        // Vocab size not multiple of 64
        let mask = TokenMask::allow_all(65);
        assert!(mask.is_set(64));
        assert!(!mask.is_set(65));
        assert_eq!(mask.count_set(), 65);
    }

    // --- GrammarEngine integration tests ---

    use crate::tokenizer::trie::ByteTrie;

    fn make_test_trie() -> ByteTrie {
        let mut trie = ByteTrie::new();
        trie.insert(b"hello", 0);
        trie.insert(b"hi", 1);
        trie.insert(b"123", 2);
        trie.insert(b"a", 3);
        trie.insert(b"ab", 4);
        trie.insert(b"abc", 5);
        trie.insert(b" ", 6);
        trie.insert(b"{", 7);
        trie.insert(b"}", 8);
        trie.insert(b"1", 9);
        trie
    }

    #[test]
    fn engine_roundtrip() {
        let mut engine = GrammarEngine::new(r#"root ::= "hello""#).unwrap();
        let trie = make_test_trie();

        // Initial mask: only "hello" (token 0) is valid
        let mask = engine.token_mask(&trie, 10);
        assert!(mask.is_set(0));
        assert_eq!(mask.count_set(), 1);

        // Accept "hello" bytes
        engine.accept_token(b"hello").unwrap();
        assert!(engine.is_complete());
    }

    #[test]
    fn engine_advancing_state() {
        let mut engine = GrammarEngine::new(r#"root ::= "{" "}""#).unwrap();
        let trie = make_test_trie();

        // Initial: only "{" is valid
        let mask = engine.token_mask(&trie, 10);
        assert!(mask.is_set(7), "'{{' should be valid initially");
        assert!(!mask.is_set(8), "'}}' should not be valid initially");

        // Accept "{"
        engine.accept_token(b"{").unwrap();
        assert!(!engine.is_complete());

        // Now only "}" is valid
        let mask = engine.token_mask(&trie, 10);
        assert!(mask.is_set(8), "'}}' should be valid after '{{'");
        assert!(!mask.is_set(7), "'{{' should not be valid after '{{'");

        // Accept "}"
        engine.accept_token(b"}").unwrap();
        assert!(engine.is_complete());
    }

    #[test]
    fn engine_mask_caching() {
        let mut engine = GrammarEngine::new(r#"root ::= [a-z]+"#).unwrap();
        let trie = make_test_trie();

        // Build mask twice — second should hit cache
        let h1 = engine.state.hash();
        let _ = engine.token_mask(&trie, 10);
        let h2 = engine.state.hash();
        assert_eq!(h1, h2, "state hash should be stable");

        let mask = engine.token_mask(&trie, 10);
        // Lowercase tokens: hello, hi, a, ab, abc
        assert!(mask.is_set(0)); // "hello"
        assert!(mask.is_set(1)); // "hi"
        assert!(mask.is_set(3)); // "a"
        assert!(mask.is_set(4)); // "ab"
        assert!(mask.is_set(5)); // "abc"
        assert!(!mask.is_set(2)); // "123"
    }

    #[test]
    fn engine_reset() {
        let mut engine = GrammarEngine::new(r#"root ::= "a""#).unwrap();
        let trie = make_test_trie();

        engine.accept_token(b"a").unwrap();
        assert!(engine.is_complete());

        engine.reset();
        assert!(!engine.is_complete());

        let mask = engine.token_mask(&trie, 10);
        assert!(mask.is_set(3)); // "a" valid again after reset
    }

    #[test]
    fn engine_reject_invalid_byte() {
        let mut engine = GrammarEngine::new(r#"root ::= "abc""#).unwrap();
        // Grammar expects 'a', but we try 'x'
        assert!(engine.accept_token(b"x").is_err());
    }

    #[test]
    fn sampler_with_grammar_mask() {
        use crate::sampling::{Sampler, SamplerParams};

        let mut engine = GrammarEngine::new(r#"root ::= "hello" | "hi""#).unwrap();
        let trie = make_test_trie();

        let mask = engine.token_mask(&trie, 10);

        // Logits: token 2 ("123") has highest logit, but grammar rejects it
        let logits = vec![1.0, 1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut sampler = Sampler::new(SamplerParams {
            temperature: 0.0, // greedy
            ..Default::default()
        });

        // Without grammar: picks token 2 (highest logit)
        assert_eq!(sampler.sample(&logits, &[]), 2);

        // With grammar: token 2 masked, picks 0 or 1 (both have logit 1.0)
        let tok = sampler.sample_with_grammar(&logits, &[], mask);
        assert!(tok == 0 || tok == 1, "grammar should force hello or hi, got {tok}");
    }
}
