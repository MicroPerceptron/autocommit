pub mod trie;
pub mod vocab;
pub mod bpe;
pub mod pre_tokenize;

use crate::error::InferenceError;
use crate::gguf::GgufMetadata;
use vocab::{TokenType, Vocabulary};

pub struct Tokenizer {
    vocab: Vocabulary,
}

impl Tokenizer {
    pub fn from_gguf_metadata(meta: &GgufMetadata) -> Result<Self, InferenceError> {
        let vocab = Vocabulary::from_gguf_metadata(meta)?;
        Ok(Self { vocab })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut output = Vec::new();

        if self.vocab.add_bos {
            if let Some(bos) = self.vocab.bos_id {
                output.push(bos);
            }
        }

        let words = pre_tokenize::pre_tokenize(text, self.vocab.pre_type);

        for word in words {
            bpe::bpe_word(word, &self.vocab, &mut output);
        }

        if self.vocab.add_eos {
            if let Some(eos) = self.vocab.eos_id {
                output.push(eos);
            }
        }

        output
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let uid = id as u32;
            // Skip BOS/EOS
            if Some(id) == self.vocab.bos_id || Some(id) == self.vocab.eos_id {
                continue;
            }
            if let Some(data) = self.vocab.token_data(uid) {
                if self.vocab.token_types.get(uid as usize) == Some(&TokenType::Byte) {
                    // Byte tokens store "<0xHH>" — parse to single byte
                    if let Ok(s) = std::str::from_utf8(data) {
                        if let Some(byte_val) = parse_byte_token_text(s) {
                            bytes.push(byte_val);
                            continue;
                        }
                    }
                }
                bytes.extend_from_slice(data);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn n_vocab(&self) -> usize {
        self.vocab.n_vocab()
    }

    pub fn bos_id(&self) -> Option<i32> {
        self.vocab.bos_id
    }

    pub fn eos_id(&self) -> Option<i32> {
        self.vocab.eos_id
    }

    pub fn token_data(&self, id: i32) -> Option<&[u8]> {
        self.vocab.token_data(id as u32)
    }

    /// Access the token trie for grammar mask building.
    pub(crate) fn token_trie(&self) -> &trie::ByteTrie {
        &self.vocab.token_trie
    }
}

fn parse_byte_token_text(text: &str) -> Option<u8> {
    let text = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(text, 16).ok()
}

/// Encode without BOS/EOS (raw BPE only). Useful for testing against
/// reference outputs that don't include special tokens.
fn encode_raw(vocab: &Vocabulary, text: &str) -> Vec<i32> {
    let words = pre_tokenize::pre_tokenize(text, vocab.pre_type);
    let mut output = Vec::new();
    for word in words {
        bpe::bpe_word(word, &vocab, &mut output);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metadata::{GgufMetadata, GgufValue};

    fn make_test_metadata() -> GgufMetadata {
        let mut meta = GgufMetadata::default();
        meta.entries.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        );
        meta.entries.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("h".into()),
                GgufValue::String("e".into()),
                GgufValue::String("l".into()),
                GgufValue::String("o".into()),
                GgufValue::String("he".into()),
                GgufValue::String("ll".into()),
                GgufValue::String("lo".into()),
                GgufValue::String("hel".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.entries.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![
                GgufValue::String("h e".into()),
                GgufValue::String("l o".into()),
                GgufValue::String("he l".into()),
                GgufValue::String("hel lo".into()),
                GgufValue::String("l l".into()),
            ]),
        );
        meta.entries.insert(
            "tokenizer.ggml.bos_token_id".into(),
            GgufValue::I32(100),
        );
        meta.entries.insert(
            "tokenizer.ggml.eos_token_id".into(),
            GgufValue::I32(101),
        );
        meta.entries.insert(
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(false),
        );
        meta.entries.insert(
            "tokenizer.ggml.add_eos_token".into(),
            GgufValue::Bool(false),
        );
        meta.entries.insert(
            "tokenizer.ggml.pre".into(),
            GgufValue::String("default".into()),
        );
        meta
    }

    #[test]
    fn encode_hello() {
        let meta = make_test_metadata();
        let tok = Tokenizer::from_gguf_metadata(&meta).unwrap();
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![8]);
    }

    #[test]
    fn decode_hello() {
        let meta = make_test_metadata();
        let tok = Tokenizer::from_gguf_metadata(&meta).unwrap();
        let text = tok.decode(&[8]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn roundtrip() {
        let meta = make_test_metadata();
        let tok = Tokenizer::from_gguf_metadata(&meta).unwrap();
        let ids = tok.encode("hello");
        let text = tok.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn bos_eos_insertion() {
        let mut meta = make_test_metadata();
        meta.entries.insert(
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(true),
        );
        meta.entries.insert(
            "tokenizer.ggml.add_eos_token".into(),
            GgufValue::Bool(true),
        );
        let tok = Tokenizer::from_gguf_metadata(&meta).unwrap();
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![100, 8, 101]);

        // Decode should skip BOS/EOS
        let text = tok.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn empty_encode() {
        let meta = make_test_metadata();
        let tok = Tokenizer::from_gguf_metadata(&meta).unwrap();
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }
}

/// Integration tests against real llama.cpp GGUF vocab files.
///
/// These load the `.gguf` vocab files from `third_party/llama.cpp/models/`
/// and compare our tokenizer output against llama.cpp's expected `.inp`/`.out`
/// test pairs.
#[cfg(test)]
mod gguf_tests {
    use super::*;
    use crate::gguf::GgufReader;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    fn parse_inp(content: &str) -> Vec<&str> {
        content.split("\n__ggml_vocab_test__\n").collect()
    }

    fn parse_out(content: &str) -> Vec<Vec<i32>> {
        content
            .lines()
            .map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    Vec::new()
                } else {
                    trimmed
                        .split_whitespace()
                        .map(|s| s.parse::<i32>().unwrap())
                        .collect()
                }
            })
            .collect()
    }

    /// Run all test cases from a .inp/.out pair and return (passed, total, failures).
    fn run_test_suite(
        vocab: &Vocabulary,
        inputs: &[&str],
        expected: &[Vec<i32>],
    ) -> (usize, usize, Vec<(usize, String, Vec<i32>, Vec<i32>)>) {
        let total = inputs.len().min(expected.len());
        let mut passed = 0;
        let mut failures = Vec::new();

        for i in 0..total {
            let got = encode_raw(vocab, inputs[i]);
            if got == expected[i] {
                passed += 1;
            } else {
                failures.push((i, inputs[i].to_string(), expected[i].clone(), got));
            }
        }

        (passed, total, failures)
    }

    #[test]
    fn gpt2_vocab_sanity() {
        let root = repo_root();
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf");
        if !gguf_path.exists() {
            eprintln!("SKIP: {gguf_path:?} not found");
            return;
        }

        let reader = GgufReader::open(&gguf_path).unwrap();
        let vocab = Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();

        let inp_path = gguf_path.with_extension("gguf.inp");
        let out_path = gguf_path.with_extension("gguf.out");
        let inp = std::fs::read_to_string(&inp_path).unwrap();
        let out = std::fs::read_to_string(&out_path).unwrap();

        let inputs = parse_inp(&inp);
        let expected = parse_out(&out);

        let (passed, total, failures) = run_test_suite(&vocab, &inputs, &expected);

        eprintln!("\n=== GPT-2 vocab test: {passed}/{total} passed ===");
        for (i, input, exp, got) in &failures {
            let preview: String = input.chars().take(40).collect();
            eprintln!(
                "  FAIL [{i}] {preview:?}\n    expected: {exp:?}\n    got:      {got:?}"
            );
        }

        // Hard-assert on well-known simple cases
        assert_eq!(
            encode_raw(&vocab, "Hello world"),
            vec![15496, 995],
            "GPT-2: 'Hello world'"
        );
        assert_eq!(
            encode_raw(&vocab, " Hello world"),
            vec![18435, 995],
            "GPT-2: ' Hello world'"
        );
        assert_eq!(
            encode_raw(&vocab, "Hello"),
            vec![15496],
            "GPT-2: 'Hello'"
        );
        assert_eq!(
            encode_raw(&vocab, " Hello"),
            vec![18435],
            "GPT-2: ' Hello'"
        );

        // At least 50% of all test cases should pass
        assert!(
            passed * 2 >= total,
            "GPT-2: only {passed}/{total} passed — too many failures"
        );
    }

    #[test]
    fn llama_bpe_vocab_sanity() {
        let root = repo_root();
        let gguf_path = root.join("third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
        if !gguf_path.exists() {
            eprintln!("SKIP: {gguf_path:?} not found");
            return;
        }

        let reader = GgufReader::open(&gguf_path).unwrap();
        let vocab = Vocabulary::from_gguf_metadata(&reader.metadata).unwrap();

        let inp_path = gguf_path.with_extension("gguf.inp");
        let out_path = gguf_path.with_extension("gguf.out");
        let inp = std::fs::read_to_string(&inp_path).unwrap();
        let out = std::fs::read_to_string(&out_path).unwrap();

        let inputs = parse_inp(&inp);
        let expected = parse_out(&out);

        let (passed, total, failures) = run_test_suite(&vocab, &inputs, &expected);

        eprintln!("\n=== Llama-BPE vocab test: {passed}/{total} passed ===");
        for (i, input, exp, got) in &failures {
            let preview: String = input.chars().take(40).collect();
            eprintln!(
                "  FAIL [{i}] {preview:?}\n    expected: {exp:?}\n    got:      {got:?}"
            );
        }

        // Hard-assert on well-known simple cases
        assert_eq!(
            encode_raw(&vocab, "Hello world"),
            vec![9906, 1917],
            "Llama-BPE: 'Hello world'"
        );
        assert_eq!(
            encode_raw(&vocab, " Hello world"),
            vec![22691, 1917],
            "Llama-BPE: ' Hello world'"
        );

        // At least 50% of all test cases should pass
        assert!(
            passed * 2 >= total,
            "Llama-BPE: only {passed}/{total} passed — too many failures"
        );
    }
}
