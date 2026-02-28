use crate::error::InferenceError;
use crate::gguf::GgufMetadata;
use super::pre_tokenize::PreTokenizerType;
use super::trie::ByteTrie;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenType {
    Normal = 0,
    Unknown = 1,
    Control = 2,
    UserDef = 3,
    Unused = 4,
    Byte = 5,
}

impl TokenType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Unknown,
            2 => Self::Control,
            3 => Self::UserDef,
            4 => Self::Unused,
            5 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

pub struct Vocabulary {
    /// Decoded token bytes (raw bytes, GPT-2 byte encoding reversed).
    /// Indexed by token_id. Used for trie lookups and decode output.
    pub(crate) token_data: Vec<Vec<u8>>,
    pub(crate) token_types: Vec<TokenType>,
    pub(crate) token_trie: ByteTrie,
    pub(crate) merge_trie: ByteTrie,
    pub(crate) byte_tokens: Option<[u32; 256]>,
    pub(crate) bos_id: Option<i32>,
    pub(crate) eos_id: Option<i32>,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    pub(crate) pre_type: PreTokenizerType,
}

impl Vocabulary {
    pub fn from_gguf_metadata(meta: &GgufMetadata) -> Result<Self, InferenceError> {
        let model_type = meta.get_str("tokenizer.ggml.model").ok_or_else(|| {
            InferenceError::Load("missing tokenizer.ggml.model".into())
        })?;
        if model_type != "gpt2" {
            return Err(InferenceError::Load(format!(
                "unsupported tokenizer model: {model_type} (expected gpt2/BPE)"
            )));
        }

        // Token texts — read from GGUF and decode GPT-2 byte encoding
        let token_strs = meta
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| {
                InferenceError::Load("missing tokenizer.ggml.tokens".into())
            })?;
        let n_vocab = token_strs.len();

        // Token types (needed before decoding to identify Byte tokens)
        let token_types: Vec<TokenType> = match meta.get_i32_array("tokenizer.ggml.token_type") {
            Some(arr) => arr.iter().map(|&v| TokenType::from_i32(v)).collect(),
            None => vec![TokenType::Normal; n_vocab],
        };

        // Decode token texts: BPE GGUF stores text using GPT-2 byte-to-unicode
        // encoding. Byte tokens (<0xHH>) and Control tokens are kept as-is.
        let mut token_data: Vec<Vec<u8>> = Vec::with_capacity(n_vocab);
        for (i, &text) in token_strs.iter().enumerate() {
            let ttype = token_types.get(i).copied().unwrap_or(TokenType::Normal);
            if ttype == TokenType::Byte || ttype == TokenType::Control {
                token_data.push(text.as_bytes().to_vec());
            } else {
                token_data.push(gpt2_decode(text));
            }
        }

        // Build token-to-id trie from decoded bytes
        let mut token_trie = ByteTrie::new();
        for (id, data) in token_data.iter().enumerate() {
            token_trie.insert(data, id as u32);
        }

        // Build byte-fallback table from <0xHH> tokens
        let mut byte_tokens = None;
        {
            let mut table = [0u32; 256];
            let mut found_any = false;
            for (id, (&ttype, raw_text)) in
                token_types.iter().zip(token_strs.iter()).enumerate()
            {
                if ttype == TokenType::Byte {
                    if let Some(byte_val) = parse_byte_token(raw_text) {
                        table[byte_val as usize] = id as u32;
                        found_any = true;
                    }
                }
            }
            if found_any {
                byte_tokens = Some(table);
            }
        }

        // Merge rules → merge trie (decode both sides from GPT-2 encoding)
        let merge_strs = meta
            .get_string_array("tokenizer.ggml.merges")
            .ok_or_else(|| {
                InferenceError::Load("missing tokenizer.ggml.merges".into())
            })?;
        let mut merge_trie = ByteTrie::new();
        for (rank, merge_str) in merge_strs.iter().enumerate() {
            let space_pos = merge_str.find(' ').ok_or_else(|| {
                InferenceError::Load(format!(
                    "invalid merge rule at index {rank}: no space separator"
                ))
            })?;
            let left_decoded = gpt2_decode(&merge_str[..space_pos]);
            let right_decoded = gpt2_decode(&merge_str[space_pos + 1..]);

            // Key: left_bytes + \0 + right_bytes
            let mut key = Vec::with_capacity(left_decoded.len() + 1 + right_decoded.len());
            key.extend_from_slice(&left_decoded);
            key.push(0);
            key.extend_from_slice(&right_decoded);
            merge_trie.insert(&key, rank as u32);
        }

        let bos_id = meta.get_i32("tokenizer.ggml.bos_token_id");
        let eos_id = meta.get_i32("tokenizer.ggml.eos_token_id");
        let add_bos = meta.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(true);
        let add_eos = meta.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);
        let pre_type = meta
            .get_str("tokenizer.ggml.pre")
            .map(PreTokenizerType::from_str)
            .unwrap_or(PreTokenizerType::Default);

        Ok(Self {
            token_data,
            token_types,
            token_trie,
            merge_trie,
            byte_tokens,
            bos_id,
            eos_id,
            add_bos,
            add_eos,
            pre_type,
        })
    }

    pub fn token_data(&self, id: u32) -> Option<&[u8]> {
        self.token_data.get(id as usize).map(|v| v.as_slice())
    }

    pub fn token_id(&self, text: &[u8]) -> Option<u32> {
        self.token_trie.get(text)
    }

    pub fn merge_rank(&self, left: &[u8], right: &[u8]) -> Option<u32> {
        self.merge_trie.get_merge_rank(left, right)
    }

    pub fn n_vocab(&self) -> usize {
        self.token_data.len()
    }
}

/// Decode GPT-2 byte-to-unicode encoding back to raw bytes.
///
/// GPT-2 BPE maps every byte (0x00-0xFF) to a visible Unicode character:
/// - Printable ASCII bytes (0x21-0x7E) and Latin-1 bytes (0xA1-0xAC, 0xAE-0xFF)
///   map to themselves (same codepoint as byte value).
/// - All other bytes (control chars, space, DEL, 0x80-0xA0, 0xAD) map to
///   sequential Unicode starting at U+0100.
///
/// This reverses that mapping: each Unicode char → original byte value.
fn gpt2_decode(encoded: &str) -> Vec<u8> {
    let mut result = Vec::with_capacity(encoded.len());
    for c in encoded.chars() {
        let cp = c as u32;
        // Printable ASCII: 0x21-0x7E → same byte
        if (0x21..=0x7E).contains(&cp) {
            result.push(cp as u8);
        }
        // Latin-1 Supplement printable: 0xA1-0xAC, 0xAE-0xFF → same byte
        else if (0xA1..=0xAC).contains(&cp) || (0xAE..=0xFF).contains(&cp) {
            result.push(cp as u8);
        }
        // Remapped non-printable bytes: U+0100..U+0143 → sequential bytes
        else if (0x100..=0x143).contains(&cp) {
            // The 68 non-printable bytes in ascending order:
            // 0x00-0x20 (33), 0x7F (1), 0x80-0xA0 (33), 0xAD (1) = 68 total
            static NON_PRINTABLE: [u8; 68] = [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                0x20, // 0x00-0x20 (33 bytes)
                0x7F, // DEL
                0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
                0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
                0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
                0xA0, // 0x80-0xA0 (33 bytes)
                0xAD, // soft hyphen
            ];
            let offset = (cp - 0x100) as usize;
            result.push(NON_PRINTABLE[offset]);
        }
        // Fallback: emit the character's UTF-8 bytes directly
        else {
            let mut buf = [0u8; 4];
            result.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    result
}

fn parse_byte_token(text: &str) -> Option<u8> {
    let text = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(text, 16).ok()
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
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(false),
        );
        meta.entries.insert(
            "tokenizer.ggml.pre".into(),
            GgufValue::String("default".into()),
        );
        meta
    }

    #[test]
    fn load_from_metadata() {
        let meta = make_test_metadata();
        let vocab = Vocabulary::from_gguf_metadata(&meta).unwrap();
        assert_eq!(vocab.n_vocab(), 9);
        assert_eq!(vocab.bos_id, Some(100));
        assert!(!vocab.add_bos);
    }

    #[test]
    fn token_roundtrip() {
        let meta = make_test_metadata();
        let vocab = Vocabulary::from_gguf_metadata(&meta).unwrap();
        assert_eq!(vocab.token_id(b"hello"), Some(8));
        assert_eq!(vocab.token_data(8), Some(b"hello".as_ref()));
        assert_eq!(vocab.token_id(b"he"), Some(4));
        assert_eq!(vocab.token_id(b"xyz"), None);
    }

    #[test]
    fn merge_rank_lookup() {
        let meta = make_test_metadata();
        let vocab = Vocabulary::from_gguf_metadata(&meta).unwrap();
        assert_eq!(vocab.merge_rank(b"h", b"e"), Some(0));
        assert_eq!(vocab.merge_rank(b"l", b"o"), Some(1));
        assert_eq!(vocab.merge_rank(b"hel", b"lo"), Some(3));
        assert_eq!(vocab.merge_rank(b"l", b"l"), Some(4));
        assert_eq!(vocab.merge_rank(b"h", b"o"), None);
    }

    #[test]
    fn byte_fallback() {
        let mut meta = make_test_metadata();
        // Add some byte tokens
        let mut tokens = match meta.entries.remove("tokenizer.ggml.tokens").unwrap() {
            GgufValue::Array(v) => v,
            _ => unreachable!(),
        };
        tokens.push(GgufValue::String("<0x41>".into())); // 'A' = 0x41, id=9
        meta.entries.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(tokens),
        );

        // Add token types: all normal except last which is Byte
        let mut types = vec![GgufValue::I32(0); 9];
        types.push(GgufValue::I32(5)); // Byte type
        meta.entries.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(types),
        );

        let vocab = Vocabulary::from_gguf_metadata(&meta).unwrap();
        assert!(vocab.byte_tokens.is_some());
        assert_eq!(vocab.byte_tokens.unwrap()[0x41], 9);
    }

    #[test]
    fn error_on_wrong_model_type() {
        let mut meta = make_test_metadata();
        meta.entries.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        );
        assert!(Vocabulary::from_gguf_metadata(&meta).is_err());
    }

    #[test]
    fn parse_byte_token_format() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0));
        assert_eq!(parse_byte_token("<0xFF>"), Some(255));
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xZZ>"), None);
    }
}
