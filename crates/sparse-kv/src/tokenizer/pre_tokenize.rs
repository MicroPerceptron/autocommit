/// Pre-tokenization: split input text into word spans before BPE merging.
///
/// Hand-written byte scanners replace regex for zero-dependency operation.
/// ASCII-centric for v1: multi-byte UTF-8 sequences are heuristically
/// treated as letters (correct for CJK, Cyrillic, Arabic; slightly wrong
/// for Unicode punctuation).

/// Pre-tokenizer pattern type, read from `tokenizer.ggml.pre`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerType {
    Gpt2,
    Llama3,
    Default,
}

impl PreTokenizerType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "gpt2" => Self::Gpt2,
            "llama-bpe" | "llama3" => Self::Llama3,
            _ => Self::Default,
        }
    }
}

pub fn pre_tokenize<'a>(text: &'a str, pattern: PreTokenizerType) -> Vec<&'a str> {
    match pattern {
        PreTokenizerType::Gpt2 => split_gpt2(text),
        PreTokenizerType::Llama3 => split_llama3(text),
        PreTokenizerType::Default => split_default(text),
    }
}

/// GPT-2 pre-tokenization.
///
/// Approximates the regex:
///   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn split_gpt2(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::new();
    let mut i = 0;

    while i < len {
        let start = i;

        // Contraction suffixes: 's, 't, 're, 've, 'm, 'll, 'd
        if let Some(end) = try_contraction(bytes, i) {
            spans.push(&text[start..end]);
            i = end;
            continue;
        }

        let b = bytes[i];

        // Optional leading space for next group
        let has_space = b == b' ';
        let content_start = if has_space && i + 1 < len {
            i + 1
        } else {
            i
        };

        if content_start < len && (has_space || !b.is_ascii_whitespace()) {
            let cb = bytes[content_start];

            // Letters (ASCII or multi-byte UTF-8)
            if is_letter_byte(cb) {
                let end = scan_letters(bytes, content_start);
                if end > content_start {
                    spans.push(&text[start..end]);
                    i = end;
                    continue;
                }
            }

            // Digits
            if cb.is_ascii_digit() {
                let mut j = content_start + 1;
                while j < len && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                spans.push(&text[start..j]);
                i = j;
                continue;
            }

            // Punctuation/symbols: not whitespace, not letter, not digit
            if !cb.is_ascii_whitespace() && !is_letter_byte(cb) && !cb.is_ascii_digit() {
                let mut j = content_start + 1;
                while j < len {
                    let jb = bytes[j];
                    if jb.is_ascii_whitespace() || is_letter_byte(jb) || jb.is_ascii_digit() {
                        break;
                    }
                    j += 1;
                }
                spans.push(&text[start..j]);
                i = j;
                continue;
            }
        }

        // Whitespace run
        if b.is_ascii_whitespace() {
            let mut j = i + 1;
            while j < len && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            spans.push(&text[start..j]);
            i = j;
            continue;
        }

        // Fallback: single UTF-8 character
        let char_len = utf8_char_len(bytes[i]);
        let end = (i + char_len).min(len);
        spans.push(&text[start..end]);
        i = end;
    }

    spans
}

/// Llama-3 pre-tokenization.
///
/// Approximates the regex:
///   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
///   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
///
/// Key differences from GPT-2:
/// - Contractions are case-insensitive
/// - Single non-letter/non-digit/non-newline prefix attached to letter groups
/// - Digit groups capped at 3
/// - Whitespace handling: \s*[\r\n] takes priority, then trailing \s+(?!\S), then \s+
fn split_llama3(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::new();
    let mut i = 0;

    while i < len {
        let start = i;

        // 1. Contractions: 's|'t|'re|'ve|'m|'ll|'d (case-insensitive)
        if let Some(end) = try_contraction(bytes, i) {
            spans.push(&text[start..end]);
            i = end;
            continue;
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        //    Optional non-letter/non-digit/non-newline prefix + letters
        {
            let mut prefix_end = i;
            let b = bytes[i];
            if !is_letter_byte(b) && !b.is_ascii_digit() && b != b'\n' && b != b'\r' {
                // This char could be a prefix (space, tab, punctuation, etc.)
                prefix_end = i + utf8_char_len(b).min(len - i);
            }

            let letter_start = prefix_end;
            if letter_start < len && is_letter_byte(bytes[letter_start]) {
                let end = scan_letters(bytes, letter_start);
                if end > letter_start {
                    spans.push(&text[start..end]);
                    i = end;
                    continue;
                }
            }
        }

        // 3. Digits: \p{N}{1,3}
        if bytes[i].is_ascii_digit() {
            let mut j = i + 1;
            let limit = (i + 3).min(len);
            while j < limit && bytes[j].is_ascii_digit() {
                j += 1;
            }
            spans.push(&text[start..j]);
            i = j;
            continue;
        }

        // 4. Optional space + symbols + optional newlines:
        //    ?[^\s\p{L}\p{N}]+[\r\n]*
        if is_symbol_byte(bytes[i]) || (bytes[i] == b' ' && i + 1 < len && is_symbol_byte(bytes[i + 1])) {
            let mut j = if bytes[i] == b' ' { i + 1 } else { i };
            while j < len && is_symbol_byte(bytes[j]) {
                j += utf8_char_len(bytes[j]).min(len - j);
            }
            // Consume optional trailing newlines
            while j < len && (bytes[j] == b'\n' || bytes[j] == b'\r') {
                j += 1;
            }
            spans.push(&text[start..j]);
            i = j;
            continue;
        }

        // 5-7. Whitespace: \s*[\r\n] | \s+(?!\S) | \s+
        // Consume whitespace but leave the last space/tab if a letter or symbol
        // follows (so it can serve as a prefix on the next iteration).
        if bytes[i].is_ascii_whitespace() {
            let mut j = i + 1;
            while j < len && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            // If the next char is a letter or symbol and the last consumed char
            // could be a valid prefix (not \r or \n), yield it back.
            if j < len && j > i + 1 {
                let last = bytes[j - 1];
                if last != b'\n' && last != b'\r' {
                    let next = bytes[j];
                    if is_letter_byte(next) || is_symbol_byte(next) {
                        j -= 1;
                    }
                }
            }
            if j > i {
                spans.push(&text[start..j]);
                i = j;
                continue;
            }
        }

        // Fallback: single UTF-8 character
        let char_len = utf8_char_len(bytes[i]);
        let end = (i + char_len).min(len);
        spans.push(&text[start..end]);
        i = end;
    }

    spans
}

fn is_symbol_byte(b: u8) -> bool {
    // Not whitespace, not a letter start, not a digit, and not a UTF-8
    // continuation byte (0x80-0xBF)
    !b.is_ascii_whitespace()
        && !is_letter_byte(b)
        && !b.is_ascii_digit()
        && !(b >= 0x80 && b <= 0xBF) // continuation bytes are part of multi-byte sequences
}

/// Default: split on whitespace boundaries, attaching leading whitespace to the word.
fn split_default(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::new();
    let mut i = 0;

    while i < len {
        let start = i;
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i > start {
            spans.push(&text[start..i]);
        }
    }

    spans
}

// --- Helpers ---

fn try_contraction(bytes: &[u8], i: usize) -> Option<usize> {
    if i >= bytes.len() {
        return None;
    }
    let b = bytes[i];
    // ASCII apostrophe or UTF-8 right single quotation mark (U+2019 = E2 80 99)
    let (quote_len, after) = if b == b'\'' {
        (1, i + 1)
    } else if b == 0xE2
        && i + 2 < bytes.len()
        && bytes[i + 1] == 0x80
        && bytes[i + 2] == 0x99
    {
        (3, i + 3)
    } else {
        return None;
    };

    if after >= bytes.len() {
        return None;
    }

    // Longest match first: 'll, 're, 've then 's, 't, 'm, 'd
    let suffixes: &[&[u8]] = &[b"ll", b"re", b"ve", b"s", b"t", b"m", b"d"];
    for suffix in suffixes {
        let end = after + suffix.len();
        if end <= bytes.len()
            && bytes[after..end].eq_ignore_ascii_case(suffix)
            && (end >= bytes.len() || !bytes[end].is_ascii_alphabetic())
        {
            let _ = quote_len; // used implicitly via `after`
            return Some(end);
        }
    }
    None
}

/// Check if a byte is (likely) the start of a Unicode letter.
///
/// Heuristic by leading UTF-8 byte:
/// - 0xC3-0xCF: Latin Extended, Greek, Cyrillic
/// - 0xD0-0xDF: Cyrillic, Armenian, Hebrew, Arabic
/// - 0xE0-0xE1: Indic scripts, Myanmar, etc.
/// - 0xE3-0xE9: CJK, Hiragana, Katakana, Hangul
/// - 0xEA-0xED: Hangul, Yi, more CJK
///
/// Excludes:
/// - 0xC2: Latin-1 Supplement symbols (½, ©, etc.)
/// - 0xE2: General Punctuation, Math Symbols, Arrows
/// - 0xEE-0xEF: Private Use, CJK Compatibility
/// - 0xF0+: Supplementary planes (emojis, etc.)
fn is_letter_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || (b >= 0xC3 && b <= 0xED && b != 0xE2)
}

fn scan_letters(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphabetic() {
            i += 1;
        } else if is_letter_byte(b) {
            // Multi-byte UTF-8: consume whole character
            i += utf8_char_len(b).min(bytes.len() - i);
        } else {
            break;
        }
    }
    i
}

fn utf8_char_len(first_byte: u8) -> usize {
    match first_byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GPT-2

    #[test]
    fn gpt2_contractions() {
        let spans = split_gpt2("I've been");
        assert_eq!(spans, vec!["I", "'ve", " been"]);
    }

    #[test]
    fn gpt2_space_letters() {
        let spans = split_gpt2(" Hello world");
        assert_eq!(spans, vec![" Hello", " world"]);
    }

    #[test]
    fn gpt2_digits() {
        let spans = split_gpt2("abc123def");
        assert_eq!(spans, vec!["abc", "123", "def"]);
    }

    #[test]
    fn gpt2_punctuation() {
        let spans = split_gpt2("hello, world!");
        assert_eq!(spans, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn gpt2_whitespace() {
        let spans = split_gpt2("a  b");
        assert_eq!(spans, vec!["a", "  ", "b"]);
    }

    // Llama-3

    #[test]
    fn llama3_digit_groups() {
        let spans = split_llama3("12345");
        assert_eq!(spans, vec!["123", "45"]);
    }

    #[test]
    fn llama3_letters() {
        let spans = split_llama3("Hello world");
        assert_eq!(spans, vec!["Hello", " world"]);
    }

    #[test]
    fn llama3_newline_split() {
        let spans = split_llama3("a\n\nb");
        assert_eq!(spans, vec!["a", "\n\n", "b"]);
    }

    // Default

    #[test]
    fn default_split() {
        let spans = split_default("hello world  test");
        assert_eq!(spans, vec!["hello", " world", "  test"]);
    }

    // Edge cases

    #[test]
    fn empty_string() {
        assert_eq!(split_gpt2(""), Vec::<&str>::new());
        assert_eq!(split_llama3(""), Vec::<&str>::new());
        assert_eq!(split_default(""), Vec::<&str>::new());
    }

    #[test]
    fn pure_whitespace() {
        let spans = split_gpt2("   ");
        assert_eq!(spans, vec!["   "]);
    }
}
