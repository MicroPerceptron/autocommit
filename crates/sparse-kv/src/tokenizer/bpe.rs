use heapless::binary_heap::{BinaryHeap, Min};

use super::vocab::Vocabulary;

const NONE: u32 = u32::MAX;

/// Stack-allocated priority queue capacity.
/// Handles words up to ~128 UTF-8 characters (HEAP_CAP / 2 initial pairs,
/// plus stale entries that accumulate during merging).
const HEAP_CAP: usize = 256;

#[derive(Debug)]
struct Symbol {
    start: u32,
    len: u32,
    prev: u32,
    next: u32,
}

#[derive(Debug, Clone, Copy)]
struct MergeCandidate {
    rank: u32,
    left: u32,
    left_len: u32,
    right_len: u32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.left == other.left
    }
}
impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank
            .cmp(&other.rank)
            .then(self.left.cmp(&other.left))
    }
}

/// Tokenize a single pre-tokenized word into token IDs.
pub fn bpe_word(word: &str, vocab: &Vocabulary, output: &mut Vec<i32>) {
    let bytes = word.as_bytes();
    if bytes.is_empty() {
        return;
    }

    let mut symbols = init_symbols(bytes);

    if symbols.len() <= 1 {
        emit_symbol(bytes, &symbols[0], vocab, output);
        return;
    }

    if symbols.len() <= HEAP_CAP / 2 {
        bpe_merge_heapless(bytes, &mut symbols, vocab, output);
    } else {
        bpe_merge_std(bytes, &mut symbols, vocab, output);
    }
}

fn init_symbols(bytes: &[u8]) -> Vec<Symbol> {
    let n = bytes.len();
    let mut symbols = Vec::with_capacity(n);
    for i in 0..n {
        symbols.push(Symbol {
            start: i as u32,
            len: 1,
            prev: if i > 0 { (i - 1) as u32 } else { NONE },
            next: if i + 1 < n { (i + 1) as u32 } else { NONE },
        });
    }
    symbols
}

fn bpe_merge_heapless(
    bytes: &[u8],
    symbols: &mut Vec<Symbol>,
    vocab: &Vocabulary,
    output: &mut Vec<i32>,
) {
    let mut heap: BinaryHeap<MergeCandidate, Min, HEAP_CAP> = BinaryHeap::new();

    // Build initial bigrams
    for i in 0..symbols.len() - 1 {
        if let Some(mc) = make_candidate(bytes, symbols, i as u32, vocab) {
            let _ = heap.push(mc);
        }
    }

    merge_loop(bytes, symbols, &mut heap, vocab);
    emit_all(bytes, symbols, vocab, output);
}

fn bpe_merge_std(
    bytes: &[u8],
    symbols: &mut Vec<Symbol>,
    vocab: &Vocabulary,
    output: &mut Vec<i32>,
) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap as StdHeap;

    let mut heap: StdHeap<Reverse<MergeCandidate>> = StdHeap::new();

    for i in 0..symbols.len() - 1 {
        if let Some(mc) = make_candidate(bytes, symbols, i as u32, vocab) {
            heap.push(Reverse(mc));
        }
    }

    // Same merge loop but using std heap
    while let Some(Reverse(candidate)) = heap.pop() {
        let li = candidate.left as usize;
        if symbols[li].len != candidate.left_len || symbols[li].next == NONE {
            continue;
        }
        let ri = symbols[li].next as usize;
        if symbols[ri].len != candidate.right_len {
            continue;
        }

        do_merge(symbols, li, ri);

        if symbols[li].prev != NONE {
            let pi = symbols[li].prev as usize;
            if let Some(mc) = make_candidate(bytes, symbols, pi as u32, vocab) {
                heap.push(Reverse(mc));
            }
        }
        if symbols[li].next != NONE {
            if let Some(mc) = make_candidate(bytes, symbols, li as u32, vocab) {
                heap.push(Reverse(mc));
            }
        }
    }

    emit_all(bytes, symbols, vocab, output);
}

/// Pop candidates from the heapless heap and perform merges.
fn merge_loop(
    bytes: &[u8],
    symbols: &mut Vec<Symbol>,
    heap: &mut BinaryHeap<MergeCandidate, Min, HEAP_CAP>,
    vocab: &Vocabulary,
) {
    while let Some(candidate) = heap.pop() {
        let li = candidate.left as usize;

        // Staleness check
        if symbols[li].len != candidate.left_len || symbols[li].next == NONE {
            continue;
        }
        let ri = symbols[li].next as usize;
        if symbols[ri].len != candidate.right_len {
            continue;
        }

        do_merge(symbols, li, ri);

        // Push new bigrams for newly-adjacent pairs
        if symbols[li].prev != NONE {
            let pi = symbols[li].prev as usize;
            if let Some(mc) = make_candidate(bytes, symbols, pi as u32, vocab) {
                let _ = heap.push(mc);
            }
        }
        if symbols[li].next != NONE {
            if let Some(mc) = make_candidate(bytes, symbols, li as u32, vocab) {
                let _ = heap.push(mc);
            }
        }
    }
}

fn do_merge(symbols: &mut Vec<Symbol>, li: usize, ri: usize) {
    symbols[li].len += symbols[ri].len;
    let right_next = symbols[ri].next;
    symbols[li].next = right_next;
    if right_next != NONE {
        symbols[right_next as usize].prev = li as u32;
    }
    symbols[ri].len = 0;
}

fn make_candidate(
    bytes: &[u8],
    symbols: &[Symbol],
    left_idx: u32,
    vocab: &Vocabulary,
) -> Option<MergeCandidate> {
    let left = &symbols[left_idx as usize];
    let right_idx = left.next;
    if right_idx == NONE {
        return None;
    }
    let right = &symbols[right_idx as usize];
    let left_text = &bytes[left.start as usize..(left.start + left.len) as usize];
    let right_text = &bytes[right.start as usize..(right.start + right.len) as usize];

    let rank = vocab.merge_rank(left_text, right_text)?;
    Some(MergeCandidate {
        rank,
        left: left_idx,
        left_len: left.len,
        right_len: right.len,
    })
}

fn emit_all(
    bytes: &[u8],
    symbols: &[Symbol],
    vocab: &Vocabulary,
    output: &mut Vec<i32>,
) {
    // Find head of linked list
    let mut head = None;
    for (idx, sym) in symbols.iter().enumerate() {
        if sym.len > 0 && sym.prev == NONE {
            head = Some(idx as u32);
            break;
        }
    }

    let mut cur = match head {
        Some(h) => h,
        None => return,
    };

    loop {
        let sym = &symbols[cur as usize];
        if sym.len > 0 {
            emit_symbol(bytes, sym, vocab, output);
        }
        if sym.next == NONE {
            break;
        }
        cur = sym.next;
    }
}

fn emit_symbol(
    bytes: &[u8],
    sym: &Symbol,
    vocab: &Vocabulary,
    output: &mut Vec<i32>,
) {
    let text = &bytes[sym.start as usize..(sym.start + sym.len) as usize];

    if let Some(id) = vocab.token_id(text) {
        output.push(id as i32);
        return;
    }

    // Byte fallback
    if let Some(ref byte_table) = vocab.byte_tokens {
        for &b in text {
            output.push(byte_table[b as usize] as i32);
        }
    } else {
        output.push(0); // unknown token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metadata::{GgufMetadata, GgufValue};

    fn make_vocab() -> Vocabulary {
        let mut meta = GgufMetadata::default();
        meta.entries.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        );
        meta.entries.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("h".into()),     // 0
                GgufValue::String("e".into()),     // 1
                GgufValue::String("l".into()),     // 2
                GgufValue::String("o".into()),     // 3
                GgufValue::String("he".into()),    // 4
                GgufValue::String("ll".into()),    // 5
                GgufValue::String("lo".into()),    // 6
                GgufValue::String("hel".into()),   // 7
                GgufValue::String("hello".into()), // 8
            ]),
        );
        meta.entries.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![
                GgufValue::String("h e".into()),    // rank 0
                GgufValue::String("l o".into()),    // rank 1
                GgufValue::String("he l".into()),   // rank 2
                GgufValue::String("hel lo".into()), // rank 3
                GgufValue::String("l l".into()),    // rank 4
            ]),
        );
        meta.entries.insert(
            "tokenizer.ggml.pre".into(),
            GgufValue::String("default".into()),
        );
        Vocabulary::from_gguf_metadata(&meta).unwrap()
    }

    #[test]
    fn merge_hello() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        // "hello" → h,e,l,l,o → he,l,l,o → he,ll,o → he,lo → hel,lo → hello
        bpe_word("hello", &vocab, &mut output);
        assert_eq!(output, vec![8]); // token 8 = "hello"
    }

    #[test]
    fn single_char() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        bpe_word("h", &vocab, &mut output);
        assert_eq!(output, vec![0]); // token 0 = "h"
    }

    #[test]
    fn partial_merge() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        // "he" → h,e → he (merge rank 0)
        bpe_word("he", &vocab, &mut output);
        assert_eq!(output, vec![4]); // token 4 = "he"
    }

    #[test]
    fn no_merge_available() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        // "oh" has no merge rule: o,h stay separate
        bpe_word("oh", &vocab, &mut output);
        assert_eq!(output, vec![3, 0]); // o=3, h=0
    }

    #[test]
    fn empty_word() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        bpe_word("", &vocab, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn merge_order_matters() {
        let vocab = make_vocab();
        let mut output = Vec::new();
        // "ll" → l,l → ll (merge rank 1)
        bpe_word("ll", &vocab, &mut output);
        assert_eq!(output, vec![5]); // token 5 = "ll"
    }
}
