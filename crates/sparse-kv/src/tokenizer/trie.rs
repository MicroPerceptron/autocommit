/// A byte-oriented trie for zero-allocation token and merge rank lookups.
///
/// Children are stored as sorted `(byte, child_index)` pairs rather than
/// a `[Option<u32>; 256]` array. For a 128K vocab (~500K nodes), this
/// keeps memory under 20 MB instead of 512 MB while adding only O(log 256)
/// = 8 comparisons per byte lookup via binary search.

pub(crate) struct TrieNode {
    pub(crate) children: Vec<(u8, u32)>,
    pub(crate) value: Option<u32>,
}

pub struct ByteTrie {
    pub(crate) nodes: Vec<TrieNode>,
}

impl ByteTrie {
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                children: Vec::new(),
                value: None,
            }],
        }
    }

    pub fn insert(&mut self, key: &[u8], value: u32) {
        let mut idx = 0u32;
        for &byte in key {
            idx = self.get_or_create_child(idx, byte);
        }
        self.nodes[idx as usize].value = Some(value);
    }

    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let mut idx = 0u32;
        for &byte in key {
            idx = self.get_child(idx, byte)?;
        }
        self.nodes[idx as usize].value
    }

    /// Look up a merge rank for the pair (left, right) using the
    /// null-byte separator convention: walks left bytes, then 0x00, then right bytes.
    pub fn get_merge_rank(&self, left: &[u8], right: &[u8]) -> Option<u32> {
        let mut idx = 0u32;
        for &byte in left {
            idx = self.get_child(idx, byte)?;
        }
        idx = self.get_child(idx, 0)?;
        for &byte in right {
            idx = self.get_child(idx, byte)?;
        }
        self.nodes[idx as usize].value
    }

    fn get_child(&self, node_idx: u32, byte: u8) -> Option<u32> {
        let node = &self.nodes[node_idx as usize];
        node.children
            .binary_search_by_key(&byte, |(b, _)| *b)
            .ok()
            .map(|i| node.children[i].1)
    }

    fn get_or_create_child(&mut self, node_idx: u32, byte: u8) -> u32 {
        let node = &self.nodes[node_idx as usize];
        match node.children.binary_search_by_key(&byte, |(b, _)| *b) {
            Ok(i) => self.nodes[node_idx as usize].children[i].1,
            Err(insert_pos) => {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(TrieNode {
                    children: Vec::new(),
                    value: None,
                });
                self.nodes[node_idx as usize]
                    .children
                    .insert(insert_pos, (byte, new_idx));
                new_idx
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut trie = ByteTrie::new();
        trie.insert(b"hello", 42);
        trie.insert(b"help", 99);
        trie.insert(b"world", 7);

        assert_eq!(trie.get(b"hello"), Some(42));
        assert_eq!(trie.get(b"help"), Some(99));
        assert_eq!(trie.get(b"world"), Some(7));
        assert_eq!(trie.get(b"hell"), None);
        assert_eq!(trie.get(b"helpers"), None);
        assert_eq!(trie.get(b""), None);
    }

    #[test]
    fn overlapping_prefixes() {
        let mut trie = ByteTrie::new();
        trie.insert(b"a", 1);
        trie.insert(b"ab", 2);
        trie.insert(b"abc", 3);

        assert_eq!(trie.get(b"a"), Some(1));
        assert_eq!(trie.get(b"ab"), Some(2));
        assert_eq!(trie.get(b"abc"), Some(3));
        assert_eq!(trie.get(b"abcd"), None);
    }

    #[test]
    fn merge_rank_with_separator() {
        let mut trie = ByteTrie::new();
        // Insert "he\0ll" -> rank 0
        let mut key = Vec::new();
        key.extend_from_slice(b"he");
        key.push(0);
        key.extend_from_slice(b"ll");
        trie.insert(&key, 0);

        // Insert "hel\0lo" -> rank 1
        key.clear();
        key.extend_from_slice(b"hel");
        key.push(0);
        key.extend_from_slice(b"lo");
        trie.insert(&key, 1);

        assert_eq!(trie.get_merge_rank(b"he", b"ll"), Some(0));
        assert_eq!(trie.get_merge_rank(b"hel", b"lo"), Some(1));
        assert_eq!(trie.get_merge_rank(b"h", b"e"), None);
        assert_eq!(trie.get_merge_rank(b"he", b"lo"), None);
    }

    #[test]
    fn missing_keys() {
        let trie = ByteTrie::new();
        assert_eq!(trie.get(b"anything"), None);
        assert_eq!(trie.get_merge_rank(b"a", b"b"), None);
    }

    #[test]
    fn single_byte_keys() {
        let mut trie = ByteTrie::new();
        for b in 0..=255u8 {
            trie.insert(&[b], b as u32);
        }
        for b in 0..=255u8 {
            assert_eq!(trie.get(&[b]), Some(b as u32));
        }
    }

    #[test]
    fn overwrite_value() {
        let mut trie = ByteTrie::new();
        trie.insert(b"key", 1);
        trie.insert(b"key", 2);
        assert_eq!(trie.get(b"key"), Some(2));
    }
}
