/// Grammar state machine: pushdown automaton for tracking parse position.
///
/// The state is a set of "stacks" — each stack is a possible parse path
/// through the grammar. Multiple stacks arise from alternatives. Advancing
/// a byte filters to stacks that can accept it.

use super::parser::{CompiledGrammar, GrammarElement};

/// Maximum stack depth per parse path. Prevents infinite recursion.
const MAX_STACK_DEPTH: usize = 64;

/// Grammar parse state: a set of possible parse positions.
#[derive(Clone, Debug)]
pub struct GrammarState {
    /// Each inner Vec is a stack of element positions.
    /// Top of stack = current position in the grammar.
    /// Below = return addresses for nested rule calls.
    pub(crate) stacks: Vec<Vec<usize>>,
}

impl GrammarState {
    /// Initial state: positioned at the start of the root rule's alternatives.
    pub fn initial(grammar: &CompiledGrammar) -> Self {
        let mut stacks = Vec::new();
        let root = grammar.root_rule;
        for &alt_start in &grammar.rule_alts[root] {
            let mut stack = vec![alt_start];
            // Advance past any leading RuleRefs/Ends to reach character elements
            advance_to_terminal(grammar, &mut stack, &mut stacks, 0);
        }
        Self { stacks }
    }

    /// Advance by one byte. Returns None if no stack can accept this byte.
    pub fn advance_byte(&self, byte: u8, grammar: &CompiledGrammar) -> Option<Self> {
        let cp = byte as u32;
        let mut new_stacks = Vec::new();

        for stack in &self.stacks {
            if stack.is_empty() {
                continue;
            }
            let pos = *stack.last().unwrap();

            if char_matches(grammar, pos, cp) {
                // Find the next position after this character element
                let next_pos = skip_char_element(grammar, pos);
                let mut new_stack = stack[..stack.len() - 1].to_vec();
                new_stack.push(next_pos);
                advance_to_terminal(grammar, &mut new_stack, &mut new_stacks, 0);
            }
        }

        if new_stacks.is_empty() {
            None
        } else {
            // Deduplicate stacks
            new_stacks.sort();
            new_stacks.dedup();
            Some(Self { stacks: new_stacks })
        }
    }

    /// Check if any stack is in an accepting state (completed the root rule).
    pub fn is_accepting(&self, grammar: &CompiledGrammar) -> bool {
        for stack in &self.stacks {
            if stack.is_empty() {
                return true; // empty stack = completed root rule
            }
            // Also check if top of stack is at an End element with nothing below
            if stack.len() == 1 {
                let pos = stack[0];
                if pos < grammar.elements.len()
                    && grammar.elements[pos] == GrammarElement::End
                {
                    return true;
                }
            }
        }
        false
    }

    /// Hash the state for mask caching. Uses FNV-1a for speed.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for stack in &self.stacks {
            for &pos in stack {
                h ^= pos as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0xFF; // separator between stacks
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Check if the element at `pos` matches codepoint `cp`.
///
/// Handles both positive and negated character classes.
/// A char element is followed by zero or more (Char, CharRangeUpper) pairs.
fn char_matches(grammar: &CompiledGrammar, pos: usize, cp: u32) -> bool {
    let elements = &grammar.elements;
    if pos >= elements.len() {
        return false;
    }

    match elements[pos] {
        GrammarElement::Char(_) => {
            // Positive char class: check if cp falls in any range
            char_class_contains(elements, pos, cp)
        }
        GrammarElement::CharNot(_) => {
            // Negated char class: cp must NOT match any range
            !char_class_contains(elements, pos, cp)
        }
        _ => false,
    }
}

/// Check if `cp` is in the character class starting at `pos`.
///
/// A character class is: Char/CharNot + CharRangeUpper, then zero or more
/// CharAlt + CharRangeUpper pairs (same class, different ranges).
fn char_class_contains(elements: &[GrammarElement], pos: usize, cp: u32) -> bool {
    let mut i = pos;

    // First element is Char or CharNot
    let first_lo = match elements[i] {
        GrammarElement::Char(v) | GrammarElement::CharNot(v) => v,
        _ => return false,
    };
    i += 1;

    // Must be followed by CharRangeUpper
    if i >= elements.len() {
        return false;
    }
    if let GrammarElement::CharRangeUpper(hi) = elements[i] {
        if cp >= first_lo && cp <= hi {
            return true;
        }
        i += 1;
    } else {
        return false;
    }

    // Additional ranges: CharAlt + CharRangeUpper pairs (same class)
    while i + 1 < elements.len() {
        if let GrammarElement::CharAlt(lo) = elements[i] {
            if let GrammarElement::CharRangeUpper(hi) = elements[i + 1] {
                if cp >= lo && cp <= hi {
                    return true;
                }
                i += 2;
                continue;
            }
        }
        break;
    }

    false
}

/// Skip past a character element (Char/CharNot + ranges) to find the next element.
///
/// Skips: initial Char/CharNot, its CharRangeUpper, then any CharAlt + CharRangeUpper
/// pairs (additional ranges in the same class). Stops at the next Char/End/RuleRef
/// (which starts a new element).
fn skip_char_element(grammar: &CompiledGrammar, pos: usize) -> usize {
    let elements = &grammar.elements;
    let mut i = pos;

    // Skip initial Char or CharNot
    match elements.get(i) {
        Some(GrammarElement::Char(_)) | Some(GrammarElement::CharNot(_)) => i += 1,
        _ => return i,
    }

    // Skip CharRangeUpper
    if i < elements.len() && matches!(elements[i], GrammarElement::CharRangeUpper(_)) {
        i += 1;
    }

    // Skip additional CharAlt + CharRangeUpper pairs (same char class)
    while i + 1 < elements.len() {
        if matches!(elements[i], GrammarElement::CharAlt(_))
            && matches!(elements[i + 1], GrammarElement::CharRangeUpper(_))
        {
            i += 2;
        } else {
            break;
        }
    }

    i
}

/// Advance a stack past non-terminal elements (RuleRef, End) until we
/// reach a character element or the stack is empty/complete.
///
/// This may produce multiple stacks (from alternatives).
fn advance_to_terminal(
    grammar: &CompiledGrammar,
    stack: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
    depth: usize,
) {
    if depth > MAX_STACK_DEPTH {
        return; // prevent infinite recursion
    }

    loop {
        if stack.is_empty() {
            // Completed the root rule — accepting state
            output.push(stack.clone());
            return;
        }

        let pos = *stack.last().unwrap();
        if pos >= grammar.elements.len() {
            output.push(stack.clone());
            return;
        }

        match grammar.elements[pos] {
            GrammarElement::Char(_) | GrammarElement::CharNot(_) => {
                // Reached a character element — ready for matching
                output.push(stack.clone());
                return;
            }
            GrammarElement::End => {
                // End of this alternative. Pop to return to caller.
                stack.pop();
                if stack.is_empty() {
                    output.push(stack.clone());
                    return;
                }
                // Continue at the return address
            }
            GrammarElement::RuleRef(rule_id) => {
                if stack.len() >= MAX_STACK_DEPTH {
                    return;
                }
                // Push return address (next element after this RuleRef)
                *stack.last_mut().unwrap() = pos + 1;
                // Fork into each alternative of the referenced rule
                let rule_id = rule_id as usize;
                if rule_id >= grammar.rule_alts.len() {
                    return; // invalid rule ref
                }
                for &alt_start in &grammar.rule_alts[rule_id] {
                    let mut fork = stack.clone();
                    fork.push(alt_start);
                    advance_to_terminal(grammar, &mut fork, output, depth + 1);
                }
                return; // all forks handled
            }
            GrammarElement::CharRangeUpper(_) | GrammarElement::CharAlt(_) => {
                // Shouldn't be at this position directly — skip it
                *stack.last_mut().unwrap() = pos + 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parser::parse_gbnf;

    #[test]
    fn simple_literal_accept() {
        let g = parse_gbnf(r#"root ::= "abc""#).unwrap();
        let mut state = GrammarState::initial(&g);
        assert!(!state.is_accepting(&g));

        state = state.advance_byte(b'a', &g).unwrap();
        assert!(!state.is_accepting(&g));

        state = state.advance_byte(b'b', &g).unwrap();
        assert!(!state.is_accepting(&g));

        state = state.advance_byte(b'c', &g).unwrap();
        assert!(state.is_accepting(&g));
    }

    #[test]
    fn simple_literal_reject() {
        let g = parse_gbnf(r#"root ::= "abc""#).unwrap();
        let state = GrammarState::initial(&g);
        assert!(state.advance_byte(b'x', &g).is_none());
    }

    #[test]
    fn alternatives() {
        let g = parse_gbnf(r#"root ::= "a" | "b""#).unwrap();
        let state = GrammarState::initial(&g);

        let sa = state.advance_byte(b'a', &g).unwrap();
        assert!(sa.is_accepting(&g));

        let sb = state.advance_byte(b'b', &g).unwrap();
        assert!(sb.is_accepting(&g));

        assert!(state.advance_byte(b'c', &g).is_none());
    }

    #[test]
    fn char_class() {
        let g = parse_gbnf(r#"root ::= [a-z]"#).unwrap();
        let state = GrammarState::initial(&g);

        let s = state.advance_byte(b'm', &g).unwrap();
        assert!(s.is_accepting(&g));

        assert!(state.advance_byte(b'A', &g).is_none());
        assert!(state.advance_byte(b'0', &g).is_none());
    }

    #[test]
    fn negated_char_class() {
        let g = parse_gbnf(r#"root ::= [^ab]"#).unwrap();
        let state = GrammarState::initial(&g);

        assert!(state.advance_byte(b'a', &g).is_none());
        assert!(state.advance_byte(b'b', &g).is_none());

        let s = state.advance_byte(b'c', &g).unwrap();
        assert!(s.is_accepting(&g));
    }

    #[test]
    fn rule_reference() {
        let g = parse_gbnf("root ::= greeting\ngreeting ::= \"hi\"").unwrap();
        let state = GrammarState::initial(&g);

        let s1 = state.advance_byte(b'h', &g).unwrap();
        let s2 = s1.advance_byte(b'i', &g).unwrap();
        assert!(s2.is_accepting(&g));
    }

    #[test]
    fn sequence() {
        let g = parse_gbnf(r#"root ::= "a" "b""#).unwrap();
        let state = GrammarState::initial(&g);

        let s1 = state.advance_byte(b'a', &g).unwrap();
        assert!(!s1.is_accepting(&g));

        let s2 = s1.advance_byte(b'b', &g).unwrap();
        assert!(s2.is_accepting(&g));
    }

    #[test]
    fn star_quantifier() {
        let g = parse_gbnf(r#"root ::= "a"*"#).unwrap();
        let state = GrammarState::initial(&g);

        // Empty is valid (zero repetitions)
        assert!(state.is_accepting(&g));

        // One 'a' is valid
        let s1 = state.advance_byte(b'a', &g).unwrap();
        assert!(s1.is_accepting(&g));

        // Multiple 'a's are valid
        let s2 = s1.advance_byte(b'a', &g).unwrap();
        assert!(s2.is_accepting(&g));

        // Non-'a' is rejected
        assert!(state.advance_byte(b'b', &g).is_none());
    }

    #[test]
    fn plus_quantifier() {
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        let state = GrammarState::initial(&g);

        // Empty is NOT valid (need at least one)
        assert!(!state.is_accepting(&g));

        let s1 = state.advance_byte(b'5', &g).unwrap();
        assert!(s1.is_accepting(&g));

        let s2 = s1.advance_byte(b'3', &g).unwrap();
        assert!(s2.is_accepting(&g));
    }

    #[test]
    fn optional_quantifier() {
        let g = parse_gbnf(r#"root ::= "x"?"#).unwrap();
        let state = GrammarState::initial(&g);

        // Empty is valid
        assert!(state.is_accepting(&g));

        // One 'x' is valid
        let s1 = state.advance_byte(b'x', &g).unwrap();
        assert!(s1.is_accepting(&g));

        // Two 'x' is invalid (only 0 or 1 allowed)
        assert!(s1.advance_byte(b'x', &g).is_none());
    }

    #[test]
    fn state_hash_deterministic() {
        let g = parse_gbnf(r#"root ::= "abc""#).unwrap();
        let s1 = GrammarState::initial(&g);
        let s2 = GrammarState::initial(&g);
        assert_eq!(s1.hash(), s2.hash());

        let s1a = s1.advance_byte(b'a', &g).unwrap();
        let s2a = s2.advance_byte(b'a', &g).unwrap();
        assert_eq!(s1a.hash(), s2a.hash());
    }

    #[test]
    fn multi_rule_sequence() {
        let g = parse_gbnf(
            "root ::= key \":\" value\nkey ::= \"k\"\nvalue ::= \"v\"",
        )
        .unwrap();
        let state = GrammarState::initial(&g);

        let s = state.advance_byte(b'k', &g).unwrap();
        let s = s.advance_byte(b':', &g).unwrap();
        let s = s.advance_byte(b'v', &g).unwrap();
        assert!(s.is_accepting(&g));
    }

    // --- llama.cpp behavioral consistency tests ---
    //
    // These mirror the validation behavior of llama.cpp's test-gbnf-validator:
    // feed input bytes one at a time, check accept/reject at each position.

    /// Helper: validate a string against a grammar, returning the index of
    /// the first rejected byte (or None if fully accepted).
    fn validate(grammar: &CompiledGrammar, input: &[u8]) -> (bool, usize) {
        let mut state = GrammarState::initial(grammar);
        for (i, &byte) in input.iter().enumerate() {
            match state.advance_byte(byte, grammar) {
                Some(next) => state = next,
                None => return (false, i),
            }
        }
        (state.is_accepting(grammar), input.len())
    }

    #[test]
    fn validate_exact_match() {
        let g = parse_gbnf(r#"root ::= "hello""#).unwrap();
        assert_eq!(validate(&g, b"hello"), (true, 5));
        assert_eq!(validate(&g, b"hell"), (false, 4)); // incomplete
        assert_eq!(validate(&g, b"help"), (false, 3)); // diverges at 'p'
        assert_eq!(validate(&g, b""), (false, 0)); // empty
    }

    #[test]
    fn validate_alternatives() {
        let g = parse_gbnf(r#"root ::= "true" | "false" | "null""#).unwrap();
        assert_eq!(validate(&g, b"true"), (true, 4));
        assert_eq!(validate(&g, b"false"), (true, 5));
        assert_eq!(validate(&g, b"null"), (true, 4));
        assert_eq!(validate(&g, b"tru"), (false, 3)); // incomplete
        assert_eq!(validate(&g, b"x"), (false, 0)); // no alt starts with x
    }

    #[test]
    fn validate_digit_sequence() {
        // llama.cpp: [0-9]+ accepts one or more digits
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        assert_eq!(validate(&g, b"0"), (true, 1));
        assert_eq!(validate(&g, b"42"), (true, 2));
        assert_eq!(validate(&g, b"12345"), (true, 5));
        assert_eq!(validate(&g, b""), (false, 0)); // need at least one
        assert_eq!(validate(&g, b"a"), (false, 0)); // not a digit
        assert_eq!(validate(&g, b"12a"), (false, 2)); // digit then non-digit
    }

    #[test]
    fn validate_bounded_repeat() {
        // {1,3}: exactly 1 to 3 digits
        let g = parse_gbnf(r#"root ::= [0-9]{1,3}"#).unwrap();
        assert_eq!(validate(&g, b"1"), (true, 1));
        assert_eq!(validate(&g, b"12"), (true, 2));
        assert_eq!(validate(&g, b"123"), (true, 3));
        assert_eq!(validate(&g, b""), (false, 0)); // need at least 1
        // After 3 digits, a 4th digit should be rejected
        assert_eq!(validate(&g, b"1234"), (false, 3));
    }

    #[test]
    fn validate_exact_count() {
        // {2}: exactly 2
        let g = parse_gbnf(r#"root ::= "a"{2}"#).unwrap();
        assert_eq!(validate(&g, b"aa"), (true, 2));
        assert_eq!(validate(&g, b"a"), (false, 1)); // too few
        assert_eq!(validate(&g, b"aaa"), (false, 2)); // too many
    }

    #[test]
    fn validate_star_accepts_empty() {
        let g = parse_gbnf(r#"root ::= "x"*"#).unwrap();
        assert_eq!(validate(&g, b""), (true, 0));
        assert_eq!(validate(&g, b"x"), (true, 1));
        assert_eq!(validate(&g, b"xxx"), (true, 3));
        assert_eq!(validate(&g, b"y"), (false, 0));
    }

    #[test]
    fn validate_optional() {
        let g = parse_gbnf(r#"root ::= "a" "b"?"#).unwrap();
        assert_eq!(validate(&g, b"a"), (true, 1)); // 'b' is optional
        assert_eq!(validate(&g, b"ab"), (true, 2));
        assert_eq!(validate(&g, b"abc"), (false, 2)); // only 0 or 1 'b'
        assert_eq!(validate(&g, b""), (false, 0));
    }

    #[test]
    fn validate_negated_class() {
        // Accept any single byte NOT in the set
        let g = parse_gbnf(r#"root ::= [^abc]"#).unwrap();
        assert_eq!(validate(&g, b"a"), (false, 0));
        assert_eq!(validate(&g, b"b"), (false, 0));
        assert_eq!(validate(&g, b"c"), (false, 0));
        assert_eq!(validate(&g, b"d"), (true, 1));
        assert_eq!(validate(&g, b"z"), (true, 1));
        assert_eq!(validate(&g, b"0"), (true, 1));
    }

    #[test]
    fn validate_multi_range_char_class() {
        // [a-zA-Z]: letters only
        let g = parse_gbnf(r#"root ::= [a-zA-Z]+"#).unwrap();
        assert_eq!(validate(&g, b"hello"), (true, 5));
        assert_eq!(validate(&g, b"Hello"), (true, 5));
        assert_eq!(validate(&g, b"ABC"), (true, 3));
        assert_eq!(validate(&g, b"abc123"), (false, 3)); // stops at '1'
        assert_eq!(validate(&g, b"123"), (false, 0));
    }

    #[test]
    fn validate_multi_rule_grammar() {
        // key-value grammar: key = letters, value = digits
        let g = parse_gbnf(
            "root ::= key \":\" value\n\
             key ::= [a-z]+\n\
             value ::= [0-9]+"
        ).unwrap();
        assert_eq!(validate(&g, b"abc:123"), (true, 7));
        assert_eq!(validate(&g, b"x:0"), (true, 3));
        assert_eq!(validate(&g, b"abc:"), (false, 4)); // value missing
        assert_eq!(validate(&g, b":123"), (false, 0)); // key missing
    }

    #[test]
    fn validate_recursive_rule() {
        // Nested structure: expr = digit | "(" expr ")"
        let g = parse_gbnf(
            "root ::= expr\n\
             expr ::= [0-9] | \"(\" expr \")\""
        ).unwrap();
        assert_eq!(validate(&g, b"5"), (true, 1));
        assert_eq!(validate(&g, b"(3)"), (true, 3));
        assert_eq!(validate(&g, b"((7))"), (true, 5));
        assert_eq!(validate(&g, b"("), (false, 1)); // unclosed
        assert_eq!(validate(&g, b"()"), (false, 1)); // no expr inside
    }

    #[test]
    fn validate_sequence_with_star() {
        // llama.cpp pattern: literal + repetition
        let g = parse_gbnf(r#"root ::= "\"" [^"\\]* "\""  "#).unwrap();
        assert_eq!(validate(&g, b"\"\""), (true, 2)); // empty string
        assert_eq!(validate(&g, b"\"abc\""), (true, 5));
        assert_eq!(validate(&g, b"\"hello world\""), (true, 13));
        assert_eq!(validate(&g, b"\""), (false, 1)); // unclosed
    }

    #[test]
    fn validate_list_gbnf() {
        // llama.cpp's list.gbnf (simplified — no unicode exclusions)
        let g = parse_gbnf(
            "root ::= item+\n\
             item ::= \"- \" [a-zA-Z0-9 ]+ \"\\n\""
        ).unwrap();
        assert_eq!(validate(&g, b"- hello\n"), (true, 8));
        assert_eq!(validate(&g, b"- hello\n- world\n"), (true, 16));
        assert_eq!(validate(&g, b"- a\n"), (true, 4));
        assert_eq!(validate(&g, b"hello\n"), (false, 0)); // no "- " prefix
        assert_eq!(validate(&g, b"- \n"), (false, 2)); // empty item content
    }

    #[test]
    fn validate_whitespace_rule() {
        // Common pattern in llama.cpp grammars
        let g = parse_gbnf(
            "root ::= \"a\" ws \"b\"\n\
             ws ::= \" \"*"
        ).unwrap();
        assert_eq!(validate(&g, b"ab"), (true, 2));
        assert_eq!(validate(&g, b"a b"), (true, 3));
        assert_eq!(validate(&g, b"a   b"), (true, 5));
        assert_eq!(validate(&g, b"a"), (false, 1)); // missing 'b'
    }

    #[test]
    fn validate_escape_sequences() {
        // Verify escape handling matches llama.cpp
        let g = parse_gbnf(r#"root ::= "\n" | "\t" | "\\" | "\""  "#).unwrap();
        assert_eq!(validate(&g, b"\n"), (true, 1));
        assert_eq!(validate(&g, b"\t"), (true, 1));
        assert_eq!(validate(&g, b"\\"), (true, 1));
        assert_eq!(validate(&g, b"\""), (true, 1));
        assert_eq!(validate(&g, b"a"), (false, 0));
    }

    #[test]
    fn validate_hex_escape() {
        let g = parse_gbnf(r#"root ::= [^\x00-\x1F]+"#).unwrap();
        // Printable ASCII should be accepted
        assert_eq!(validate(&g, b"hello"), (true, 5));
        assert_eq!(validate(&g, b"ABC 123!"), (true, 8));
        // Control chars should be rejected
        assert_eq!(validate(&g, b"\x00"), (false, 0));
        assert_eq!(validate(&g, b"\x1F"), (false, 0));
        // 0x20 (space) is above 0x1F, accepted
        assert_eq!(validate(&g, b" "), (true, 1));
    }

    #[test]
    fn validate_alternation_with_shared_prefix() {
        // Both alts start with "a" — NFA-style state splitting
        let g = parse_gbnf(r#"root ::= "abc" | "axyz""#).unwrap();
        assert_eq!(validate(&g, b"abc"), (true, 3));
        assert_eq!(validate(&g, b"axyz"), (true, 4));
        assert_eq!(validate(&g, b"ab"), (false, 2)); // incomplete "abc"
        assert_eq!(validate(&g, b"ax"), (false, 2)); // incomplete "axyz"
        assert_eq!(validate(&g, b"a"), (false, 1)); // ambiguous, incomplete
    }

    #[test]
    fn validate_complex_json_value() {
        // Simplified JSON-like string rule (no groups needed)
        let g = parse_gbnf(
            "root ::= string\n\
             string ::= \"\\\"\" char* \"\\\"\"\n\
             char ::= [^\"\\\\\\x7F\\x00-\\x1F]"
        ).unwrap();
        assert_eq!(validate(&g, b"\"\""), (true, 2)); // empty string
        assert_eq!(validate(&g, b"\"hello\""), (true, 7));
        assert_eq!(validate(&g, b"\"abc 123\""), (true, 9));
        // Control char inside string should be rejected
        assert_eq!(validate(&g, b"\"\x01\""), (false, 1));
    }

    // --- Ported from llama.cpp test-grammar-integration.cpp ---
    //
    // These tests validate behavioral consistency with llama.cpp's grammar
    // implementation. Each test uses the same GBNF grammar and accept/reject
    // strings from llama.cpp's test suite.

    /// Helper: check that the grammar accepts all `pass` strings
    /// and rejects all `fail` strings.
    fn check_grammar(gbnf: &str, pass: &[&[u8]], fail: &[&[u8]]) {
        let g = parse_gbnf(gbnf).unwrap();
        for input in pass {
            let (accepted, _) = validate(&g, input);
            assert!(
                accepted,
                "expected ACCEPT for {:?} with grammar:\n{gbnf}",
                String::from_utf8_lossy(input)
            );
        }
        for input in fail {
            let (accepted, _) = validate(&g, input);
            assert!(
                !accepted,
                "expected REJECT for {:?} with grammar:\n{gbnf}",
                String::from_utf8_lossy(input)
            );
        }
    }

    #[test]
    fn llama_simple_grammar() {
        // From test-grammar-integration.cpp: "simple grammar"
        check_grammar(
            r#"
                root ::= expr
                expr ::= term ("+" term)*
                term ::= number
                number ::= [0-9]+
            "#,
            &[b"42", b"1+2+3+4+5", b"123+456"],
            &[b"+", b"1+2+3+4+5+", b"12a45"],
        );
    }

    #[test]
    fn llama_medium_complexity_grammar() {
        // From test-grammar-integration.cpp: "medium complexity grammar"
        check_grammar(
            r#"
                root ::= expression
                expression ::= term ws (("+" | "-") ws term)*
                term ::= factor ws (("*" | "/") ws factor)*
                factor ::= number | variable | "(" expression ")" | function-call
                number ::= [0-9]+
                variable ::= [a-zA-Z_] [a-zA-Z0-9_]*
                function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
                ws ::= [ \t\n\r]?
            "#,
            &[
                b"42",
                b"1*2*3*4*5",
                b"x",
                b"x+10",
                b"x1+y2",
                b"(a+b)*(c-d)",
                b"func()",
                b"func(x,y+2)",
                b"a*(b+c)-d/e",
                b"f(g(x),h(y,z))",
                b"x + 10",
                b"x1 + y2",
                b"(a + b) * (c - d)",
                b"func(x, y + 2)",
                b"a * (b + c) - d / e",
                b"f(g(x), h(y, z))",
                b"123+456",
                b"123*456*789-123/456+789*123",
            ],
            &[
                b"+",
                b"x + + y",
                b"a * / b",
                b"func(,)",
                b"func(x y)",
                b"(a + b",
                b"x + y)",
                b"a + b * (c - d",
                b"42 +",
                b"x +",
                b"x + 10 +",
                b"(a + b) * (c - d",
                b"func(",
                b"func(x, y + 2",
                b"a * (b + c) - d /",
                b"f(g(x), h(y, z)",
            ],
        );
    }

    #[test]
    fn llama_star_quantifier() {
        check_grammar(
            r#"root ::= "a"*"#,
            &[b"", b"a", b"aaaaa", b"aaaaaaaaaaaaaaaaaa"],
            &[b"b", b"ab", b"aab", b"ba"],
        );
    }

    #[test]
    fn llama_plus_quantifier() {
        check_grammar(
            r#"root ::= "a"+"#,
            &[b"a", b"aaaaa", b"aaaaaaaaaaaaaaaaaa"],
            &[b"", b"b", b"ab", b"aab", b"ba"],
        );
    }

    #[test]
    fn llama_optional_quantifier() {
        check_grammar(
            r#"root ::= "a"?"#,
            &[b"", b"a"],
            &[b"b", b"ab", b"aa", b"ba"],
        );
    }

    #[test]
    fn llama_mixed_quantifiers() {
        // From test-grammar-integration.cpp: "mixed quantifiers"
        check_grammar(
            r#"
                root ::= cons+ vowel* cons? (vowel cons)*
                vowel ::= [aeiouy]
                cons ::= [bcdfghjklmnpqrstvwxyz]
            "#,
            &[b"yes", b"no", b"noyes", b"crwth", b"four", b"bryyyy"],
            &[b"yess", b"yesno", b"forty", b"catyyy"],
        );
    }

    #[test]
    fn llama_exact_repetition() {
        check_grammar(
            r#"root ::= [ab]{4}"#,
            &[b"aaaa", b"bbbb", b"abab"],
            &[b"a", b"b", b"aaaaa"],
        );
    }

    #[test]
    fn llama_min_repetition() {
        check_grammar(
            r#"root ::= [ab]{4,}"#,
            &[b"aaaa", b"aaaaab", b"bbbb", b"ababab"],
            &[b"", b"aba"],
        );
    }

    #[test]
    fn llama_max_repetition() {
        check_grammar(
            r#"root ::= [ab]{0,4}"#,
            &[b"", b"a", b"aa", b"aaa", b"aaab"],
            &[b"aaaaa"],
        );
    }

    #[test]
    fn llama_min_max_repetition_with_group() {
        // From test-grammar-integration.cpp: "min / max repetition"
        // Uses parenthesized groups!
        check_grammar(
            r#"root ::= ("0x" [A-F0-9]{2} " "?){3,5}"#,
            &[b"0xFF 0x12 0xAB", b"0xFF 0x12 0xAB 0x00 0x00"],
            &[
                b"",
                b"0xFF",
                b"0xFF 0x12",
                b"0xFF 0x12 0xAB 0x00 0x00 0x00",
            ],
        );
    }
}
