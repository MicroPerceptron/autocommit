/// GBNF grammar parser.
///
/// Compiles GBNF text into a flat element array with rule references.
/// Quantifiers (*, +, ?) are desugared into generated helper rules.

use crate::error::InferenceError;

/// A single grammar element in the compiled representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarElement {
    /// End of a rule alternative.
    End,
    /// Start of a character class. value = first range lower bound.
    Char(u32),
    /// Upper bound of the current range.
    CharRangeUpper(u32),
    /// Additional range in the SAME character class (lower bound).
    /// Distinguishes multi-range classes ([a-zA-Z]) from sequential
    /// char literals ("ab") which use separate Char elements.
    CharAlt(u32),
    /// Start of a negated character class.
    CharNot(u32),
    /// Reference to another rule by index.
    RuleRef(u32),
}

/// Compiled grammar: flat element array + rule index.
pub struct CompiledGrammar {
    /// Flat array of elements. Each rule alternative is a sequence
    /// of elements ending with `End`.
    pub elements: Vec<GrammarElement>,
    /// For each rule_id: list of start positions in `elements`
    /// (one per alternative).
    pub rule_alts: Vec<Vec<usize>>,
    /// Rule names (for diagnostics).
    pub rule_names: Vec<String>,
    /// Index of the root rule.
    pub root_rule: usize,
}

/// Parse a GBNF grammar string into a compiled grammar.
pub fn parse_gbnf(input: &str) -> Result<CompiledGrammar, InferenceError> {
    let mut ctx = ParseCtx {
        rule_names: Vec::new(),
        rule_alts: Vec::new(),
        elements: Vec::new(),
        deferred: Vec::new(),
    };

    // First pass: collect all rule names so forward references work.
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(pos) = line.find("::=") {
            let name = line[..pos].trim();
            if !name.is_empty() {
                ctx.get_or_create_rule(name);
            }
        }
    }

    // Second pass: parse rule bodies.
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(pos) = line.find("::=") {
            let name = line[..pos].trim().to_string();
            let body = line[pos + 3..].trim();
            let rule_id = ctx.get_or_create_rule(&name);
            parse_rule_body(&mut ctx, rule_id, body)?;
        }
    }

    // Flush deferred helper rule alternatives into the elements array.
    // This must happen after all rule bodies are parsed so that helper
    // elements don't interleave with the current rule's sequence.
    flush_deferred(&mut ctx);

    let root_rule = if ctx.rule_names.is_empty() {
        return Err(InferenceError::Tokenize("empty grammar".into()));
    } else {
        0 // first rule is root
    };

    // Verify all rules have at least one alternative
    for (i, name) in ctx.rule_names.iter().enumerate() {
        if ctx.rule_alts[i].is_empty() {
            return Err(InferenceError::Tokenize(format!(
                "rule '{name}' referenced but never defined"
            )));
        }
    }

    Ok(CompiledGrammar {
        elements: ctx.elements,
        rule_alts: ctx.rule_alts,
        rule_names: ctx.rule_names,
        root_rule,
    })
}

struct ParseCtx {
    rule_names: Vec<String>,
    rule_alts: Vec<Vec<usize>>,
    elements: Vec<GrammarElement>,
    /// Deferred helper rule alternatives from quantifier desugaring.
    /// Each entry is (rule_id, alternative_elements).
    /// Flushed after all rule bodies are parsed, keeping each rule's
    /// elements contiguous (no interleaving with the current alternative).
    deferred: Vec<(usize, Vec<GrammarElement>)>,
}

impl ParseCtx {
    fn get_or_create_rule(&mut self, name: &str) -> usize {
        if let Some(idx) = self.rule_names.iter().position(|n| n == name) {
            idx
        } else {
            let idx = self.rule_names.len();
            self.rule_names.push(name.to_string());
            self.rule_alts.push(Vec::new());
            idx
        }
    }

    /// Create a generated helper rule (for quantifiers).
    fn gen_rule(&mut self) -> usize {
        let name = format!("_gen{}", self.rule_names.len());
        self.get_or_create_rule(&name)
    }
}

/// Parse one rule body (everything after `::=`), handling `|` alternatives.
fn parse_rule_body(ctx: &mut ParseCtx, rule_id: usize, body: &str) -> Result<(), InferenceError> {
    let bytes = body.as_bytes();
    let mut pos = 0;

    loop {
        skip_ws(bytes, &mut pos);
        let alt_start = ctx.elements.len();
        parse_alternative(ctx, bytes, &mut pos)?;
        ctx.elements.push(GrammarElement::End);
        ctx.rule_alts[rule_id].push(alt_start);

        skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b'|' {
            pos += 1; // consume '|'
        } else {
            break;
        }
    }
    Ok(())
}

/// Parse one alternative (sequence of terms).
fn parse_alternative(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    loop {
        skip_ws(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] == b'|' || bytes[*pos] == b')' {
            break;
        }
        parse_term(ctx, bytes, pos)?;
    }
    Ok(())
}

/// Parse a single term: atom + optional quantifier.
fn parse_term(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    // Remember where this term's elements start
    let term_start = ctx.elements.len();

    parse_atom(ctx, bytes, pos)?;

    // Check for quantifier
    skip_ws(bytes, pos);
    if *pos < bytes.len() {
        match bytes[*pos] {
            b'*' => {
                *pos += 1;
                desugar_star(ctx, term_start);
            }
            b'+' => {
                *pos += 1;
                desugar_plus(ctx, term_start);
            }
            b'?' => {
                *pos += 1;
                desugar_optional(ctx, term_start);
            }
            b'{' => {
                *pos += 1;
                let (min, max) = parse_repeat_range(bytes, pos)?;
                desugar_repeat(ctx, term_start, min, max);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse an atom: string literal, char class, rule reference, or group.
fn parse_atom(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    if *pos >= bytes.len() {
        return Err(InferenceError::Tokenize("unexpected end of grammar".into()));
    }

    match bytes[*pos] {
        b'"' => parse_string_literal(ctx, bytes, pos),
        b'[' => parse_char_class(ctx, bytes, pos),
        b'(' => parse_group(ctx, bytes, pos),
        _ => parse_rule_ref(ctx, bytes, pos),
    }
}

/// Parse a parenthesized group: `(...)` → helper rule + RuleRef.
///
/// Creates a generated rule for the group body and emits a RuleRef.
/// Group body elements are collected separately and deferred, just like
/// quantifier helpers, to prevent interleaving with the parent rule's
/// element sequence.
fn parse_group(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    *pos += 1; // consume '('
    let group_rule = ctx.gen_rule();

    // Parse alternatives within the group into a temporary element buffer.
    // We swap out ctx.elements so the group body doesn't interleave with
    // the parent rule's elements.
    let parent_elements = std::mem::take(&mut ctx.elements);

    loop {
        skip_ws(bytes, pos);
        let alt_start = ctx.elements.len();
        parse_alternative(ctx, bytes, pos)?;
        ctx.elements.push(GrammarElement::End);
        // Record as (alt_start, elements) — we'll flush via deferred
        let alt_elements = ctx.elements[alt_start..].to_vec();
        ctx.elements.truncate(alt_start);
        ctx.deferred.push((group_rule, alt_elements));

        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'|' {
            *pos += 1;
        } else {
            break;
        }
    }

    // Restore parent elements and add the RuleRef
    ctx.elements = parent_elements;

    if *pos >= bytes.len() || bytes[*pos] != b')' {
        return Err(InferenceError::Tokenize("unterminated parenthesized group".into()));
    }
    *pos += 1; // consume ')'

    ctx.elements.push(GrammarElement::RuleRef(group_rule as u32));
    Ok(())
}

/// Parse a string literal: `"..."` → sequence of Char elements.
fn parse_string_literal(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    *pos += 1; // skip opening '"'
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        let cp = parse_escape_or_char(bytes, pos)?;
        ctx.elements.push(GrammarElement::Char(cp));
        ctx.elements.push(GrammarElement::CharRangeUpper(cp));
    }
    if *pos >= bytes.len() {
        return Err(InferenceError::Tokenize("unterminated string literal".into()));
    }
    *pos += 1; // skip closing '"'
    Ok(())
}

/// Parse a character class: `[...]` or `[^...]`.
///
/// First range uses Char/CharNot; subsequent ranges use CharAlt.
/// This distinguishes multi-range classes from sequential char literals.
fn parse_char_class(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    *pos += 1; // skip '['

    let negated = *pos < bytes.len() && bytes[*pos] == b'^';
    if negated {
        *pos += 1;
    }

    let mut first = true;
    while *pos < bytes.len() && bytes[*pos] != b']' {
        let cp = parse_escape_or_char(bytes, pos)?;

        // Check for range: a-z
        let (lo, hi) = if *pos + 1 < bytes.len()
            && bytes[*pos] == b'-'
            && bytes[*pos + 1] != b']'
        {
            *pos += 1; // skip '-'
            let cp_end = parse_escape_or_char(bytes, pos)?;
            (cp, cp_end)
        } else {
            (cp, cp) // single char = range [cp, cp]
        };

        if first {
            ctx.elements.push(if negated {
                GrammarElement::CharNot(lo)
            } else {
                GrammarElement::Char(lo)
            });
            first = false;
        } else {
            ctx.elements.push(GrammarElement::CharAlt(lo));
        }
        ctx.elements.push(GrammarElement::CharRangeUpper(hi));
    }
    if *pos >= bytes.len() {
        return Err(InferenceError::Tokenize("unterminated character class".into()));
    }
    *pos += 1; // skip ']'
    Ok(())
}

/// Parse a rule reference (identifier).
fn parse_rule_ref(
    ctx: &mut ParseCtx,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(), InferenceError> {
    let start = *pos;
    while *pos < bytes.len()
        && (bytes[*pos].is_ascii_alphanumeric() || bytes[*pos] == b'-' || bytes[*pos] == b'_')
    {
        *pos += 1;
    }
    if *pos == start {
        return Err(InferenceError::Tokenize(format!(
            "unexpected character '{}' at position {start}",
            bytes[start] as char
        )));
    }
    let name = std::str::from_utf8(&bytes[start..*pos]).map_err(|_| {
        InferenceError::Tokenize("invalid UTF-8 in rule name".into())
    })?;
    let rule_id = ctx.get_or_create_rule(name);
    ctx.elements.push(GrammarElement::RuleRef(rule_id as u32));
    Ok(())
}

/// Parse a single character or escape sequence, advancing `pos`.
fn parse_escape_or_char(bytes: &[u8], pos: &mut usize) -> Result<u32, InferenceError> {
    if *pos >= bytes.len() {
        return Err(InferenceError::Tokenize("unexpected end in character".into()));
    }
    if bytes[*pos] == b'\\' {
        *pos += 1;
        if *pos >= bytes.len() {
            return Err(InferenceError::Tokenize("trailing backslash".into()));
        }
        let c = bytes[*pos];
        *pos += 1;
        match c {
            b'n' => Ok(0x0A),
            b'r' => Ok(0x0D),
            b't' => Ok(0x09),
            b'\\' => Ok(0x5C),
            b'"' => Ok(0x22),
            b'[' => Ok(0x5B),
            b']' => Ok(0x5D),
            b'x' => {
                // \xHH
                let hex = parse_hex_digits(bytes, pos, 2)?;
                Ok(hex)
            }
            b'u' => {
                // \uHHHH
                let hex = parse_hex_digits(bytes, pos, 4)?;
                Ok(hex)
            }
            _ => Ok(c as u32),
        }
    } else {
        // Regular UTF-8 character
        let c = decode_utf8_char(bytes, pos)?;
        Ok(c as u32)
    }
}

fn parse_hex_digits(bytes: &[u8], pos: &mut usize, count: usize) -> Result<u32, InferenceError> {
    let start = *pos;
    let end = (start + count).min(bytes.len());
    if end - start < count {
        return Err(InferenceError::Tokenize("not enough hex digits".into()));
    }
    let hex_str = std::str::from_utf8(&bytes[start..end]).map_err(|_| {
        InferenceError::Tokenize("invalid hex digits".into())
    })?;
    let val = u32::from_str_radix(hex_str, 16).map_err(|_| {
        InferenceError::Tokenize(format!("invalid hex: {hex_str}"))
    })?;
    *pos = end;
    Ok(val)
}

/// Decode one UTF-8 character, advancing pos.
fn decode_utf8_char(bytes: &[u8], pos: &mut usize) -> Result<char, InferenceError> {
    let b = bytes[*pos];
    let (char_len, codepoint) = if b < 0x80 {
        (1, b as u32)
    } else if b < 0xE0 {
        if *pos + 1 >= bytes.len() {
            return Err(InferenceError::Tokenize("truncated UTF-8".into()));
        }
        let cp = ((b as u32 & 0x1F) << 6) | (bytes[*pos + 1] as u32 & 0x3F);
        (2, cp)
    } else if b < 0xF0 {
        if *pos + 2 >= bytes.len() {
            return Err(InferenceError::Tokenize("truncated UTF-8".into()));
        }
        let cp = ((b as u32 & 0x0F) << 12)
            | ((bytes[*pos + 1] as u32 & 0x3F) << 6)
            | (bytes[*pos + 2] as u32 & 0x3F);
        (3, cp)
    } else {
        if *pos + 3 >= bytes.len() {
            return Err(InferenceError::Tokenize("truncated UTF-8".into()));
        }
        let cp = ((b as u32 & 0x07) << 18)
            | ((bytes[*pos + 1] as u32 & 0x3F) << 12)
            | ((bytes[*pos + 2] as u32 & 0x3F) << 6)
            | (bytes[*pos + 3] as u32 & 0x3F);
        (4, cp)
    };
    *pos += char_len;
    char::from_u32(codepoint)
        .ok_or_else(|| InferenceError::Tokenize(format!("invalid codepoint: U+{codepoint:04X}")))
}

/// Parse `{min,max}` repeat range.
fn parse_repeat_range(
    bytes: &[u8],
    pos: &mut usize,
) -> Result<(u32, Option<u32>), InferenceError> {
    skip_ws(bytes, pos);
    let min = parse_u32(bytes, pos)?;
    skip_ws(bytes, pos);
    let max = if *pos < bytes.len() && bytes[*pos] == b',' {
        *pos += 1;
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'}' {
            None // unbounded
        } else {
            Some(parse_u32(bytes, pos)?)
        }
    } else {
        Some(min) // {n} = exactly n
    };
    skip_ws(bytes, pos);
    if *pos < bytes.len() && bytes[*pos] == b'}' {
        *pos += 1;
    } else {
        return Err(InferenceError::Tokenize("unterminated repeat range".into()));
    }
    Ok((min, max))
}

fn parse_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, InferenceError> {
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return Err(InferenceError::Tokenize("expected number in repeat range".into()));
    }
    let s = std::str::from_utf8(&bytes[start..*pos]).unwrap();
    s.parse().map_err(|_| InferenceError::Tokenize(format!("invalid number: {s}")))
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

// --- Quantifier desugaring ---
//
// Helper rule bodies are pushed to `ctx.deferred` rather than directly
// into `ctx.elements`. This prevents interleaving helper elements with
// the current rule's alternative, which would corrupt return addresses
// in the pushdown automaton (pos+1 after a RuleRef must point to the
// next element in the SAME alternative, not into a helper body).

/// `expr*` → create helper rule: `_genN ::= expr _genN | ε`
fn desugar_star(ctx: &mut ParseCtx, term_start: usize) {
    let helper = ctx.gen_rule();
    let term_elements: Vec<GrammarElement> = ctx.elements[term_start..].to_vec();
    ctx.elements.truncate(term_start);
    ctx.elements.push(GrammarElement::RuleRef(helper as u32));

    // Alternative 1: expr helper (recursive)
    let mut alt1 = term_elements.clone();
    alt1.push(GrammarElement::RuleRef(helper as u32));
    alt1.push(GrammarElement::End);
    ctx.deferred.push((helper, alt1));

    // Alternative 2: ε (empty)
    ctx.deferred.push((helper, vec![GrammarElement::End]));
}

/// `expr+` → create helper rule: `_genN ::= expr _genN | expr`
fn desugar_plus(ctx: &mut ParseCtx, term_start: usize) {
    let helper = ctx.gen_rule();
    let term_elements: Vec<GrammarElement> = ctx.elements[term_start..].to_vec();
    ctx.elements.truncate(term_start);
    ctx.elements.push(GrammarElement::RuleRef(helper as u32));

    // Alternative 1: expr helper (recursive)
    let mut alt1 = term_elements.clone();
    alt1.push(GrammarElement::RuleRef(helper as u32));
    alt1.push(GrammarElement::End);
    ctx.deferred.push((helper, alt1));

    // Alternative 2: expr (base case, at least one)
    let mut alt2 = term_elements;
    alt2.push(GrammarElement::End);
    ctx.deferred.push((helper, alt2));
}

/// `expr?` → create helper rule: `_genN ::= expr | ε`
fn desugar_optional(ctx: &mut ParseCtx, term_start: usize) {
    let helper = ctx.gen_rule();
    let term_elements: Vec<GrammarElement> = ctx.elements[term_start..].to_vec();
    ctx.elements.truncate(term_start);
    ctx.elements.push(GrammarElement::RuleRef(helper as u32));

    // Alternative 1: expr
    let mut alt1 = term_elements;
    alt1.push(GrammarElement::End);
    ctx.deferred.push((helper, alt1));

    // Alternative 2: ε
    ctx.deferred.push((helper, vec![GrammarElement::End]));
}

/// `expr{min,max}` → unroll min mandatory copies + (max-min) optional copies.
fn desugar_repeat(ctx: &mut ParseCtx, term_start: usize, min: u32, max: Option<u32>) {
    let term_elements: Vec<GrammarElement> = ctx.elements[term_start..].to_vec();
    ctx.elements.truncate(term_start);

    // Emit `min` mandatory copies inline
    for _ in 0..min {
        ctx.elements.extend_from_slice(&term_elements);
    }

    // For the optional part: create a chain of optional rules
    let optional_count = match max {
        Some(m) => m.saturating_sub(min),
        None => {
            // Unbounded: use star-like recursion for the rest
            let helper = ctx.gen_rule();
            ctx.elements.push(GrammarElement::RuleRef(helper as u32));
            let mut alt1 = term_elements.clone();
            alt1.push(GrammarElement::RuleRef(helper as u32));
            alt1.push(GrammarElement::End);
            ctx.deferred.push((helper, alt1));
            ctx.deferred.push((helper, vec![GrammarElement::End]));
            return;
        }
    };

    for _ in 0..optional_count {
        let helper = ctx.gen_rule();
        ctx.elements.push(GrammarElement::RuleRef(helper as u32));
        let mut alt1 = term_elements.clone();
        alt1.push(GrammarElement::End);
        ctx.deferred.push((helper, alt1));
        ctx.deferred.push((helper, vec![GrammarElement::End]));
    }
}

/// Flush deferred helper rule alternatives into the elements array.
fn flush_deferred(ctx: &mut ParseCtx) {
    // Drain in order — deferred rules don't themselves defer further
    // (quantifier desugaring only copies already-parsed atom elements).
    for (rule_id, alt_elements) in std::mem::take(&mut ctx.deferred) {
        let alt_start = ctx.elements.len();
        ctx.elements.extend(alt_elements);
        ctx.rule_alts[rule_id].push(alt_start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_literal() {
        let g = parse_gbnf(r#"root ::= "hello""#).unwrap();
        assert_eq!(g.rule_names.len(), 1);
        assert_eq!(g.rule_names[0], "root");
        assert_eq!(g.rule_alts[0].len(), 1); // one alternative
        // 5 chars × (Char + CharRangeUpper) + End = 11 elements
        assert_eq!(g.elements.len(), 11);
    }

    #[test]
    fn parse_alternatives() {
        let g = parse_gbnf(r#"root ::= "a" | "b" | "c""#).unwrap();
        assert_eq!(g.rule_alts[0].len(), 3); // three alternatives
    }

    #[test]
    fn parse_char_class() {
        let g = parse_gbnf(r#"root ::= [a-z]"#).unwrap();
        // Char('a') + CharRangeUpper('z') + End = 3
        assert_eq!(g.elements.len(), 3);
        assert_eq!(g.elements[0], GrammarElement::Char(b'a' as u32));
        assert_eq!(g.elements[1], GrammarElement::CharRangeUpper(b'z' as u32));
    }

    #[test]
    fn parse_negated_char_class() {
        let g = parse_gbnf(r#"root ::= [^"\\]"#).unwrap();
        assert!(matches!(g.elements[0], GrammarElement::CharNot(_)));
    }

    #[test]
    fn parse_rule_reference() {
        let g = parse_gbnf("root ::= foo\nfoo ::= \"x\"").unwrap();
        assert_eq!(g.rule_names.len(), 2);
        assert_eq!(g.rule_names[0], "root");
        assert_eq!(g.rule_names[1], "foo");
        // root has a RuleRef to foo
        let root_start = g.rule_alts[0][0];
        assert_eq!(g.elements[root_start], GrammarElement::RuleRef(1));
    }

    #[test]
    fn parse_star_quantifier() {
        let g = parse_gbnf(r#"root ::= "a"*"#).unwrap();
        // root should contain a RuleRef to a generated helper
        assert!(g.rule_names.len() >= 2); // root + helper
    }

    #[test]
    fn parse_plus_quantifier() {
        let g = parse_gbnf(r#"root ::= [0-9]+"#).unwrap();
        assert!(g.rule_names.len() >= 2);
    }

    #[test]
    fn parse_optional_quantifier() {
        let g = parse_gbnf(r#"root ::= "x"?"#).unwrap();
        assert!(g.rule_names.len() >= 2);
    }

    #[test]
    fn parse_repeat_range() {
        let g = parse_gbnf(r#"root ::= [0-9]{1,3}"#).unwrap();
        // 1 mandatory + 2 optional helpers
        assert!(g.rule_names.len() >= 3);
    }

    #[test]
    fn parse_escape_sequences() {
        let g = parse_gbnf(r#"root ::= "\n\t\\\"""#).unwrap();
        let start = g.rule_alts[0][0];
        // \n = 0x0A
        assert_eq!(g.elements[start], GrammarElement::Char(0x0A));
    }

    #[test]
    fn parse_hex_escape() {
        let g = parse_gbnf(r#"root ::= [^\x00-\x1F]"#).unwrap();
        assert!(matches!(g.elements[0], GrammarElement::CharNot(0x00)));
        assert_eq!(g.elements[1], GrammarElement::CharRangeUpper(0x1F));
    }

    #[test]
    fn parse_multi_rule() {
        let g = parse_gbnf(
            "root ::= greeting ws\ngreeting ::= \"hello\" | \"hi\"\nws ::= \" \"",
        )
        .unwrap();
        assert_eq!(g.rule_names.len(), 3);
        assert_eq!(g.rule_alts[1].len(), 2); // greeting has 2 alternatives
    }

    #[test]
    fn parse_real_gbnf() {
        // Simplified version of the ANALYZE_GBNF from the codebase
        let gbnf = r#"
root ::= "{" ws "\"key\"" ws ":" ws string "}"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F]
ws ::= " "?
"#;
        let g = parse_gbnf(gbnf).unwrap();
        assert!(g.rule_names.contains(&"root".to_string()));
        assert!(g.rule_names.contains(&"string".to_string()));
        assert!(g.rule_names.contains(&"char".to_string()));
        assert!(g.rule_names.contains(&"ws".to_string()));
    }

    #[test]
    fn parse_group_simple() {
        // (a | b) should create a helper rule with 2 alternatives
        let g = parse_gbnf(r#"root ::= ("a" | "b")"#).unwrap();
        assert!(g.rule_names.len() >= 2); // root + group helper
        // Root should have exactly one alternative with one RuleRef
        assert_eq!(g.rule_alts[0].len(), 1);
        let root_start = g.rule_alts[0][0];
        assert!(matches!(g.elements[root_start], GrammarElement::RuleRef(_)));
    }

    #[test]
    fn parse_group_with_quantifier() {
        // ("a" | "b")+ should parse and accept "a", "ab", "ba", etc.
        let g = parse_gbnf(r#"root ::= ("a" | "b")+"#).unwrap();
        assert!(g.rule_names.len() >= 3); // root + group + plus helper
    }

    #[test]
    fn parse_group_nested() {
        // Nested groups: (("a"))
        let g = parse_gbnf(r#"root ::= (("a"))"#).unwrap();
        assert!(g.rule_names.len() >= 3); // root + outer group + inner group
    }

    #[test]
    fn parse_arithmetic_grammar() {
        // Matches llama.cpp test case: (expr "=" term "\n")+
        let g = parse_gbnf(r#"
            root ::= (expr "=" term "\n")+
            expr ::= term ([-+*/] term)*
            term ::= [0-9]+
        "#).unwrap();
        assert!(g.rule_names.contains(&"root".to_string()));
        assert!(g.rule_names.contains(&"expr".to_string()));
        assert!(g.rule_names.contains(&"term".to_string()));
    }

    #[test]
    fn parse_full_arithmetic_grammar() {
        // Matches llama.cpp's more complex arithmetic test case
        let g = parse_gbnf(r#"
            root  ::= (expr "=" ws term "\n")+
            expr  ::= term ([-+*/] term)*
            term  ::= ident | num | "(" ws expr ")" ws
            ident ::= [a-z] [a-z0-9_]* ws
            num   ::= [0-9]+ ws
            ws    ::= [ \t\n]*
        "#).unwrap();
        assert!(g.rule_names.contains(&"root".to_string()));
        assert!(g.rule_names.contains(&"expr".to_string()));
        assert!(g.rule_names.contains(&"term".to_string()));
        assert!(g.rule_names.contains(&"ident".to_string()));
        assert!(g.rule_names.contains(&"num".to_string()));
        assert!(g.rule_names.contains(&"ws".to_string()));
    }

    #[test]
    fn error_empty_grammar() {
        assert!(parse_gbnf("").is_err());
    }

    #[test]
    fn error_undefined_rule() {
        assert!(parse_gbnf("root ::= missing_rule").is_err());
    }

    #[test]
    fn error_unterminated_group() {
        assert!(parse_gbnf(r#"root ::= ("a""#).is_err());
    }
}
