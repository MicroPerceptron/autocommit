pub const ANALYZE_GBNF: &str = r#"
root ::= object
object ::= "{" ws "\"summary\"" ws ":" ws string ws "," ws "\"items\"" ws ":" ws "[" ws "]" ws "}"
string ::= "\"" chars "\""
chars ::= [^"\\]*
ws ::= [ \t\n\r]*
"#;

pub const REDUCE_GBNF: &str = ANALYZE_GBNF;
