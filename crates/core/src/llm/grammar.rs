pub const ANALYZE_GBNF: &str = r#"
root ::= object ws
object ::= "{" ws "\"summary\"" ws ":" ws string "," ws "\"bucket\"" ws ":" ws bucket "," ws "\"type_tag\"" ws ":" ws typetag "," ws "\"title\"" ws ":" ws string "," ws "\"intent\"" ws ":" ws string "}" ws

bucket ::= "\"Feature\"" | "\"Patch\"" | "\"Addition\"" | "\"Other\""
typetag ::= "\"Feat\"" | "\"Fix\"" | "\"Refactor\"" | "\"Docs\"" | "\"Test\"" | "\"Chore\"" | "\"Perf\"" | "\"Style\"" | "\"Mixed\""

string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= | " " | "\n" [ \t]{0,20}
"#;

pub const REDUCE_GBNF: &str = r#"
root ::= object ws
object ::= "{" ws "\"commit_message\"" ws ":" ws string "," ws "\"summary\"" ws ":" ws string "," ws "\"risk_level\"" ws ":" ws risklevel "," ws "\"risk_notes\"" ws ":" ws stringarray "}" ws

risklevel ::= "\"low\"" | "\"medium\"" | "\"high\""
stringarray ::= "[" ws (string (ws "," ws string)*)? ws "]" ws

string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= | " " | "\n" [ \t]{0,20}
"#;
