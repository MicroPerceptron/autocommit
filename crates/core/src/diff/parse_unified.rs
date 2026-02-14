use crate::types::LineRange;

pub fn extract_ranges(text: &str) -> Vec<LineRange> {
    text.lines().filter_map(parse_hunk_header).collect()
}

pub fn parse_hunk_header(line: &str) -> Option<LineRange> {
    if !line.starts_with("@@") {
        return None;
    }

    let header_end = line.rfind("@@")?;
    let body = line.get(2..header_end)?.trim();
    let mut parts = body.split_whitespace();

    let old_part = parts.next()?;
    let new_part = parts.next()?;

    let (old_start, old_count) = parse_side(old_part, '-')?;
    let (new_start, new_count) = parse_side(new_part, '+')?;

    Some(LineRange {
        old_start,
        old_count,
        new_start,
        new_count,
    })
}

fn parse_side(part: &str, prefix: char) -> Option<(i32, i32)> {
    if !part.starts_with(prefix) {
        return None;
    }

    let raw = &part[1..];
    let mut fields = raw.split(',');
    let start = fields.next()?.parse().ok()?;
    let count = fields.next().unwrap_or("1").parse().ok()?;
    Some((start, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unified_hunk_ranges() {
        let range = parse_hunk_header("@@ -10,4 +20,9 @@ fn sample() {").expect("range");
        assert_eq!(range.old_start, 10);
        assert_eq!(range.old_count, 4);
        assert_eq!(range.new_start, 20);
        assert_eq!(range.new_count, 9);
    }
}
