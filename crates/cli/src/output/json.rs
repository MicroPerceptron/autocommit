use serde::Serialize;

pub fn to_pretty_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut rendered = serde_json::to_string_pretty(value)?;
    rendered.push('\n');
    Ok(rendered)
}
