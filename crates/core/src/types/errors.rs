use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidDiff(String),
    Validation(String),
    Engine(String),
    Io(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDiff(msg) => write!(f, "invalid diff: {msg}"),
            Self::Validation(msg) => write!(f, "validation failed: {msg}"),
            Self::Engine(msg) => write!(f, "engine error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl Error for CoreError {}

impl From<std::io::Error> for CoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(value: serde_json::Error) -> Self {
        Self::Validation(value.to_string())
    }
}
