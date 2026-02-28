use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    Load(String),
    Shape(String),
    Compute(String),
    Arena(String),
    Agent(String),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(msg) => write!(f, "load error: {msg}"),
            Self::Shape(msg) => write!(f, "shape mismatch: {msg}"),
            Self::Compute(msg) => write!(f, "compute error: {msg}"),
            Self::Arena(msg) => write!(f, "arena error: {msg}"),
            Self::Agent(msg) => write!(f, "agent error: {msg}"),
        }
    }
}

impl Error for InferenceError {}

impl From<std::io::Error> for InferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Load(value.to_string())
    }
}
