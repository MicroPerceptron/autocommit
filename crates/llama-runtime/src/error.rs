use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Init(String),
    State(String),
    Inference(String),
    Embed(String),
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(msg) => write!(f, "runtime init error: {msg}"),
            Self::State(msg) => write!(f, "runtime state error: {msg}"),
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
            Self::Embed(msg) => write!(f, "embedding error: {msg}"),
        }
    }
}

impl Error for RuntimeError {}
