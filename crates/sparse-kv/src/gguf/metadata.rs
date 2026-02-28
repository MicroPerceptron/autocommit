use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => u32::try_from(*v).ok(),
            Self::U64(v) => u32::try_from(*v).ok(),
            Self::I64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            Self::U32(v) => Some(*v as f32),
            Self::I32(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    pub entries: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.entries.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(|v| v.as_u32())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(|v| v.as_f32())
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }
}
