use ggml_sys::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32 = ffi::ggml_type_GGML_TYPE_F32,
    F16 = ffi::ggml_type_GGML_TYPE_F16,
    Q4_0 = ffi::ggml_type_GGML_TYPE_Q4_0,
    Q4_1 = ffi::ggml_type_GGML_TYPE_Q4_1,
    Q5_0 = ffi::ggml_type_GGML_TYPE_Q5_0,
    Q5_1 = ffi::ggml_type_GGML_TYPE_Q5_1,
    Q8_0 = ffi::ggml_type_GGML_TYPE_Q8_0,
    Q8_1 = ffi::ggml_type_GGML_TYPE_Q8_1,
    Q2_K = ffi::ggml_type_GGML_TYPE_Q2_K,
    Q3_K = ffi::ggml_type_GGML_TYPE_Q3_K,
    Q4_K = ffi::ggml_type_GGML_TYPE_Q4_K,
    Q5_K = ffi::ggml_type_GGML_TYPE_Q5_K,
    Q6_K = ffi::ggml_type_GGML_TYPE_Q6_K,
    BF16 = ffi::ggml_type_GGML_TYPE_BF16,
    I32 = ffi::ggml_type_GGML_TYPE_I32,
}

impl QuantType {
    pub fn to_ggml_type(self) -> u32 {
        self as u32
    }

    pub fn from_ggml_type(t: u32) -> Option<Self> {
        Some(match t {
            ffi::ggml_type_GGML_TYPE_F32 => Self::F32,
            ffi::ggml_type_GGML_TYPE_F16 => Self::F16,
            ffi::ggml_type_GGML_TYPE_Q4_0 => Self::Q4_0,
            ffi::ggml_type_GGML_TYPE_Q4_1 => Self::Q4_1,
            ffi::ggml_type_GGML_TYPE_Q5_0 => Self::Q5_0,
            ffi::ggml_type_GGML_TYPE_Q5_1 => Self::Q5_1,
            ffi::ggml_type_GGML_TYPE_Q8_0 => Self::Q8_0,
            ffi::ggml_type_GGML_TYPE_Q8_1 => Self::Q8_1,
            ffi::ggml_type_GGML_TYPE_Q2_K => Self::Q2_K,
            ffi::ggml_type_GGML_TYPE_Q3_K => Self::Q3_K,
            ffi::ggml_type_GGML_TYPE_Q4_K => Self::Q4_K,
            ffi::ggml_type_GGML_TYPE_Q5_K => Self::Q5_K,
            ffi::ggml_type_GGML_TYPE_Q6_K => Self::Q6_K,
            ffi::ggml_type_GGML_TYPE_BF16 => Self::BF16,
            ffi::ggml_type_GGML_TYPE_I32 => Self::I32,
            _ => return None,
        })
    }

    /// Bytes per quantization block.
    pub fn block_size_bytes(self) -> usize {
        // SAFETY: querying a valid ggml type for its type size.
        unsafe { ffi::ggml_type_size(self.to_ggml_type()) }
    }

    /// Elements (floats) per quantization block.
    pub fn block_elements(self) -> usize {
        // SAFETY: querying a valid ggml type for its block size.
        unsafe { ffi::ggml_blck_size(self.to_ggml_type()) as usize }
    }

    /// Total bytes needed for `n_elements` values of this type.
    pub fn row_size(self, n_elements: usize) -> usize {
        let blk = self.block_elements();
        assert!(blk > 0, "invalid block size for {:?}", self);
        let n_blocks = (n_elements + blk - 1) / blk;
        n_blocks * self.block_size_bytes()
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::I32)
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::I32 => "I32",
        };
        f.write_str(name)
    }
}
