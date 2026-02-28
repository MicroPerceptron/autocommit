use crate::quant::QuantType;

/// Zero-copy view into a quantized weight buffer (typically memory-mapped).
#[derive(Debug, Clone, Copy)]
pub struct QuantSlice<'a> {
    pub quant_type: QuantType,
    pub shape: [usize; 2],
    pub data: &'a [u8],
}

impl<'a> QuantSlice<'a> {
    pub fn new(quant_type: QuantType, shape: [usize; 2], data: &'a [u8]) -> Self {
        Self {
            quant_type,
            shape,
            data,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.shape[1]
    }

    pub fn n_cols(&self) -> usize {
        self.shape[0]
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}
