use std::collections::HashMap;

use crate::quant::QuantType;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub quant_type: QuantType,
    pub dims: Vec<u64>,
    pub offset: u64,
    pub size: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    pub fn shape_2d(&self) -> [usize; 2] {
        match self.dims.len() {
            0 => [1, 1],
            1 => [self.dims[0] as usize, 1],
            _ => [self.dims[0] as usize, self.dims[1] as usize],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WeightMap {
    pub entries: HashMap<String, TensorInfo>,
}

impl WeightMap {
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.entries.get(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }
}
