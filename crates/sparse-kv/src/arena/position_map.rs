use crate::arena::slot::LogicalPos;

/// Maps dense (compacted) indices to their original logical sequence positions.
/// Used to compute correct RoPE angles after sparse compaction.
#[derive(Debug, Clone)]
pub struct PositionMap {
    pub positions: Vec<LogicalPos>,
}

impl PositionMap {
    pub fn new(positions: Vec<LogicalPos>) -> Self {
        Self { positions }
    }

    pub fn from_contiguous(len: usize) -> Self {
        Self {
            positions: (0..len as LogicalPos).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Compute RoPE (cos, sin) frequency pairs for each position.
    /// Returns two vectors of length `positions.len() * half_dim`.
    ///
    /// RoPE formula: for position `p` and dimension index `i` (0..half_dim):
    ///   freq = p / theta^(2*i / dim)
    ///   cos_cache[p * half_dim + i] = cos(freq)
    ///   sin_cache[p * half_dim + i] = sin(freq)
    pub fn rope_cos_sin(&self, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
        let half_dim = head_dim / 2;
        let n = self.positions.len();
        let mut cos_cache = vec![0.0f32; n * half_dim];
        let mut sin_cache = vec![0.0f32; n * half_dim];

        for (idx, &pos) in self.positions.iter().enumerate() {
            let p = pos as f32;
            for i in 0..half_dim {
                let freq = p / theta.powf(2.0 * i as f32 / head_dim as f32);
                cos_cache[idx * half_dim + i] = freq.cos();
                sin_cache[idx * half_dim + i] = freq.sin();
            }
        }

        (cos_cache, sin_cache)
    }

    /// Get the logical positions as i32 (for passing to ggml_rope_ext).
    pub fn as_i32(&self) -> Vec<i32> {
        self.positions.iter().map(|&p| p as i32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_position_map() {
        let pm = PositionMap::from_contiguous(4);
        assert_eq!(pm.positions, vec![0, 1, 2, 3]);
        assert_eq!(pm.len(), 4);
    }

    #[test]
    fn sparse_position_map_preserves_logical() {
        let pm = PositionMap::new(vec![0, 3, 7, 15]);
        assert_eq!(pm.as_i32(), vec![0, 3, 7, 15]);
    }

    #[test]
    fn rope_cos_sin_dimensions() {
        let pm = PositionMap::from_contiguous(4);
        let (cos, sin) = pm.rope_cos_sin(8, 10000.0);
        assert_eq!(cos.len(), 4 * 4); // 4 positions * (8/2) half_dim
        assert_eq!(sin.len(), 4 * 4);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let pm = PositionMap::new(vec![0]);
        let (cos, sin) = pm.rope_cos_sin(4, 10000.0);
        // At position 0, freq = 0/theta^x = 0, so cos(0)=1, sin(0)=0
        for &c in &cos {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for &s in &sin {
            assert!(s.abs() < 1e-6);
        }
    }
}
