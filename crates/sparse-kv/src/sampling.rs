/// Token sampling from logit distributions.
///
/// Pipeline: grammar mask → repetition penalty → temperature → top-k → softmax → top-p → sample.
/// Uses SplitMix64 PRNG — no external dependencies.

use crate::grammar::TokenMask;

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus): keep smallest set with cumulative prob >= p. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty applied to recently seen tokens. 1.0 = disabled.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered for repetition penalty.
    pub repeat_window: usize,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_window: 64,
            seed: 42,
        }
    }
}

/// Token sampler with internal RNG state.
pub struct Sampler {
    pub params: SamplerParams,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(params: SamplerParams) -> Self {
        let rng = SplitMix64::new(params.seed);
        Self { params, rng }
    }

    /// Sample a token ID from logits, penalizing tokens in `recent_tokens`.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[i32]) -> i32 {
        sample_token(logits, recent_tokens, None, &self.params, &mut self.rng)
    }

    /// Sample with grammar constraint. Disallowed tokens are masked to -inf
    /// before any other sampling step.
    pub fn sample_with_grammar(
        &mut self,
        logits: &[f32],
        recent_tokens: &[i32],
        grammar_mask: &TokenMask,
    ) -> i32 {
        sample_token(logits, recent_tokens, Some(grammar_mask), &self.params, &mut self.rng)
    }
}

fn sample_token(
    logits: &[f32],
    recent_tokens: &[i32],
    grammar_mask: Option<&TokenMask>,
    params: &SamplerParams,
    rng: &mut SplitMix64,
) -> i32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    // Work on a mutable copy of logits paired with token IDs.
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();

    // 0. Grammar mask: set disallowed tokens to -inf.
    if let Some(mask) = grammar_mask {
        for (id, logit) in candidates.iter_mut() {
            if !mask.is_set(*id) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    // 1. Repetition penalty: reduce logits of recently seen tokens.
    //    Positive logits are divided, negative logits are multiplied,
    //    so the penalty always pushes toward less likely.
    if params.repeat_penalty != 1.0 {
        for &tok in recent_tokens {
            let idx = tok as usize;
            if let Some((_, logit)) = candidates.get_mut(idx) {
                if *logit > 0.0 {
                    *logit /= params.repeat_penalty;
                } else {
                    *logit *= params.repeat_penalty;
                }
            }
        }
    }

    // 2. Greedy decoding (temperature = 0): return argmax.
    if params.temperature <= 0.0 {
        return candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|c| c.0 as i32)
            .unwrap_or(0);
    }

    // 3. Temperature scaling.
    if params.temperature != 1.0 {
        let inv_temp = 1.0 / params.temperature;
        for (_, logit) in candidates.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // 4. Top-k: partition to keep only the k highest logits.
    let k = if params.top_k > 0 && params.top_k < n {
        params.top_k
    } else {
        n
    };
    if k < n {
        candidates.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
    }

    // Sort descending by logit (needed for top-p cumulative scan).
    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
    });

    // 5. Softmax: convert logits to probabilities.
    let max_logit = candidates[0].1;
    let mut sum = 0.0f32;
    for (_, logit) in candidates.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    let inv_sum = 1.0 / sum;
    for (_, prob) in candidates.iter_mut() {
        *prob *= inv_sum;
    }

    // 6. Top-p (nucleus): keep smallest set with cumulative prob >= p.
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, &(_, prob)) in candidates.iter().enumerate() {
            cumsum += prob;
            if cumsum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
    }

    // 7. Weighted random sample from the remaining candidates.
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r = rng.next_f32() * total;
    let mut cumsum = 0.0f32;
    for &(id, prob) in &candidates {
        cumsum += prob;
        if cumsum > r {
            return id as i32;
        }
    }

    // Float precision fallback
    candidates.last().map(|&(id, _)| id as i32).unwrap_or(0)
}

/// SplitMix64 PRNG.
///
/// Fast, statistically solid for non-cryptographic use. Single u64 state.
/// Used by Java's SplittableRandom and as the standard seeder for xoshiro.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1) with full mantissa precision.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut sampler = Sampler::new(SamplerParams {
            temperature: 0.0,
            ..Default::default()
        });
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn greedy_with_repeat_penalty() {
        let mut sampler = Sampler::new(SamplerParams {
            temperature: 0.0,
            repeat_penalty: 100.0,
            ..Default::default()
        });
        let logits = vec![1.0, 5.0, 4.9, 3.0];
        // Token 1 penalized: 5.0/100 = 0.05 → token 2 (4.9) wins
        assert_eq!(sampler.sample(&logits, &[1]), 2);
    }

    #[test]
    fn negative_logit_repeat_penalty() {
        let mut sampler = Sampler::new(SamplerParams {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Default::default()
        });
        // Token 0 has logit -1.0; with penalty: -1.0 * 2.0 = -2.0 (pushed more negative)
        let logits = vec![-1.0, -0.5];
        assert_eq!(sampler.sample(&logits, &[0]), 1);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut s1 = Sampler::new(SamplerParams::default());
        let mut s2 = Sampler::new(SamplerParams::default());

        let t1 = s1.sample(&logits, &[]);
        let t2 = s2.sample(&logits, &[]);
        assert_eq!(t1, t2);
    }

    #[test]
    fn different_seeds_differ() {
        let logits: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut s1 = Sampler::new(SamplerParams { seed: 1, ..Default::default() });
        let mut s2 = Sampler::new(SamplerParams { seed: 999, ..Default::default() });

        // Over many samples, different seeds should produce different sequences
        let mut same = 0;
        for _ in 0..20 {
            if s1.sample(&logits, &[]) == s2.sample(&logits, &[]) {
                same += 1;
            }
        }
        assert!(same < 18, "seeds should produce different sequences, got {same}/20 same");
    }

    #[test]
    fn top_k_limits_candidates() {
        let mut logits = vec![0.0; 100];
        logits[3] = 100.0;
        logits[4] = 99.0;

        // With top_k=2, only tokens 3 and 4 should ever appear
        for seed in 0..50 {
            let mut s = Sampler::new(SamplerParams {
                temperature: 1.0,
                top_k: 2,
                top_p: 1.0,
                repeat_penalty: 1.0,
                repeat_window: 0,
                seed,
            });
            let tok = s.sample(&logits, &[]);
            assert!(tok == 3 || tok == 4, "top_k=2 produced token {tok}");
        }
    }

    #[test]
    fn top_p_concentrates() {
        // One dominant token
        let mut logits = vec![-100.0; 10];
        logits[5] = 100.0;

        let mut sampler = Sampler::new(SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repeat_penalty: 1.0,
            repeat_window: 0,
            seed: 42,
        });
        assert_eq!(sampler.sample(&logits, &[]), 5);
    }

    #[test]
    fn empty_logits_returns_zero() {
        let mut sampler = Sampler::new(SamplerParams::default());
        assert_eq!(sampler.sample(&[], &[]), 0);
    }

    #[test]
    fn rng_uniform_distribution() {
        let mut rng = SplitMix64::new(12345);
        let n = 10_000;
        let mut sum = 0.0f64;
        let mut in_range = true;
        for _ in 0..n {
            let v = rng.next_f32();
            if v < 0.0 || v >= 1.0 {
                in_range = false;
            }
            sum += v as f64;
        }
        let mean = sum / n as f64;
        assert!(in_range, "RNG produced value outside [0, 1)");
        assert!((mean - 0.5).abs() < 0.02, "mean was {mean}, expected ~0.5");
    }

    #[test]
    fn high_temperature_spreads_distribution() {
        // With high temperature, even low-logit tokens get sampled
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let mut counts = [0u32; 5];
        for seed in 0..500 {
            let mut s = Sampler::new(SamplerParams {
                temperature: 5.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                repeat_window: 0,
                seed,
            });
            let tok = s.sample(&logits, &[]) as usize;
            counts[tok] += 1;
        }
        // With temp=5.0, non-zero tokens should appear at least sometimes
        let non_zero_sampled: u32 = counts[1..].iter().sum();
        assert!(
            non_zero_sampled > 50,
            "high temperature should spread distribution, got {counts:?}"
        );
    }

    #[test]
    fn low_temperature_concentrates() {
        let logits = vec![10.0, 9.5, 0.0, 0.0, 0.0];
        let mut top_count = 0u32;
        for seed in 0..200 {
            let mut s = Sampler::new(SamplerParams {
                temperature: 0.1,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                repeat_window: 0,
                seed,
            });
            if s.sample(&logits, &[]) == 0 {
                top_count += 1;
            }
        }
        // With temp=0.1, token 0 (logit 10.0) should dominate
        assert!(
            top_count > 180,
            "low temperature should concentrate on top token, got {top_count}/200"
        );
    }
}
