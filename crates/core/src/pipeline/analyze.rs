use std::path::PathBuf;

use crate::CoreError;
use crate::cache::anchor_embeddings::{self, AnchorEmbeddingCache};
use crate::diff::collect;
use crate::diff::features::{self, DiffFeatures};
use crate::dispatch::{embedding_gate, heuristics, policy};
use crate::llm::prompts;
use crate::llm::traits::LlmEngine;
use crate::pipeline::{fanout, reduce, validate};
use crate::progress::{self, ProgressCallback, ProgressStage};
use crate::types::{AnalysisReport, DiffStats, DispatchDecision, DispatchRoute};

/// Default token cap per merged chunk. Prevents merging from creating prompts
/// that exceed the model's context window.
const MERGE_MAX_TOKENS: usize = 4_000;

/// When the total chunk count (after merging) is at or below this threshold,
/// skip the per-chunk fanout and go straight to a single reduce-style call.
const SMALL_CHUNK_THRESHOLD: usize = 3;

pub struct AnalyzeOptions {
    pub embedding_threshold: f32,
    pub anchor_cache_dir: Option<PathBuf>,
    pub progress: Option<ProgressCallback>,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            embedding_threshold: 0.72,
            anchor_cache_dir: None,
            progress: None,
        }
    }
}

impl std::fmt::Debug for AnalyzeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyzeOptions")
            .field("embedding_threshold", &self.embedding_threshold)
            .field("anchor_cache_dir", &self.anchor_cache_dir)
            .field("progress", &self.progress.as_ref().map(|_| ".."))
            .finish()
    }
}

pub fn run(
    engine: &dyn LlmEngine,
    diff_text: &str,
    options: &AnalyzeOptions,
) -> Result<AnalysisReport, CoreError> {
    let cb = options.progress.as_ref();

    // Forward the progress callback to the engine for Embedding/Analyze/Reduce events.
    engine.set_progress_callback(options.progress.clone());

    let result = run_inner(engine, diff_text, options, cb);

    engine.set_progress_callback(None);
    result
}

fn run_inner(
    engine: &dyn LlmEngine,
    diff_text: &str,
    options: &AnalyzeOptions,
    cb: Option<&ProgressCallback>,
) -> Result<AnalysisReport, CoreError> {
    let raw_chunks = collect::collect(diff_text);

    if raw_chunks.is_empty() {
        return Ok(AnalysisReport::empty(DispatchDecision {
            route: DispatchRoute::DraftOnly,
            reason_codes: vec!["empty_diff".to_string()],
            estimated_cost_tokens: 0,
        }));
    }

    // Feature extraction uses the original per-file chunks for accuracy.
    let features = features::extract(&raw_chunks);
    let stats = to_stats(&features);

    let heuristic = heuristics::score(&features);
    let embedding_hint = if embedding_gate::should_run_embedding(heuristic) {
        let anchor_cache = resolve_anchor_cache(options);
        let fingerprint = engine.model_fingerprint();

        // Try to load cached anchor embeddings.
        let anchors = anchor_cache
            .as_ref()
            .zip(fingerprint.as_deref())
            .and_then(|(cache, fp)| {
                cache.load(fp, prompts::DISPATCH_DRAFT_ANCHOR, prompts::DISPATCH_FULL_ANCHOR)
            });

        let (draft_anchor_embedding, full_anchor_embedding) = match anchors {
            Some(cached) => (cached.draft, cached.full),
            None => {
                let draft = engine.embed(prompts::DISPATCH_DRAFT_ANCHOR)?;
                let full = engine.embed(prompts::DISPATCH_FULL_ANCHOR)?;

                // Store for next run.
                if let Some((cache, fp)) = anchor_cache.as_ref().zip(fingerprint.as_deref()) {
                    cache.store(
                        fp,
                        prompts::DISPATCH_DRAFT_ANCHOR,
                        prompts::DISPATCH_FULL_ANCHOR,
                        &draft,
                        &full,
                    );
                }

                (draft, full)
            }
        };

        let prompt = prompts::build_embedding_prompt(&raw_chunks, 8_000);
        let signal_embedding = engine.embed(&prompt)?;

        embedding_gate::classify_embedding_with_engine(
            engine,
            &signal_embedding,
            &draft_anchor_embedding,
            &full_anchor_embedding,
        )
    } else {
        None
    };

    let decision = policy::decide(&features, embedding_hint);
    progress::emit(cb, ProgressStage::Dispatch);

    // Merge related chunks by directory scope to reduce fanout call count.
    let raw_count = raw_chunks.len();
    let chunks = collect::merge_by_scope(raw_chunks, MERGE_MAX_TOKENS);
    if chunks.len() != raw_count {
        progress::emit(cb, ProgressStage::Merging { from: raw_count, to: chunks.len() });
    }

    // DraftOnly fast path: skip the reduce inference call.
    if decision.route == DispatchRoute::DraftOnly {
        let partials = fanout::analyze_chunks(engine, &chunks)?;
        let report = reduce::synthesize_draft_report(&partials, &decision, &stats);
        progress::emit(cb, ProgressStage::DraftSynthesis);
        validate::validate(&report)?;
        return Ok(report);
    }

    // Small chunk count: skip per-chunk fanout, do a single reduce-style call.
    // For ≤3 merged chunks the overhead of N+1 calls isn't justified.
    if chunks.len() <= SMALL_CHUNK_THRESHOLD {
        let partials = fanout::analyze_chunks(engine, &chunks)?;
        let report = reduce::reduce(engine, &partials, &decision, &stats)?;
        validate::validate(&report)?;
        return Ok(report);
    }

    let partials = fanout::analyze_chunks(engine, &chunks)?;
    let report = reduce::reduce(engine, &partials, &decision, &stats)?;
    validate::validate(&report)?;

    Ok(report)
}

fn resolve_anchor_cache(options: &AnalyzeOptions) -> Option<AnchorEmbeddingCache> {
    if let Some(dir) = options.anchor_cache_dir.as_ref() {
        return Some(anchor_embeddings::anchor_cache_from_path(dir));
    }
    anchor_embeddings::anchor_cache_from_env()
}

fn to_stats(features: &DiffFeatures) -> DiffStats {
    DiffStats {
        files_changed: features.files_changed,
        lines_changed: features.lines_changed,
        hunks: features.hunks,
        binary_files: features.binary_files,
    }
}
