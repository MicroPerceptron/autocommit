use std::path::{Path, PathBuf};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use autocommit_core::CoreError;
use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::llm::{grammar, prompts};
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};
use llama_sys::ffi;
use serde::Deserialize;

use crate::context::RuntimeContext;
use crate::context_handle::ContextHandle;
use crate::embed::{EMBEDDING_MODEL_ENV, FALLBACK_MODEL_ENV, resolve_embedding_model_path};
use crate::error::RuntimeError;
use crate::model_handle::ModelHandle;

static BACKEND_REFCOUNT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn silent_llama_log_callback(
    _level: ffi::ggml_log_level,
    _text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
}

fn llama_logs_enabled() -> bool {
    std::env::var("AUTOCOMMIT_LLAMA_LOG")
        .ok()
        .as_deref()
        .map(|v| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[derive(Debug)]
struct BackendGuard;

impl BackendGuard {
    fn acquire() -> Self {
        if BACKEND_REFCOUNT.fetch_add(1, Ordering::SeqCst) == 0 {
            unsafe {
                // SAFETY: loads available ggml backends once per process startup.
                ffi::ggml_backend_load_all();
                // SAFETY: callback is process-global for llama logging and callback is static.
                if llama_logs_enabled() {
                    ffi::llama_log_set(None, std::ptr::null_mut());
                } else {
                    ffi::llama_log_set(Some(silent_llama_log_callback), std::ptr::null_mut());
                }
                // SAFETY: llama backend init is process-global and intended to be called before runtime usage.
                ffi::llama_backend_init();
            };
        }
        Self
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        if BACKEND_REFCOUNT.fetch_sub(1, Ordering::SeqCst) == 1 {
            // SAFETY: backend free pairs with backend init once global users are drained.
            unsafe { ffi::llama_backend_free() };
        }
    }
}

#[derive(Debug)]
struct LoadedRuntime {
    generation_ctx: Option<ContextHandle>,
    embedding_ctx: Option<ContextHandle>,
    model: Arc<ModelHandle>,
    cpu_only: bool,
}

impl LoadedRuntime {
    fn load(model_path: &Path, profile: &str) -> Result<Self, RuntimeError> {
        let cpu_only = profile.eq_ignore_ascii_case("cpu");
        let model = Arc::new(ModelHandle::load(model_path, cpu_only)?);

        Ok(Self {
            generation_ctx: None,
            embedding_ctx: None,
            model,
            cpu_only,
        })
    }

    fn generation_ctx(&mut self) -> Result<&mut ContextHandle, RuntimeError> {
        if self.generation_ctx.is_none() {
            self.generation_ctx = Some(ContextHandle::new_generation(
                Arc::clone(&self.model),
                self.cpu_only,
            )?);
        }

        Ok(self
            .generation_ctx
            .as_mut()
            .expect("generation context just initialized"))
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        if self.embedding_ctx.is_none() {
            self.embedding_ctx = Some(ContextHandle::new_embedding(
                Arc::clone(&self.model),
                self.cpu_only,
            )?);
        }

        self.embedding_ctx
            .as_mut()
            .expect("embedding context just initialized")
            .embed(text)
    }

    fn analyze_chunks(
        &mut self,
        runtime_context: &RuntimeContext,
        chunks: &[DiffChunk],
    ) -> Result<Vec<PartialReport>, RuntimeError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let model_budget = if self.model.has_gpu_devices() {
            chunks.len().min(24)
        } else {
            0
        };
        let model_indices: HashSet<usize> = sampled_indices(chunks.len(), model_budget).into_iter().collect();

        let model = Arc::clone(&self.model);
        let prompts = chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                if !model_indices.contains(&idx) {
                    return None;
                }
                let prompt = build_analyze_prompt_capped(chunk);
                Some(model
                    .apply_chat_template(Some(prompts::SYSTEM_PROMPT), &prompt)
                    .unwrap_or_else(|| {
                        format!(
                            "System:\n{}\n\nUser:\n{}\n\nAssistant:\n",
                            prompts::SYSTEM_PROMPT,
                            prompt
                        )
                    }))
            })
            .collect::<Vec<_>>();

        let generation_ctx = self.generation_ctx()?;
        let mut out = Vec::with_capacity(chunks.len());

        for (ordinal, chunk) in chunks.iter().enumerate() {
            let partial = if let Some(prompt) = prompts[ordinal].as_ref() {
                let parsed =
                    generate_analyze_output(generation_ctx, prompt, analyze_token_budget(chunk));
                match parsed {
                    Ok(model_output) => {
                        partial_from_model_chunk(runtime_context, ordinal, chunk, model_output)
                    }
                    Err(err) => partial_from_chunk_with_error(runtime_context, ordinal, chunk, &err),
                }
            } else {
                partial_from_chunk(runtime_context, ordinal, chunk)
            };
            out.push(partial);
        }

        Ok(out)
    }

    fn generate_reduce_json(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<ReduceModelOutput, RuntimeError> {
        let prompt = self
            .model
            .apply_chat_template(Some(prompts::SYSTEM_PROMPT), prompt)
            .unwrap_or_else(|| {
                format!(
                    "System:\n{}\n\nUser:\n{}\n\nAssistant:\n",
                    prompts::SYSTEM_PROMPT,
                    prompt
                )
            });
        let generation_ctx = self.generation_ctx()?;

        let raw = generation_ctx.generate_text(&prompt, Some(grammar::REDUCE_GBNF), max_tokens);
        match raw {
            Ok(raw) => match parse_reduce_output(&raw) {
                Ok(parsed) => Ok(parsed),
                Err(parse_err) => {
                    let retry_raw = generation_ctx.generate_text(
                        &prompt,
                        None,
                        max_tokens.saturating_add(48),
                    )?;
                    parse_reduce_output(&retry_raw).map_err(|retry_err| {
                        RuntimeError::Inference(format!(
                            "reduce parse failed with and without grammar: primary={parse_err}; retry={retry_err}"
                        ))
                    })
                }
            },
            Err(err) => {
                if err
                    .to_string()
                    .contains("grammar rejected all sampled candidates")
                {
                    let retry_raw = generation_ctx.generate_text(
                        &prompt,
                        None,
                        max_tokens.saturating_add(48),
                    )?;
                    parse_reduce_output(&retry_raw).map_err(|retry_err| {
                        RuntimeError::Inference(format!(
                            "reduce grammar pass failed ({err}); unconstrained retry parse failed ({retry_err})"
                        ))
                    })
                } else {
                    Err(err)
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    _backend: Arc<BackendGuard>,
    context: RuntimeContext,
    runtime_model_path: Option<PathBuf>,
    runtime: Mutex<Option<LoadedRuntime>>,
}

impl Engine {
    pub fn new(profile: &str) -> Result<Self, RuntimeError> {
        let backend = Arc::new(BackendGuard::acquire());
        Ok(Self {
            _backend: backend,
            context: RuntimeContext::new(profile),
            runtime_model_path: resolve_embedding_model_path(),
            runtime: Mutex::new(None),
        })
    }

    pub fn supports_gpu_offload() -> bool {
        // SAFETY: pure FFI query with no arguments.
        unsafe { ffi::llama_supports_gpu_offload() }
    }

    pub fn backend_refcount() -> usize {
        BACKEND_REFCOUNT.load(Ordering::SeqCst)
    }

    fn with_runtime<T>(
        &self,
        f: impl FnOnce(&mut LoadedRuntime) -> Result<T, RuntimeError>,
    ) -> Result<T, RuntimeError> {
        let model_path = self.runtime_model_path.as_deref().ok_or_else(|| {
            RuntimeError::Inference(format!(
                "runtime model path is not configured; set {EMBEDDING_MODEL_ENV} or {FALLBACK_MODEL_ENV}"
            ))
        })?;

        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| RuntimeError::Inference("runtime lock poisoned".to_string()))?;

        if guard.is_none() {
            *guard = Some(LoadedRuntime::load(model_path, &self.context.profile)?);
        }

        f(guard.as_mut().expect("runtime just initialized"))
    }

    fn embed_with_runtime(&self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        self.with_runtime(|runtime| runtime.embed(text))
    }

    fn analyze_chunks_with_runtime(
        &self,
        chunks: &[DiffChunk],
    ) -> Result<Vec<PartialReport>, RuntimeError> {
        self.with_runtime(|runtime| runtime.analyze_chunks(&self.context, chunks))
    }

    fn reduce_output_with_runtime(
        &self,
        plan: &ReducePromptPlan,
    ) -> Result<ReduceModelOutput, RuntimeError> {
        self.with_runtime(|runtime| {
            if !runtime.model.has_gpu_devices() {
                return Err(RuntimeError::Inference(
                    "skipping model reducer because no usable GPU backend was detected"
                        .to_string(),
                ));
            }
            runtime.generate_reduce_json(&plan.prompt, plan.max_tokens)
        })
    }
}

impl LlmEngine for Engine {
    fn analyze_chunk(&self, chunk: &DiffChunk) -> Result<PartialReport, CoreError> {
        self.analyze_chunks_with_runtime(std::slice::from_ref(chunk))
            .map_err(|err| CoreError::Engine(err.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| CoreError::Engine("missing single-chunk inference output".to_string()))
    }

    fn analyze_chunks_batched(
        &self,
        chunks: &[DiffChunk],
    ) -> Option<Result<Vec<PartialReport>, CoreError>> {
        Some(
            self.analyze_chunks_with_runtime(chunks)
                .map_err(|err| CoreError::Engine(err.to_string())),
        )
    }

    fn reduce_report(
        &self,
        partials: &[PartialReport],
        decision: &DispatchDecision,
        stats: &DiffStats,
    ) -> Result<AnalysisReport, CoreError> {
        let mut items = Vec::new();
        for partial in partials {
            items.extend(partial.items.clone());
        }

        let reduce_plan = build_reduce_prompt_plan(partials, decision, stats);
        let generated_result = self.reduce_output_with_runtime(&reduce_plan);
        let generated = generated_result.as_ref().ok();

        let default_commit_message = {
            let commit_type = dominant_type_tag(&items).map_or("chore", type_tag_prefix);
            format!(
                "{commit_type}(autocommit): synthesize {} chunk analyses",
                partials.len()
            )
        };

        let commit_message_raw = generated
            .as_ref()
            .map(|g| g.commit_message.clone())
            .unwrap_or_else(|| default_commit_message.clone());

        let commit_message = normalize_commit_message(&commit_message_raw)
            .unwrap_or_else(|| default_commit_message.clone());

        let summary = generated
            .as_ref()
            .map(|g| g.summary.clone())
            .unwrap_or_else(|| {
                format!(
                    "Reduced {} chunk analyses via adaptive reducer",
                    partials.len()
                )
            });

        let risk_level = generated
            .as_ref()
            .map(|g| g.risk_level.clone())
            .unwrap_or_else(|| {
                if stats.lines_changed > 400 {
                    "medium".to_string()
                } else {
                    "low".to_string()
                }
            });

        let mut risk_notes = generated
            .as_ref()
            .map(|g| g.risk_notes.clone())
            .unwrap_or_else(|| {
                let mut notes = vec!["fallback_reduce_generation".to_string()];
                if let Err(err) = &generated_result {
                    notes.push(format!("reduce_error:{}", compact_error_note(&err.to_string())));
                }
                notes
            });
        if reduce_plan.sampled < reduce_plan.total {
            risk_notes.push(format!(
                "reduce_sampling:{}/{}",
                reduce_plan.sampled, reduce_plan.total
            ));
        }
        risk_notes.push(format!("dispatch:{:?}", decision.route));

        Ok(AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message,
            summary,
            items,
            risk: RiskReport {
                level: risk_level,
                notes: risk_notes,
            },
            stats: stats.clone(),
            dispatch: decision.clone(),
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        self.embed_with_runtime(text)
            .map_err(|err| CoreError::Engine(err.to_string()))
    }
}

#[derive(Debug, Deserialize)]
struct ReduceModelOutput {
    commit_message: String,
    summary: String,
    risk_level: String,
    #[serde(default)]
    risk_notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AnalyzeModelOutput {
    summary: String,
    bucket: ChangeBucket,
    type_tag: TypeTag,
    title: String,
    intent: String,
}

#[derive(Debug, Clone)]
struct ReducePromptPlan {
    prompt: String,
    max_tokens: usize,
    sampled: usize,
    total: usize,
}

fn build_reduce_prompt_plan(
    partials: &[PartialReport],
    decision: &DispatchDecision,
    stats: &DiffStats,
) -> ReducePromptPlan {
    let total = partials.len();
    let max_tokens = reduce_token_budget(stats, total);
    let max_items = reduce_item_budget(stats, total);
    let max_prompt_chars = reduce_char_budget(stats, total);

    let indices = sampled_indices(total, max_items);
    let mut prompt = prompts::build_reduce_prompt(total);
    prompt.push_str("\nContext:\n");
    prompt.push_str(&format!(
        "- files_changed={}\n- lines_changed={}\n- hunks={}\n- route={:?}\n- reasons={}\n",
        stats.files_changed,
        stats.lines_changed,
        stats.hunks,
        decision.route,
        decision.reason_codes.join(",")
    ));
    prompt.push_str("Representative items:\n");

    let mut sampled = 0usize;
    for idx in indices {
        if let Some(partial) = partials.get(idx) {
            if prompt.len() >= max_prompt_chars {
                break;
            }

            sampled += 1;
            prompt.push_str("- ");
            prompt.push_str(&format_partial_for_reduce(partial));
            prompt.push('\n');
        }
    }

    if sampled < total {
        prompt.push_str(&format!(
            "- NOTE: {} additional items omitted for brevity. Infer the dominant theme.\n",
            total - sampled
        ));
    }

    ReducePromptPlan {
        prompt,
        max_tokens,
        sampled,
        total,
    }
}

fn reduce_token_budget(stats: &DiffStats, total: usize) -> usize {
    if stats.lines_changed > 1_500 || total > 24 {
        128
    } else if stats.lines_changed > 900 || total > 12 {
        160
    } else {
        192
    }
}

fn reduce_item_budget(stats: &DiffStats, total: usize) -> usize {
    if total <= 6 {
        total
    } else if stats.lines_changed > 1_500 || total > 24 {
        8
    } else if stats.lines_changed > 900 || total > 12 {
        10
    } else {
        14
    }
}

fn reduce_char_budget(stats: &DiffStats, total: usize) -> usize {
    if stats.lines_changed > 1_500 || total > 24 {
        2_500
    } else if stats.lines_changed > 900 || total > 12 {
        3_000
    } else {
        4_500
    }
}

fn sampled_indices(total: usize, max_items: usize) -> Vec<usize> {
    if total == 0 || max_items == 0 {
        return Vec::new();
    }

    if total <= max_items {
        return (0..total).collect();
    }

    let mut out = Vec::with_capacity(max_items);
    out.push(0);
    if max_items > 1 {
        out.push(total - 1);
    }

    let remaining = max_items.saturating_sub(out.len());
    if remaining > 0 {
        let step = (total - 1) as f32 / (remaining + 1) as f32;
        for n in 1..=remaining {
            let idx = ((n as f32) * step).round() as usize;
            if idx < total && !out.contains(&idx) {
                out.push(idx);
            }
        }
    }

    out.sort_unstable();
    out.truncate(max_items);
    out
}

fn format_partial_for_reduce(partial: &PartialReport) -> String {
    if let Some(item) = partial.items.first() {
        let file_paths = item
            .files
            .iter()
            .take(2)
            .map(|f| f.path.as_str())
            .collect::<Vec<_>>()
            .join("|");
        format!(
            "bucket={:?} type={:?} title={} files={} confidence={:.2} summary={}",
            item.bucket, item.type_tag, item.title, file_paths, item.confidence, partial.summary
        )
    } else {
        partial.summary.clone()
    }
}

fn parse_reduce_output(raw: &str) -> Result<ReduceModelOutput, RuntimeError> {
    let mut last: Option<ReduceModelOutput> = None;

    for (idx, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }

        let segment = &raw[idx..];
        let mut de = serde_json::Deserializer::from_str(segment);
        if let Ok(parsed) = ReduceModelOutput::deserialize(&mut de) {
            last = Some(parsed);
        }
    }

    let mut output = last.ok_or_else(|| {
        RuntimeError::Inference(format!(
            "failed to parse reduce JSON output from model: {}",
            raw.trim()
        ))
    })?;

    output.commit_message = output.commit_message.trim().to_string();
    output.summary = output.summary.trim().to_string();
    output.risk_level = output.risk_level.trim().to_ascii_lowercase();

    if output.commit_message.is_empty() {
        return Err(RuntimeError::Inference(
            "reduce output commit_message is empty".to_string(),
        ));
    }

    if output.summary.is_empty() {
        return Err(RuntimeError::Inference(
            "reduce output summary is empty".to_string(),
        ));
    }

    if !matches!(output.risk_level.as_str(), "low" | "medium" | "high") {
        output.risk_level = "medium".to_string();
    }

    output.risk_notes.retain(|note| !note.trim().is_empty());
    Ok(output)
}

fn generate_analyze_output(
    generation_ctx: &mut ContextHandle,
    prompt: &str,
    max_tokens: usize,
) -> Result<AnalyzeModelOutput, RuntimeError> {
    let raw = generation_ctx.generate_text(prompt, Some(grammar::ANALYZE_GBNF), max_tokens);
    match raw {
        Ok(raw) => parse_analyze_output(&raw),
        Err(err) => {
            if err
                .to_string()
                .contains("grammar rejected all sampled candidates")
            {
                let retry_raw =
                    generation_ctx.generate_text(prompt, None, max_tokens.saturating_add(64))?;
                parse_analyze_output(&retry_raw).map_err(|retry_err| {
                    RuntimeError::Inference(format!(
                        "analyze grammar pass failed ({err}); unconstrained retry parse failed ({retry_err})"
                    ))
                })
            } else {
                Err(err)
            }
        }
    }
}

fn parse_analyze_output(raw: &str) -> Result<AnalyzeModelOutput, RuntimeError> {
    let mut last: Option<AnalyzeModelOutput> = None;

    for (idx, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }

        let segment = &raw[idx..];
        let mut de = serde_json::Deserializer::from_str(segment);
        if let Ok(parsed) = AnalyzeModelOutput::deserialize(&mut de) {
            last = Some(parsed);
        }
    }

    let mut output = last.ok_or_else(|| {
        RuntimeError::Inference(format!(
            "failed to parse analyze JSON output from model: {}",
            raw.trim()
        ))
    })?;

    output.summary = output.summary.trim().to_string();
    output.title = output.title.trim().to_string();
    output.intent = output.intent.trim().to_string();
    if output.summary.is_empty() || output.title.is_empty() || output.intent.is_empty() {
        return Err(RuntimeError::Inference(
            "analyze output has empty required field".to_string(),
        ));
    }

    Ok(output)
}

fn compact_error_note(raw: &str) -> String {
    let squashed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let max_chars = 1000usize;
    if squashed.chars().count() <= max_chars {
        return squashed;
    }

    let mut out = String::with_capacity(max_chars + 1);
    for ch in squashed.chars().take(max_chars) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn analyze_token_budget(chunk: &DiffChunk) -> usize {
    if chunk.estimated_tokens > 3_000 {
        80
    } else if chunk.estimated_tokens > 1_500 {
        96
    } else {
        112
    }
}

fn build_analyze_prompt_capped(chunk: &DiffChunk) -> String {
    let max_chars = if chunk.estimated_tokens > 3_000 {
        3_000
    } else if chunk.estimated_tokens > 1_500 {
        4_000
    } else {
        6_000
    };

    if chunk.text.len() <= max_chars {
        return prompts::build_analyze_prompt(chunk);
    }

    let mut truncated = String::with_capacity(max_chars + 128);
    for ch in chunk.text.chars().take(max_chars) {
        truncated.push(ch);
    }
    truncated.push_str("\n... [diff truncated for budget]");

    format!(
        "Task: Analyze one diff chunk.\n\
Return ONLY JSON with keys: summary, bucket, type_tag, title, intent.\n\
Allowed bucket values: Feature, Patch, Addition, Other.\n\
Allowed type_tag values: Feat, Fix, Refactor, Docs, Test, Chore, Perf, Style, Mixed.\n\
Path: {}\n\
Diff:\n```diff\n{}\n```",
        chunk.path, truncated
    )
}

fn sanitize_sentence(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn partial_from_model_chunk(
    runtime_context: &RuntimeContext,
    ordinal: usize,
    chunk: &DiffChunk,
    model: AnalyzeModelOutput,
) -> PartialReport {
    let (added, removed) = diff_line_counts(&chunk.text);
    let confidence = heuristic_confidence(chunk, added, removed);
    let summary = sanitize_sentence(&model.summary);
    let title = sanitize_sentence(&model.title);
    let intent = sanitize_sentence(&model.intent);

    PartialReport {
        summary: if summary.is_empty() {
            format!(
                "{} (+{}, -{}) analyzed via sequence {}",
                chunk.path, added, removed, ordinal
            )
        } else {
            summary
        },
        items: vec![ChangeItem {
            id: format!(
                "rt-{}-{}-{}",
                runtime_context.id,
                ordinal,
                chunk.path.replace('/', "_")
            ),
            bucket: model.bucket,
            type_tag: model.type_tag,
            title: if title.is_empty() {
                format!("update {}", chunk.path)
            } else {
                title
            },
            intent: if intent.is_empty() {
                format!("update {}", chunk.path)
            } else {
                intent
            },
            files: vec![FileRef {
                path: chunk.path.clone(),
                status: FileStatus::Modified,
                ranges: chunk.ranges.clone(),
            }],
            confidence,
        }],
    }
}

fn partial_from_chunk_with_error(
    runtime_context: &RuntimeContext,
    ordinal: usize,
    chunk: &DiffChunk,
    err: &RuntimeError,
) -> PartialReport {
    let mut partial = partial_from_chunk(runtime_context, ordinal, chunk);
    partial.summary = format!(
        "{}; fallback heuristic due to analyze_error:{}",
        partial.summary,
        compact_error_note(&err.to_string())
    );
    partial
}

fn partial_from_chunk(
    runtime_context: &RuntimeContext,
    ordinal: usize,
    chunk: &DiffChunk,
) -> PartialReport {
    let (bucket, type_tag, intent_label) = classify_chunk(chunk);
    let (added, removed) = diff_line_counts(&chunk.text);
    let confidence = heuristic_confidence(chunk, added, removed);

    PartialReport {
        summary: format!(
            "{} (+{}, -{}) analyzed via batched sequence {}",
            chunk.path, added, removed, ordinal
        ),
        items: vec![ChangeItem {
            id: format!(
                "rt-{}-{}-{}",
                runtime_context.id,
                ordinal,
                chunk.path.replace('/', "_")
            ),
            bucket,
            type_tag,
            title: format!("{} {}", intent_label, chunk.path),
            intent: format!(
                "{} (tokens≈{}, signal={:.3})",
                intent_label, chunk.estimated_tokens, confidence
            ),
            files: vec![FileRef {
                path: chunk.path.clone(),
                status: FileStatus::Modified,
                ranges: chunk.ranges.clone(),
            }],
            confidence,
        }],
    }
}

fn heuristic_confidence(chunk: &DiffChunk, added: usize, removed: usize) -> f32 {
    let churn = added.saturating_add(removed);
    let range_factor = (chunk.ranges.len().min(6) as f32) * 0.03;
    let churn_factor = (churn.min(200) as f32) / 1000.0;
    (0.62 + range_factor + churn_factor).clamp(0.55, 0.9)
}

fn classify_chunk(chunk: &DiffChunk) -> (ChangeBucket, TypeTag, &'static str) {
    let path = chunk.path.to_ascii_lowercase();
    let (added, removed) = diff_line_counts(&chunk.text);

    if path.contains("test") || path.contains("spec") {
        return (ChangeBucket::Patch, TypeTag::Test, "update tests in");
    }

    if added > removed.saturating_mul(2) && added >= 8 {
        return (
            ChangeBucket::Feature,
            TypeTag::Feat,
            "introduce functionality in",
        );
    }

    if removed > added.saturating_mul(2) && removed >= 8 {
        return (
            ChangeBucket::Patch,
            TypeTag::Refactor,
            "refactor implementation in",
        );
    }

    if path.contains("readme") || path.ends_with(".md") {
        return (ChangeBucket::Addition, TypeTag::Docs, "document changes in");
    }

    (ChangeBucket::Patch, TypeTag::Fix, "adjust behavior in")
}

fn diff_line_counts(text: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;

    for line in text.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }

        if line.starts_with('+') {
            added += 1;
        } else if line.starts_with('-') {
            removed += 1;
        }
    }

    (added, removed)
}

fn dominant_type_tag(items: &[ChangeItem]) -> Option<TypeTag> {
    let mut feat = 0usize;
    let mut fix = 0usize;
    let mut refactor = 0usize;
    let mut docs = 0usize;
    let mut test = 0usize;
    let mut chore = 0usize;

    for item in items {
        match item.type_tag {
            TypeTag::Feat => feat += 1,
            TypeTag::Fix => fix += 1,
            TypeTag::Refactor => refactor += 1,
            TypeTag::Docs => docs += 1,
            TypeTag::Test => test += 1,
            TypeTag::Chore => chore += 1,
            TypeTag::Perf | TypeTag::Style | TypeTag::Mixed => chore += 1,
        }
    }

    let ranked = [
        (feat, TypeTag::Feat),
        (fix, TypeTag::Fix),
        (refactor, TypeTag::Refactor),
        (docs, TypeTag::Docs),
        (test, TypeTag::Test),
        (chore, TypeTag::Chore),
    ];

    ranked
        .into_iter()
        .max_by_key(|(count, _)| *count)
        .and_then(|(count, tag)| if count == 0 { None } else { Some(tag) })
}

fn type_tag_prefix(tag: TypeTag) -> &'static str {
    match tag {
        TypeTag::Feat => "feat",
        TypeTag::Fix => "fix",
        TypeTag::Refactor => "refactor",
        TypeTag::Docs => "docs",
        TypeTag::Test => "test",
        TypeTag::Chore | TypeTag::Perf | TypeTag::Style | TypeTag::Mixed => "chore",
    }
}

fn normalize_commit_message(raw: &str) -> Option<String> {
    let header = raw.lines().next()?.trim();
    if header.is_empty() {
        return None;
    }

    if let Some((kind, scope, desc)) = parse_conventional_commit(header) {
        let mut out = kind.to_string();
        if let Some(scope) = scope {
            out.push('(');
            out.push_str(scope);
            out.push(')');
        }
        out.push_str(": ");
        out.push_str(desc);
        return Some(out);
    }

    if let Some((kind, desc)) = parse_type_prefixed_header(header) {
        return Some(format!("{kind}(autocommit): {desc}"));
    }

    None
}

fn parse_conventional_commit(header: &str) -> Option<(&'static str, Option<&str>, &str)> {
    let (head, desc) = header.split_once(':')?;
    let desc = desc.trim();
    if desc.is_empty() {
        return None;
    }

    let mut head = head.trim();
    if head.ends_with('!') {
        head = &head[..head.len().saturating_sub(1)];
    }

    if let Some(open) = head.find('(') {
        if !head.ends_with(')') || open == 0 {
            return None;
        }
        let kind = canonical_commit_type(&head[..open])?;
        let scope = head[open + 1..head.len() - 1].trim();
        if scope.is_empty() {
            return None;
        }
        return Some((kind, Some(scope), desc));
    }

    let kind = canonical_commit_type(head)?;
    Some((kind, None, desc))
}

fn parse_type_prefixed_header(header: &str) -> Option<(&'static str, &str)> {
    let lower = header.to_ascii_lowercase();
    for (alias, canonical) in [
        ("feat", "feat"),
        ("feature", "feat"),
        ("fix", "fix"),
        ("refactor", "refactor"),
        ("docs", "docs"),
        ("doc", "docs"),
        ("test", "test"),
        ("chore", "chore"),
        ("perf", "perf"),
        ("style", "style"),
    ] {
        if !lower.starts_with(alias) {
            continue;
        }
        let rest = header[alias.len()..].trim_start_matches([':', '-', ' ']);
        let rest = rest.trim();
        if !rest.is_empty() {
            return Some((canonical, rest));
        }
    }

    None
}

fn canonical_commit_type(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "feat" | "feature" => Some("feat"),
        "fix" => Some("fix"),
        "refactor" => Some("refactor"),
        "docs" | "doc" => Some("docs"),
        "test" => Some("test"),
        "chore" => Some("chore"),
        "perf" => Some("perf"),
        "style" => Some("style"),
        _ => None,
    }
}
