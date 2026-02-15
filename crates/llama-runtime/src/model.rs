use std::path::{Path, PathBuf};
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
    generation_state_path: Option<PathBuf>,
}

impl LoadedRuntime {
    fn load(
        model_path: &Path,
        profile: &str,
        generation_state_path: Option<PathBuf>,
    ) -> Result<Self, RuntimeError> {
        let cpu_only = profile.eq_ignore_ascii_case("cpu");
        let model = Arc::new(ModelHandle::load(model_path, cpu_only)?);

        Ok(Self {
            generation_ctx: None,
            embedding_ctx: None,
            model,
            cpu_only,
            generation_state_path,
        })
    }

    fn generation_ctx(&mut self) -> Result<&mut ContextHandle, RuntimeError> {
        if self.generation_ctx.is_none() {
            let mut ctx = ContextHandle::new_generation(Arc::clone(&self.model), self.cpu_only)?;

            if let Some(state_path) = self.generation_state_path.as_deref() {
                if state_path.is_file() {
                    match ctx.load_state_file(state_path) {
                        Ok(tokens) => ctx.set_session_tokens(tokens),
                        Err(err) => {
                            if llama_logs_enabled() {
                                eprintln!(
                                    "autocommit warning: failed to load generation state from {}: {err}",
                                    state_path.display()
                                );
                            }
                        }
                    }
                }
            }

            self.generation_ctx = Some(ctx);
        }

        Ok(self
            .generation_ctx
            .as_mut()
            .expect("generation context just initialized"))
    }

    fn warmup_generation_cache(&mut self) -> Result<(), RuntimeError> {
        let state_path = self.generation_state_path.clone().ok_or_else(|| {
            RuntimeError::State("generation cache path is not configured for warmup".to_string())
        })?;

        let generation_ctx = self.generation_ctx()?;
        generation_ctx.warmup()?;
        generation_ctx.save_state_file(&state_path, &[])
    }

    fn persist_generation_state_if_needed(generation_ctx: &mut ContextHandle, state_path: &Path) {
        let tokens = generation_ctx.session_tokens().to_vec();
        if tokens.is_empty() {
            return;
        }

        if let Err(err) = generation_ctx.save_state_file(state_path, &tokens) {
            if llama_logs_enabled() {
                eprintln!(
                    "autocommit warning: failed to save generation state to {}: {err}",
                    state_path.display()
                );
            }
        }
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

        let model = Arc::clone(&self.model);
        let prompts = chunks
            .iter()
            .map(|chunk| {
                let prompt = build_analyze_prompt_capped(chunk);
                model
                    .apply_chat_template(Some(prompts::SYSTEM_PROMPT), &prompt)
                    .unwrap_or_else(|| {
                        format!(
                            "System:\n{}\n\nUser:\n{}\n\nAssistant:\n",
                            prompts::SYSTEM_PROMPT,
                            prompt
                        )
                    })
            })
            .collect::<Vec<_>>();
        let budgets = chunks.iter().map(analyze_token_budget).collect::<Vec<_>>();
        let generation_state_path = self.generation_state_path.clone();

        let generation_ctx = self.generation_ctx()?;
        let mut out: Vec<PartialReport> = chunks
            .iter()
            .enumerate()
            .map(|(ordinal, chunk)| partial_from_chunk(runtime_context, ordinal, chunk))
            .collect();

        let window = generation_ctx.max_sequences().max(1);
        let mut start = 0usize;
        while start < prompts.len() {
            let end = (start + window).min(prompts.len());
            let batch_prompts = &prompts[start..end];
            let batch_budgets = &budgets[start..end];

            let batch_outputs = generation_ctx.generate_texts_with_budgets(
                batch_prompts,
                Some(grammar::ANALYZE_GBNF),
                batch_budgets,
            );

            match batch_outputs {
                Ok(outputs) => {
                    for (idx, raw_result) in outputs.into_iter().enumerate() {
                        let ordinal = start + idx;
                        let chunk = &chunks[ordinal];
                        out[ordinal] = match raw_result {
                            Ok(raw) => match parse_analyze_output(&raw) {
                                Ok(model_output) => partial_from_model_chunk(
                                    runtime_context,
                                    ordinal,
                                    chunk,
                                    model_output,
                                ),
                                Err(parse_err) => {
                                    let retry = generate_analyze_output(
                                        generation_ctx,
                                        &prompts[ordinal],
                                        budgets[ordinal],
                                    );
                                    match retry {
                                        Ok(model_output) => partial_from_model_chunk(
                                            runtime_context,
                                            ordinal,
                                            chunk,
                                            model_output,
                                        ),
                                        Err(retry_err) => {
                                            let combined = RuntimeError::Inference(format!(
                                                "batched parse failed ({parse_err}); retry failed ({retry_err})"
                                            ));
                                            partial_from_chunk_with_error(
                                                runtime_context,
                                                ordinal,
                                                chunk,
                                                &combined,
                                            )
                                        }
                                    }
                                }
                            },
                            Err(batch_err) => {
                                let retry = generate_analyze_output(
                                    generation_ctx,
                                    &prompts[ordinal],
                                    budgets[ordinal],
                                );
                                match retry {
                                    Ok(model_output) => partial_from_model_chunk(
                                        runtime_context,
                                        ordinal,
                                        chunk,
                                        model_output,
                                    ),
                                    Err(retry_err) => {
                                        let combined = RuntimeError::Inference(format!(
                                            "batched generation failed ({batch_err}); retry failed ({retry_err})"
                                        ));
                                        partial_from_chunk_with_error(
                                            runtime_context,
                                            ordinal,
                                            chunk,
                                            &combined,
                                        )
                                    }
                                }
                            }
                        };
                    }
                }
                Err(batch_err) => {
                    for idx in start..end {
                        let chunk = &chunks[idx];
                        let parsed =
                            generate_analyze_output(generation_ctx, &prompts[idx], budgets[idx]);
                        out[idx] = match parsed {
                            Ok(model_output) => {
                                partial_from_model_chunk(runtime_context, idx, chunk, model_output)
                            }
                            Err(err) => {
                                let combined = RuntimeError::Inference(format!(
                                    "batched analyze failed ({batch_err}); fallback failed ({err})"
                                ));
                                partial_from_chunk_with_error(
                                    runtime_context,
                                    idx,
                                    chunk,
                                    &combined,
                                )
                            }
                        };
                    }
                }
            }

            start = end;
        }

        if let Some(path) = generation_state_path.as_deref() {
            Self::persist_generation_state_if_needed(generation_ctx, path);
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
        let generation_state_path = self.generation_state_path.clone();
        let generation_ctx = self.generation_ctx()?;

        let raw = generation_ctx.generate_text(&prompt, Some(grammar::REDUCE_GBNF), max_tokens);
        let result = match raw {
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
        };

        if let Some(path) = generation_state_path.as_deref() {
            Self::persist_generation_state_if_needed(generation_ctx, path);
        }

        result
    }
}

#[derive(Debug)]
pub struct Engine {
    _backend: Arc<BackendGuard>,
    context: RuntimeContext,
    runtime_model_path: Option<PathBuf>,
    generation_state_path: Option<PathBuf>,
    runtime: Mutex<Option<LoadedRuntime>>,
}

impl Engine {
    pub fn new(profile: &str) -> Result<Self, RuntimeError> {
        Self::new_with_generation_cache(profile, None)
    }

    pub fn new_with_generation_cache(
        profile: &str,
        generation_state_path: Option<PathBuf>,
    ) -> Result<Self, RuntimeError> {
        let backend = Arc::new(BackendGuard::acquire());
        Ok(Self {
            _backend: backend,
            context: RuntimeContext::new(profile),
            runtime_model_path: resolve_embedding_model_path(),
            generation_state_path,
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

    pub fn configured_model_path(&self) -> Option<&Path> {
        self.runtime_model_path.as_deref()
    }

    pub fn warmup_generation_cache(&self) -> Result<(), RuntimeError> {
        self.with_runtime(|runtime| runtime.warmup_generation_cache())
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
            *guard = Some(LoadedRuntime::load(
                model_path,
                &self.context.profile,
                self.generation_state_path.clone(),
            )?);
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
        self.with_runtime(|runtime| runtime.generate_reduce_json(&plan.prompt, plan.max_tokens))
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
        let generated_commit_candidate = generated
            .as_ref()
            .and_then(|g| normalize_commit_message(&g.commit_message));

        let commit_message = synthesize_fallback_commit_message(&items, partials.len());
        let summary = synthesize_fallback_summary(&items, stats);

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
                    notes.push(format!(
                        "reduce_error:{}",
                        compact_error_note(&err.to_string())
                    ));
                }
                notes
            });
        risk_notes.push("commit_source:composed_partials".to_string());
        if generated.is_some() {
            risk_notes.push(if generated_commit_candidate.is_some() {
                "reduce_commit_candidate:usable".to_string()
            } else {
                "reduce_commit_candidate:invalid".to_string()
            });
        }
        let analyze_fallbacks = partials
            .iter()
            .filter(|p| {
                p.summary
                    .contains("fallback heuristic due to analyze_error:")
            })
            .count();
        if analyze_fallbacks > 0 {
            risk_notes.push(format!(
                "analyze_fallback:{}/{}",
                analyze_fallbacks,
                partials.len()
            ));
        } else {
            risk_notes.push(format!("analyze_model:{}", partials.len()));
        }
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
        224
    } else if stats.lines_changed > 900 || total > 12 {
        256
    } else {
        320
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
    let summary = compact_reduce_text(&partial.summary, 220);
    if let Some(item) = partial.items.first() {
        let file_paths = item
            .files
            .iter()
            .take(2)
            .map(|f| f.path.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let title = compact_reduce_text(&item.title, 120);
        format!(
            "bucket={:?} type={:?} title={} files={} confidence={:.2} summary={}",
            item.bucket, item.type_tag, title, file_paths, item.confidence, summary
        )
    } else {
        summary
    }
}

fn compact_reduce_text(value: &str, max_chars: usize) -> String {
    let mut normalized = sanitize_sentence(value);
    if normalized.contains("fallback heuristic due to analyze_error:") {
        normalized = "fallback heuristic used due to analyze error".to_string();
    }
    if normalized.chars().count() <= max_chars {
        return normalized;
    }
    let mut out = String::with_capacity(max_chars + 1);
    for ch in normalized.chars().take(max_chars) {
        out.push(ch);
    }
    out.push('…');
    out
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
        Ok(raw) => match parse_analyze_output(&raw) {
            Ok(parsed) => Ok(parsed),
            Err(parse_err) => {
                let retry_raw =
                    generation_ctx.generate_text(prompt, None, max_tokens.saturating_add(64))?;
                parse_analyze_output(&retry_raw).map_err(|retry_err| {
                    RuntimeError::Inference(format!(
                        "analyze parse failed with and without grammar: primary={parse_err}; retry={retry_err}"
                    ))
                })
            }
        },
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

    if last.is_none() {
        last = salvage_analyze_output(raw).or_else(|| analyze_from_freeform_text(raw));
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
    if output.summary.is_empty() {
        output.summary = "chunk analysis".to_string();
    }
    if output.title.is_empty() {
        output.title = output.summary.clone();
    }
    if output.intent.is_empty() {
        output.intent = output.title.clone();
    }

    Ok(output)
}

fn salvage_analyze_output(raw: &str) -> Option<AnalyzeModelOutput> {
    let summary_raw = extract_json_like_field(raw, "summary");
    let title_raw = extract_json_like_field(raw, "title");
    let intent_raw = extract_json_like_field(raw, "intent");

    if summary_raw.is_none() && title_raw.is_none() && intent_raw.is_none() {
        return None;
    }

    let summary = summary_raw
        .clone()
        .or_else(|| title_raw.clone())
        .or_else(|| intent_raw.clone())
        .unwrap_or_else(|| "chunk analysis".to_string());
    let title = title_raw
        .clone()
        .or_else(|| summary_raw.clone())
        .or_else(|| intent_raw.clone())
        .unwrap_or_else(|| "update chunk".to_string());
    let intent = intent_raw
        .clone()
        .or_else(|| title_raw.clone())
        .or_else(|| summary_raw.clone())
        .unwrap_or_else(|| "summarize chunk intent".to_string());

    let bucket = extract_json_like_field(raw, "bucket")
        .and_then(|v| parse_bucket_label(&v))
        .unwrap_or(ChangeBucket::Patch);
    let type_tag = extract_json_like_field(raw, "type_tag")
        .or_else(|| extract_json_like_field(raw, "type"))
        .and_then(|v| parse_type_tag_label(&v))
        .unwrap_or(TypeTag::Mixed);

    Some(AnalyzeModelOutput {
        summary,
        bucket,
        type_tag,
        title,
        intent,
    })
}

fn analyze_from_freeform_text(raw: &str) -> Option<AnalyzeModelOutput> {
    let cleaned = sanitize_sentence(raw);
    if cleaned.is_empty() {
        return None;
    }

    let summary = truncate_chars(&cleaned, 220);
    let title = truncate_chars(&cleaned, 120);
    let intent = truncate_chars(&cleaned, 160);
    let lowercase = cleaned.to_ascii_lowercase();

    Some(AnalyzeModelOutput {
        summary,
        bucket: infer_bucket_from_text(&lowercase),
        type_tag: infer_type_tag_from_text(&lowercase),
        title,
        intent,
    })
}

fn infer_bucket_from_text(raw: &str) -> ChangeBucket {
    if raw.contains("feature")
        || raw.contains("add ")
        || raw.contains("added ")
        || raw.contains("introduc")
        || raw.contains("new ")
    {
        ChangeBucket::Feature
    } else if raw.contains("doc") || raw.contains("readme") {
        ChangeBucket::Addition
    } else if raw.contains("fix")
        || raw.contains("patch")
        || raw.contains("update")
        || raw.contains("refactor")
    {
        ChangeBucket::Patch
    } else {
        ChangeBucket::Other
    }
}

fn infer_type_tag_from_text(raw: &str) -> TypeTag {
    if raw.contains("refactor") || raw.contains("cleanup") {
        TypeTag::Refactor
    } else if raw.contains("fix") || raw.contains("bug") {
        TypeTag::Fix
    } else if raw.contains("test") {
        TypeTag::Test
    } else if raw.contains("doc") || raw.contains("readme") {
        TypeTag::Docs
    } else if raw.contains("perf") || raw.contains("optim") {
        TypeTag::Perf
    } else if raw.contains("style") || raw.contains("format") {
        TypeTag::Style
    } else if raw.contains("feat")
        || raw.contains("feature")
        || raw.contains("add ")
        || raw.contains("introduc")
    {
        TypeTag::Feat
    } else {
        TypeTag::Mixed
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut out = String::with_capacity(max_chars + 3);
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn parse_bucket_label(value: &str) -> Option<ChangeBucket> {
    match value.trim().to_ascii_lowercase().as_str() {
        "feature" => Some(ChangeBucket::Feature),
        "patch" => Some(ChangeBucket::Patch),
        "addition" => Some(ChangeBucket::Addition),
        "other" => Some(ChangeBucket::Other),
        _ => None,
    }
}

fn parse_type_tag_label(value: &str) -> Option<TypeTag> {
    match value.trim().to_ascii_lowercase().as_str() {
        "feat" => Some(TypeTag::Feat),
        "fix" => Some(TypeTag::Fix),
        "refactor" => Some(TypeTag::Refactor),
        "docs" => Some(TypeTag::Docs),
        "test" => Some(TypeTag::Test),
        "chore" => Some(TypeTag::Chore),
        "perf" => Some(TypeTag::Perf),
        "style" => Some(TypeTag::Style),
        "mixed" => Some(TypeTag::Mixed),
        _ => None,
    }
}

fn extract_json_like_field(raw: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let bytes = raw.as_bytes();
    let mut start = 0usize;

    while let Some(rel) = raw.get(start..)?.find(&needle) {
        let mut i = start + rel + needle.len();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            start = i.min(bytes.len());
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }

        if bytes[i] == b'"' {
            if let Some(end) = find_string_end(bytes, i + 1) {
                if let Some(slice) = raw.get(i..=end) {
                    if let Ok(parsed) = serde_json::from_str::<String>(slice) {
                        return Some(parsed.trim().to_string());
                    }
                }
                if let Some(fallback) = raw.get((i + 1)..end) {
                    return Some(fallback.trim().to_string());
                }
                return None;
            }

            // Truncated quote: recover until next delimiter.
            let end = find_unquoted_value_end(bytes, i + 1);
            if let Some(fallback) = raw.get((i + 1)..end) {
                return Some(fallback.trim().to_string());
            }
            return None;
        }

        let end = find_unquoted_value_end(bytes, i);
        if let Some(slice) = raw.get(i..end) {
            let cleaned = slice.trim().trim_matches('"').trim().to_string();
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }

        start = end.min(bytes.len());
    }

    None
}

fn find_string_end(bytes: &[u8], mut idx: usize) -> Option<usize> {
    while idx < bytes.len() {
        if bytes[idx] == b'"' {
            let mut slash_count = 0usize;
            let mut back = idx;
            while back > 0 && bytes[back - 1] == b'\\' {
                slash_count += 1;
                back -= 1;
            }
            if slash_count % 2 == 0 {
                return Some(idx);
            }
        }
        idx += 1;
    }
    None
}

fn find_unquoted_value_end(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() {
        match bytes[idx] {
            b',' | b'}' | b'\n' | b'\r' => break,
            _ => idx += 1,
        }
    }
    idx
}

fn compact_error_note(raw: &str) -> String {
    let squashed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let max_chars = 240usize;
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
        192
    } else if chunk.estimated_tokens > 1_500 {
        224
    } else {
        256
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

fn mapped_change_item_id(path: &str, ordinal: usize) -> String {
    format!("{path}#chunk-{}", ordinal.saturating_add(1))
}

fn partial_from_model_chunk(
    _runtime_context: &RuntimeContext,
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
            id: mapped_change_item_id(&chunk.path, ordinal),
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
    _runtime_context: &RuntimeContext,
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
            id: mapped_change_item_id(&chunk.path, ordinal),
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

#[derive(Debug)]
struct SubjectCandidate {
    score: f32,
    subject: String,
    type_tag: TypeTag,
    scope: Option<String>,
}

fn synthesize_fallback_commit_message(items: &[ChangeItem], partial_count: usize) -> String {
    if let Some(best) = top_subject_candidates(items, 1).into_iter().next() {
        let commit_type = type_tag_prefix(best.type_tag);
        let scope = best.scope.or_else(|| dominant_scope(items));
        let description = decapitalize_first(&best.subject);

        return match scope {
            Some(scope) => format!("{commit_type}({scope}): {description}"),
            None => format!("{commit_type}: {description}"),
        };
    }

    let commit_type = dominant_type_tag(items).map_or("chore", type_tag_prefix);
    let scope = dominant_scope(items);
    let description = default_commit_subject(commit_type, partial_count);

    match scope {
        Some(scope) => format!("{commit_type}({scope}): {description}"),
        None => format!("{commit_type}: {description}"),
    }
}

fn synthesize_fallback_summary(items: &[ChangeItem], stats: &DiffStats) -> String {
    let file_count = stats.files_changed.max(1);
    let noun = if file_count == 1 { "file" } else { "files" };

    let subjects = top_subject_candidates(items, 2)
        .into_iter()
        .map(|candidate| candidate.subject)
        .collect::<Vec<_>>();
    if subjects.is_empty() {
        return format!("Update project code across {file_count} {noun}.");
    }

    if subjects.len() == 1 {
        return format!(
            "{} across {file_count} {noun}.",
            capitalize_first(&subjects[0])
        );
    }

    format!(
        "{} and {} across {file_count} {noun}.",
        capitalize_first(&subjects[0]),
        subjects[1]
    )
}

fn dominant_scope(items: &[ChangeItem]) -> Option<String> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();

    for item in items {
        for file in &item.files {
            if let Some(scope) = scope_from_path(&file.path) {
                *counts.entry(scope).or_insert(0) += 1;
            }
        }
    }

    let mut best_scope: Option<String> = None;
    let mut best_count = 0usize;
    let mut tie = false;

    for (scope, count) in counts {
        if count > best_count {
            best_scope = Some(scope);
            best_count = count;
            tie = false;
        } else if count == best_count && count != 0 {
            tie = true;
        }
    }

    if tie { None } else { best_scope }
}

fn top_subject_candidates(items: &[ChangeItem], limit: usize) -> Vec<SubjectCandidate> {
    if limit == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for item in items {
        let mut best_for_item: Option<SubjectCandidate> = None;
        let item_scope = item
            .files
            .first()
            .and_then(|file| scope_from_path(&file.path));

        for (raw, bonus) in [(&item.title, 0.05f32), (&item.intent, 0.0f32)] {
            let Some(candidate) = sanitize_commit_subject(raw) else {
                continue;
            };
            if looks_like_reducer_meta(&candidate) {
                continue;
            }

            let score = item.confidence + bonus + subject_quality_bonus(&candidate);
            if best_for_item
                .as_ref()
                .map(|existing| score > existing.score)
                .unwrap_or(true)
            {
                best_for_item = Some(SubjectCandidate {
                    score,
                    subject: candidate,
                    type_tag: item.type_tag.clone(),
                    scope: item_scope.clone(),
                });
            }
        }

        if let Some(candidate) = best_for_item {
            candidates.push(candidate);
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.subject.cmp(&b.subject))
    });

    let mut out: Vec<SubjectCandidate> = Vec::new();
    for candidate in candidates {
        if out
            .iter()
            .any(|existing| existing.subject.eq_ignore_ascii_case(&candidate.subject))
        {
            continue;
        }
        out.push(candidate);
        if out.len() >= limit {
            break;
        }
    }

    out
}

fn scope_from_path(path: &str) -> Option<String> {
    let mut parts = path.split('/').filter(|part| !part.is_empty());
    let first = parts.next()?;

    let raw_scope = match first {
        "crates" | "src" => parts.next().unwrap_or(first),
        "tests" | "test" => "test",
        "docs" => "docs",
        "third_party" => "third-party",
        _ => first,
    };

    sanitize_scope(raw_scope)
}

fn sanitize_scope(raw: &str) -> Option<String> {
    let scope = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        .collect::<String>()
        .to_ascii_lowercase();

    if scope.is_empty() { None } else { Some(scope) }
}

fn subject_quality_bonus(subject: &str) -> f32 {
    let lower = subject.to_ascii_lowercase();
    let mut score = 0.0f32;

    if lower.starts_with("add ")
        || lower.starts_with("build ")
        || lower.starts_with("fix ")
        || lower.starts_with("refactor ")
        || lower.starts_with("update ")
        || lower.starts_with("improve ")
        || lower.starts_with("guard ")
        || lower.starts_with("prevent ")
        || lower.starts_with("remove ")
        || lower.starts_with("support ")
        || lower.starts_with("enable ")
        || lower.starts_with("disable ")
        || lower.starts_with("migrate ")
        || lower.starts_with("introduce ")
    {
        score += 0.08;
    }

    let word_count = lower.split_whitespace().count();
    if (3..=12).contains(&word_count) {
        score += 0.04;
    } else if word_count < 2 || word_count > 16 {
        score -= 0.06;
    }

    if lower.contains("commit message") {
        score -= 0.35;
    }
    if lower.contains("synthesis") {
        score -= 0.25;
    }
    if lower.contains("misc") || lower.contains("various") {
        score -= 0.20;
    }

    score
}

fn sanitize_commit_subject(raw: &str) -> Option<String> {
    let mut subject = sanitize_sentence(raw);
    if subject.is_empty() {
        return None;
    }

    if let Some((_, _, desc)) = parse_conventional_commit(&subject) {
        subject = desc.to_string();
    }

    subject = subject
        .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .trim()
        .trim_end_matches(['.', ';', ','])
        .to_string();

    if subject.is_empty() {
        return None;
    }

    Some(clamp_chars(&subject, 72))
}

fn default_commit_subject(commit_type: &str, partial_count: usize) -> String {
    match commit_type {
        "feat" => "add requested behavior".to_string(),
        "fix" => "fix incorrect behavior".to_string(),
        "refactor" => "reorganize implementation details".to_string(),
        "docs" => "update project documentation".to_string(),
        "test" => "expand test coverage".to_string(),
        "perf" => "improve runtime performance".to_string(),
        "style" => "clean up code style".to_string(),
        _ => format!("update code across {partial_count} changes"),
    }
}

fn clamp_chars(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    out.trim().to_string()
}

fn capitalize_first(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = first.to_uppercase().collect::<String>();
    out.push_str(chars.as_str());
    out
}

fn decapitalize_first(value: &str) -> String {
    let looks_like_title_case = value
        .split_whitespace()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_alphabetic()))
        .all(|token| {
            token
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
        });
    if looks_like_title_case {
        return value.to_ascii_lowercase();
    }

    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = first.to_lowercase().collect::<String>();
    out.push_str(chars.as_str());
    out
}

fn looks_like_reducer_meta(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let has_any = |terms: &[&str]| terms.iter().any(|term| lower.contains(term));
    let has_all = |terms: &[&str]| terms.iter().all(|term| lower.contains(term));

    has_all(&["partial", "analys"])
        || has_all(&["chunk", "analys"])
        || has_all(&["consolidated", "report"])
        || has_all(&["analysis", "report"])
        || has_all(&["adaptive", "reducer"])
        || (lower.contains("reduce")
            && has_any(&["analysis", "analyses", "report", "chunk", "partial"]))
        || (lower.contains("synthesi") && has_any(&["analysis", "analyses", "report", "chunk"]))
}

fn normalize_commit_message(raw: &str) -> Option<String> {
    let header = raw.lines().next()?.trim();
    if header.is_empty() {
        return None;
    }

    if let Some((kind, scope, desc)) = parse_conventional_commit(header) {
        if looks_like_reducer_meta(desc) {
            return None;
        }
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
        if looks_like_reducer_meta(desc) {
            return None;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_item(
        type_tag: TypeTag,
        title: &str,
        intent: &str,
        path: &str,
        confidence: f32,
    ) -> ChangeItem {
        ChangeItem {
            id: "item-1".to_string(),
            bucket: ChangeBucket::Patch,
            type_tag,
            title: title.to_string(),
            intent: intent.to_string(),
            files: vec![FileRef {
                path: path.to_string(),
                status: FileStatus::Modified,
                ranges: Vec::new(),
            }],
            confidence,
        }
    }

    #[test]
    fn parse_analyze_output_recovers_from_truncated_json() {
        let raw = r#"{"summary":"Add state loading and saving","bucket":"Feature","type_tag":"Feat","title":"Add state loading","intent":"Add"#;
        let parsed = parse_analyze_output(raw).expect("parse should recover");
        assert_eq!(parsed.bucket, ChangeBucket::Feature);
        assert_eq!(parsed.type_tag, TypeTag::Feat);
        assert!(!parsed.summary.is_empty());
        assert!(!parsed.title.is_empty());
        assert!(!parsed.intent.is_empty());
    }

    #[test]
    fn parse_analyze_output_recovers_from_freeform_text() {
        let raw = "Refactor model_handle.rs to remove manual device selection and rely on llama defaults.";
        let parsed = parse_analyze_output(raw).expect("parse should recover");
        assert_eq!(parsed.type_tag, TypeTag::Refactor);
        assert_eq!(parsed.bucket, ChangeBucket::Patch);
        assert!(parsed.summary.contains("Refactor"));
    }

    #[test]
    fn parse_analyze_output_still_rejects_empty_text() {
        let err = parse_analyze_output("   ").expect_err("empty output should fail");
        assert!(
            err.to_string()
                .contains("failed to parse analyze JSON output")
        );
    }

    #[test]
    fn normalize_commit_message_rejects_reducer_meta() {
        let raw = "refactor: Consolidate 9 partial analyses into one consolidated report";
        assert!(normalize_commit_message(raw).is_none());
    }

    #[test]
    fn reducer_meta_detection_keeps_valid_reduce_wording() {
        let raw = "perf: reduce memory usage in model initialization";
        let normalized = normalize_commit_message(raw).expect("valid commit should be kept");
        assert_eq!(normalized, raw);
    }

    #[test]
    fn synthesize_fallback_commit_message_prefers_item_details() {
        let items = vec![sample_item(
            TypeTag::Refactor,
            "Consolidate runtime prompt handling",
            "Align prompt flow for reducer output",
            "crates/llama-runtime/src/model.rs",
            0.88,
        )];

        let commit = synthesize_fallback_commit_message(&items, 3);
        assert_eq!(
            commit,
            "refactor(llama-runtime): consolidate runtime prompt handling"
        );
    }

    #[test]
    fn synthesize_fallback_summary_composes_two_subjects() {
        let items = vec![
            sample_item(
                TypeTag::Refactor,
                "Consolidate runtime prompt handling",
                "Align prompt flow",
                "crates/llama-runtime/src/model.rs",
                0.90,
            ),
            sample_item(
                TypeTag::Fix,
                "Guard backend config export",
                "Prevent invalid backend defaults",
                "crates/llama-runtime/src/model_handle.rs",
                0.84,
            ),
        ];
        let stats = DiffStats {
            files_changed: 2,
            lines_changed: 100,
            hunks: 4,
            binary_files: 0,
        };

        let summary = synthesize_fallback_summary(&items, &stats);
        assert_eq!(
            summary,
            "Guard backend config export and Consolidate runtime prompt handling across 2 files."
        );
    }

    #[test]
    fn commit_synthesis_downranks_commit_message_meta_phrasing() {
        let items = vec![
            sample_item(
                TypeTag::Feat,
                "Build Reduce Prompt",
                "Produce reduce metadata prompt",
                "crates/core/src/llm/prompts.rs",
                0.75,
            ),
            sample_item(
                TypeTag::Refactor,
                "Refactor commit message synthesis",
                "Tune commit message synthesis flow",
                "crates/llama-runtime/src/model.rs",
                0.85,
            ),
        ];

        let commit = synthesize_fallback_commit_message(&items, 2);
        assert_eq!(commit, "feat(core): build reduce prompt");
    }

    #[test]
    fn mapped_change_item_id_uses_file_path_shape() {
        assert_eq!(
            mapped_change_item_id("crates/core/src/llm/prompts.rs", 0),
            "crates/core/src/llm/prompts.rs#chunk-1"
        );
    }
}
