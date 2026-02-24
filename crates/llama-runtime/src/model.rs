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
use crate::embed::{
    DEFAULT_HF_REPO, resolve_embedding_hf_repo, resolve_embedding_model_path,
    resolve_llama_cache_dir,
};
use crate::error::RuntimeError;
use crate::model_handle::{ModelHandle, list_cached_models as list_cached_models_bridge};
use crate::progress::{ProgressCallback, ProgressEvent, ProgressStage};

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
                    // Suppress INFO-level logs from llama.cpp's common library (e.g. download.cpp
                    // "using cached file" messages) which use a separate logging path.
                    llama_sys::bridge::autocommit_common_log_set_verbosity(0);
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
        model_path: Option<&Path>,
        model_hf_repo: Option<&str>,
        model_cache_dir: Option<&Path>,
        profile: &str,
        generation_state_path: Option<PathBuf>,
    ) -> Result<Self, RuntimeError> {
        let cpu_only = profile.eq_ignore_ascii_case("cpu");
        let model = Arc::new(ModelHandle::load(
            model_path,
            model_hf_repo,
            model_cache_dir,
            cpu_only,
        )?);

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
        mut on_progress: impl FnMut(usize),
    ) -> Result<Vec<PartialReport>, RuntimeError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let model = Arc::clone(&self.model);
        let generation_state_path = self.generation_state_path.clone();
        let generation_ctx = self.generation_ctx()?;
        let context_window_tokens = generation_ctx.context_window_tokens();
        let prompts = chunks
            .iter()
            .map(|chunk| {
                let prompt = build_analyze_prompt_capped(chunk, context_window_tokens);
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
        let budgets = chunks
            .iter()
            .map(|chunk| analyze_token_budget(chunk, context_window_tokens))
            .collect::<Vec<_>>();
        let mut out: Vec<PartialReport> = chunks
            .iter()
            .enumerate()
            .map(|(ordinal, chunk)| partial_from_chunk(runtime_context, ordinal, chunk))
            .collect();

        let window = generation_ctx.max_sequences().max(1);
        let mut completed = 0usize;
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
                        completed = completed.saturating_add(1);
                        on_progress(completed);
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
                        completed = completed.saturating_add(1);
                        on_progress(completed);
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

#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub local_path: Option<PathBuf>,
    pub hf_repo: Option<String>,
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct CachedModelList {
    pub cache_dir: PathBuf,
    pub models: Vec<String>,
}

pub fn list_cached_models(cache_dir: Option<PathBuf>) -> Result<CachedModelList, RuntimeError> {
    let (cache_dir, models) = list_cached_models_bridge(cache_dir.as_deref())?;
    Ok(CachedModelList { cache_dir, models })
}

impl ModelConfig {
    pub fn from_explicit(
        local_path: Option<PathBuf>,
        hf_repo: Option<String>,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            local_path,
            hf_repo: hf_repo
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            cache_dir,
        }
    }

    pub fn from_env_or_default() -> Self {
        let local_path = resolve_embedding_model_path();
        let hf_repo = if local_path.is_some() {
            None
        } else {
            resolve_embedding_hf_repo().or_else(|| Some(DEFAULT_HF_REPO.to_string()))
        };
        let cache_dir = resolve_llama_cache_dir();

        Self {
            local_path,
            hf_repo,
            cache_dir,
        }
    }

    pub fn with_default_hf_if_unset(mut self) -> Self {
        if self.local_path.is_none() && self.hf_repo.is_none() {
            self.hf_repo = Some(DEFAULT_HF_REPO.to_string());
        }
        self
    }
}

pub struct Engine {
    _backend: Arc<BackendGuard>,
    context: RuntimeContext,
    runtime_model: ModelConfig,
    generation_state_path: Option<PathBuf>,
    runtime: Mutex<Option<LoadedRuntime>>,
    progress: Mutex<Option<ProgressCallback>>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("context", &self.context)
            .field("runtime_model", &self.runtime_model)
            .field("generation_state_path", &self.generation_state_path)
            .finish_non_exhaustive()
    }
}

impl Engine {
    pub fn new(profile: &str) -> Result<Self, RuntimeError> {
        Self::new_with_generation_cache(profile, None)
    }

    pub fn new_with_generation_cache(
        profile: &str,
        generation_state_path: Option<PathBuf>,
    ) -> Result<Self, RuntimeError> {
        Self::new_with_generation_cache_and_model(
            profile,
            generation_state_path,
            ModelConfig::from_env_or_default(),
        )
    }

    pub fn new_with_generation_cache_and_model(
        profile: &str,
        generation_state_path: Option<PathBuf>,
        model_config: ModelConfig,
    ) -> Result<Self, RuntimeError> {
        let backend = Arc::new(BackendGuard::acquire());
        let runtime_model = model_config.with_default_hf_if_unset();
        Ok(Self {
            _backend: backend,
            context: RuntimeContext::new(profile),
            runtime_model,
            generation_state_path,
            runtime: Mutex::new(None),
            progress: Mutex::new(None),
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
        self.runtime_model.local_path.as_deref()
    }

    pub fn configured_hf_repo(&self) -> Option<&str> {
        self.runtime_model.hf_repo.as_deref()
    }

    pub fn configured_cache_dir(&self) -> Option<&Path> {
        self.runtime_model.cache_dir.as_deref()
    }

    pub fn warmup_generation_cache(&self) -> Result<(), RuntimeError> {
        self.with_runtime(|runtime| runtime.warmup_generation_cache())
    }

    fn with_runtime<T>(
        &self,
        f: impl FnOnce(&mut LoadedRuntime) -> Result<T, RuntimeError>,
    ) -> Result<T, RuntimeError> {
        if self.runtime_model.local_path.is_none() && self.runtime_model.hf_repo.is_none() {
            return Err(RuntimeError::Inference(
                "runtime model is not configured".to_string(),
            ));
        }

        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| RuntimeError::Inference("runtime lock poisoned".to_string()))?;

        if guard.is_none() {
            *guard = Some(LoadedRuntime::load(
                self.runtime_model.local_path.as_deref(),
                self.runtime_model.hf_repo.as_deref(),
                self.runtime_model.cache_dir.as_deref(),
                &self.context.profile,
                self.generation_state_path.clone(),
            )?);
        }

        f(guard.as_mut().expect("runtime just initialized"))
    }

    fn emit_progress(&self, stage: ProgressStage) {
        let callback = self.progress.lock().ok().and_then(|guard| guard.clone());
        if let Some(cb) = callback {
            cb(ProgressEvent { stage });
        }
    }

    fn embed_with_runtime(&self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        self.with_runtime(|runtime| runtime.embed(text))
    }

    fn analyze_chunks_with_runtime(
        &self,
        chunks: &[DiffChunk],
    ) -> Result<Vec<PartialReport>, RuntimeError> {
        let total = chunks.len();
        self.with_runtime(|runtime| {
            runtime.analyze_chunks(&self.context, chunks, |completed| {
                self.emit_progress(ProgressStage::Analyze { completed, total });
            })
        })
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
        let generated_summary_candidate = generated
            .as_ref()
            .and_then(|g| normalize_reduce_summary(&g.summary));
        let repaired_commit_candidate = generated
            .as_ref()
            .and_then(|g| recover_commit_message_from_reduce(g, &items));

        let fallback_commit_message = synthesize_fallback_commit_message(&items, partials.len());
        let fallback_summary = synthesize_fallback_summary(&items, stats);

        let (commit_message, commit_source) =
            if let Some(commit) = generated_commit_candidate.clone() {
                (commit, "reduce_model")
            } else if let Some(commit) = repaired_commit_candidate.clone() {
                (commit, "reduce_model_repaired")
            } else {
                (fallback_commit_message.clone(), "composed_partials")
            };
        let repaired_summary_candidate = generated
            .as_ref()
            .and_then(|g| recover_summary_from_reduce(g, &commit_message, stats));
        let (summary, summary_source) = if let Some(summary) = generated_summary_candidate.clone() {
            (summary, "reduce_model")
        } else if let Some(summary) = repaired_summary_candidate.clone() {
            (summary, "reduce_model_repaired")
        } else {
            (fallback_summary.clone(), "composed_partials")
        };

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
        risk_notes.push(format!("commit_source:{commit_source}"));
        risk_notes.push(format!("summary_source:{summary_source}"));
        if generated.is_some() {
            risk_notes.push(if generated_commit_candidate.is_some() {
                "reduce_commit_candidate:usable".to_string()
            } else if repaired_commit_candidate.is_some() {
                "reduce_commit_candidate:repaired".to_string()
            } else {
                "reduce_commit_candidate:invalid".to_string()
            });
            risk_notes.push(if generated_summary_candidate.is_some() {
                "reduce_summary_candidate:usable".to_string()
            } else if repaired_summary_candidate.is_some() {
                "reduce_summary_candidate:repaired".to_string()
            } else {
                "reduce_summary_candidate:invalid".to_string()
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

        let report = AnalysisReport {
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
        };

        self.emit_progress(ProgressStage::Reduce);
        Ok(report)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        let result = self
            .embed_with_runtime(text)
            .map_err(|err| CoreError::Engine(err.to_string()))?;
        self.emit_progress(ProgressStage::Embedding);
        Ok(result)
    }

    fn model_fingerprint(&self) -> Option<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        if let Some(path) = self.runtime_model.local_path.as_ref() {
            path.to_string_lossy().hash(&mut hasher);
        }
        if let Some(repo) = self.runtime_model.hf_repo.as_ref() {
            repo.hash(&mut hasher);
        }
        self.context.profile.hash(&mut hasher);
        Some(format!("{:016x}", hasher.finish()))
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Option<f32> {
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let len = a.len().min(b.len());
        let result = unsafe {
            llama_sys::bridge::autocommit_cosine_similarity(
                a.as_ptr(),
                b.as_ptr(),
                len as std::ffi::c_int,
            )
        };
        if result.abs() < f32::EPSILON && len > 0 {
            // Zero result may indicate degenerate input; return None like the fallback.
            None
        } else {
            Some(result)
        }
    }

    fn set_progress_callback(&self, callback: Option<ProgressCallback>) {
        if let Ok(mut guard) = self.progress.lock() {
            *guard = callback;
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct ReduceModelOutput {
    #[serde(default, alias = "commit", alias = "message")]
    commit_message: String,
    #[serde(default, alias = "description")]
    summary: String,
    #[serde(default, alias = "risk")]
    risk_level: String,
    #[serde(default)]
    #[serde(alias = "notes")]
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

    let indices = sampled_indices(partials, max_items);
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
    if let Some(theme) = reduce_dominant_theme(partials) {
        prompt.push_str(&format!("- dominant_theme={theme}\n"));
    }
    let scope_distribution = reduce_scope_distribution(partials, 4);
    if !scope_distribution.is_empty() {
        prompt.push_str(&format!("- scope_distribution={scope_distribution}\n"));
    }
    let type_distribution = reduce_type_distribution(partials);
    if !type_distribution.is_empty() {
        prompt.push_str(&format!("- type_distribution={type_distribution}\n"));
    }
    // Pre-aggregate by scope for more generalized output.
    let aggregated = aggregate_partials_by_scope(partials, &indices);
    prompt.push_str("Scope summary:\n");

    let mut sampled = 0usize;
    for group in &aggregated {
        if prompt.len() >= max_prompt_chars {
            break;
        }
        sampled += group.count;
        prompt.push_str("- ");
        prompt.push_str(&group.display);
        prompt.push('\n');
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
    if let Some(override_tokens) = env_budget_override("AUTOCOMMIT_REDUCE_MAX_TOKENS") {
        return override_tokens.clamp(96, 2_048);
    }

    let complexity = reduce_complexity(stats, total);
    blend_budget(320, 448, complexity)
}

fn reduce_item_budget(stats: &DiffStats, total: usize) -> usize {
    if let Some(override_items) = env_budget_override("AUTOCOMMIT_REDUCE_MAX_ITEMS") {
        return override_items.clamp(1, 64).min(total.max(1));
    }

    let complexity = reduce_complexity(stats, total);
    blend_budget(12, 24, complexity).min(total)
}

fn reduce_char_budget(stats: &DiffStats, total: usize) -> usize {
    if let Some(override_chars) = env_budget_override("AUTOCOMMIT_REDUCE_MAX_PROMPT_CHARS") {
        return override_chars.clamp(2_000, 20_000);
    }

    let complexity = reduce_complexity(stats, total);
    blend_budget(4_800, 10_000, complexity)
}

fn reduce_complexity(stats: &DiffStats, total: usize) -> f32 {
    let lines = (stats.lines_changed as f32 / 8_000.0).clamp(0.0, 1.0);
    let chunks = (total as f32 / 96.0).clamp(0.0, 1.0);
    let hunks = (stats.hunks as f32 / 160.0).clamp(0.0, 1.0);
    ((lines * 0.6) + (chunks * 0.3) + (hunks * 0.1)).sqrt()
}

fn blend_budget(low: usize, high: usize, factor: f32) -> usize {
    let factor = factor.clamp(0.0, 1.0);
    let span = high.saturating_sub(low) as f32;
    let value = low as f32 + span * factor;
    value.round() as usize
}

fn sampled_indices(partials: &[PartialReport], max_items: usize) -> Vec<usize> {
    let total = partials.len();
    if total == 0 || max_items == 0 {
        return Vec::new();
    }

    if total <= max_items {
        return (0..total).collect();
    }

    let mut scored = Vec::with_capacity(total);
    let mut best_by_scope = std::collections::BTreeMap::<String, (usize, f32)>::new();

    for (idx, partial) in partials.iter().enumerate() {
        let score = reduce_partial_score(partial);
        scored.push((idx, score));

        let scope = reduce_partial_scope_key(partial);
        let best = best_by_scope.entry(scope).or_insert((idx, score));
        if score > best.1 {
            *best = (idx, score);
        }
    }

    let mut by_scope = best_by_scope.into_values().collect::<Vec<_>>();
    by_scope.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut selected = Vec::with_capacity(max_items);
    let mut selected_set = std::collections::BTreeSet::new();

    for (idx, _) in by_scope {
        if selected.len() >= max_items {
            break;
        }
        if selected_set.insert(idx) {
            selected.push(idx);
        }
    }

    if selected.len() < max_items {
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        for (idx, _) in scored {
            if selected.len() >= max_items {
                break;
            }
            if selected_set.insert(idx) {
                selected.push(idx);
            }
        }
    }

    selected
}

fn reduce_partial_score(partial: &PartialReport) -> f32 {
    let mut score = 0.42f32;
    let normalized_summary = sanitize_sentence(&partial.summary);
    if normalized_summary.contains("fallback heuristic due to analyze_error:") {
        score -= 0.25;
    } else {
        score += 0.02;
    }

    if let Some(item) = partial.items.first() {
        score += item.confidence.clamp(0.0, 1.0);
        if item.files.len() > 1 {
            score += 0.05;
        }
        if !matches!(item.type_tag, TypeTag::Mixed) {
            score += 0.04;
        }
        if !matches!(item.bucket, ChangeBucket::Other) {
            score += 0.03;
        }
    }

    score
}

fn reduce_partial_scope_key(partial: &PartialReport) -> String {
    partial
        .items
        .first()
        .and_then(|item| item.files.first())
        .and_then(|file| scope_from_path(&file.path))
        .unwrap_or_else(|| "unknown".to_string())
}

#[allow(dead_code)]
fn format_partial_for_reduce(partial: &PartialReport) -> String {
    let summary = compact_reduce_text(&partial.summary, 220);
    if let Some(item) = partial.items.first() {
        let scope = reduce_partial_scope_key(partial);
        let file_paths = item
            .files
            .iter()
            .take(2)
            .map(|f| f.path.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let title = compact_reduce_text(&item.title, 120);
        format!(
            "scope={} bucket={:?} type={:?} title={} files={} confidence={:.2} summary={}",
            scope, item.bucket, item.type_tag, title, file_paths, item.confidence, summary
        )
    } else {
        summary
    }
}

fn reduce_dominant_theme(partials: &[PartialReport]) -> Option<String> {
    let type_distribution = reduce_type_distribution(partials);
    let dominant_type = type_distribution
        .split(',')
        .next()
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    let scope_distribution = reduce_scope_distribution(partials, 1);
    let dominant_scope = scope_distribution
        .split(',')
        .next()
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    if dominant_type.is_empty() && dominant_scope.is_empty() {
        return None;
    }
    if dominant_type.is_empty() {
        return Some(dominant_scope.to_string());
    }
    if dominant_scope.is_empty() {
        return Some(dominant_type.to_string());
    }
    Some(format!("{dominant_type} in {dominant_scope}"))
}

fn reduce_scope_distribution(partials: &[PartialReport], max_scopes: usize) -> String {
    if max_scopes == 0 {
        return String::new();
    }

    let mut counts = std::collections::BTreeMap::<String, f32>::new();
    for partial in partials {
        let scope = reduce_partial_scope_key(partial);
        let weight = partial
            .items
            .first()
            .map(|item| item.confidence.clamp(0.05, 1.0))
            .unwrap_or(0.1);
        *counts.entry(scope).or_insert(0.0) += weight;
    }

    render_distribution(counts, max_scopes)
}

fn reduce_type_distribution(partials: &[PartialReport]) -> String {
    let mut counts = std::collections::BTreeMap::<String, f32>::new();
    for partial in partials {
        let Some(item) = partial.items.first() else {
            continue;
        };
        let weight = item.confidence.clamp(0.05, 1.0);
        *counts
            .entry(type_tag_prefix(item.type_tag.clone()).to_string())
            .or_insert(0.0) += weight;
    }

    render_distribution(counts, 4)
}

fn render_distribution(
    counts: std::collections::BTreeMap<String, f32>,
    max_items: usize,
) -> String {
    if counts.is_empty() || max_items == 0 {
        return String::new();
    }

    let total = counts.values().sum::<f32>().max(f32::EPSILON);
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    ranked
        .into_iter()
        .take(max_items)
        .map(|(label, score)| {
            let pct = (score / total) * 100.0;
            format!("{label}:{pct:.0}%")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

struct ScopeGroup {
    display: String,
    count: usize,
}

fn aggregate_partials_by_scope(partials: &[PartialReport], indices: &[usize]) -> Vec<ScopeGroup> {
    use std::collections::BTreeMap;

    // Group selected partials by scope.
    let mut groups: BTreeMap<String, Vec<&PartialReport>> = BTreeMap::new();
    for &idx in indices {
        if let Some(partial) = partials.get(idx) {
            let scope = reduce_partial_scope_key(partial);
            groups.entry(scope).or_default().push(partial);
        }
    }

    let mut result: Vec<ScopeGroup> = Vec::with_capacity(groups.len());
    for (scope, items) in &groups {
        let file_count = items.len();

        // Count type tags within this scope.
        let mut type_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut titles: Vec<&str> = Vec::new();
        for partial in items {
            if let Some(item) = partial.items.first() {
                let tag = type_tag_prefix(item.type_tag.clone());
                *type_counts.entry(tag).or_insert(0) += 1;
                if titles.len() < 2 {
                    titles.push(item.title.as_str());
                }
            }
        }

        let types_str = type_counts
            .iter()
            .map(|(tag, count)| format!("{count} {tag}"))
            .collect::<Vec<_>>()
            .join(", ");

        let titles_str = titles.join("; ");
        let titles_compact = if titles_str.len() > 120 {
            format!("{}...", &titles_str[..117])
        } else {
            titles_str
        };

        let display = format!(
            "scope={scope} ({file_count} files): {types_str} \u{2014} {titles_compact}"
        );

        result.push(ScopeGroup {
            display,
            count: file_count,
        });
    }

    // Sort by count descending (largest scope groups first).
    result.sort_by(|a, b| b.count.cmp(&a.count));
    result
}

#[allow(dead_code)]
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

    if last.is_none() {
        last = salvage_reduce_output(raw);
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

    if output.commit_message.is_empty() && output.summary.is_empty() {
        return Err(RuntimeError::Inference(
            "reduce output is missing both commit_message and summary".to_string(),
        ));
    }

    if !matches!(output.risk_level.as_str(), "low" | "medium" | "high") {
        output.risk_level = "medium".to_string();
    }

    output.risk_notes.retain(|note| !note.trim().is_empty());
    Ok(output)
}

fn salvage_reduce_output(raw: &str) -> Option<ReduceModelOutput> {
    let mut commit_message: Option<String> = None;
    let mut summary: Option<String> = None;

    for line in raw.lines() {
        let cleaned = line
            .trim()
            .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
            .trim_start_matches("- ")
            .trim_start_matches("* ")
            .trim()
            .to_string();
        if cleaned.is_empty() {
            continue;
        }

        let lowered = cleaned.to_ascii_lowercase();
        let normalized_commit = if let Some(rest) = lowered
            .strip_prefix("commit:")
            .map(|_| cleaned["commit:".len()..].trim())
        {
            normalize_commit_message(rest)
        } else {
            normalize_commit_message(&cleaned)
        };
        if commit_message.is_none() {
            if let Some(commit) = normalized_commit {
                commit_message = Some(commit);
                continue;
            }
        }

        if summary.is_none() {
            let summary_candidate = if let Some(rest) = lowered
                .strip_prefix("summary:")
                .map(|_| cleaned["summary:".len()..].trim())
            {
                normalize_reduce_summary(rest)
            } else {
                normalize_reduce_summary(&cleaned)
            };
            if let Some(candidate) = summary_candidate {
                summary = Some(candidate);
            }
        }
    }

    if commit_message.is_none() && summary.is_none() {
        return None;
    }

    Some(ReduceModelOutput {
        commit_message: commit_message.unwrap_or_default(),
        summary: summary.unwrap_or_default(),
        risk_level: "medium".to_string(),
        risk_notes: vec!["salvaged_reduce_output".to_string()],
    })
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
                let retry_raw = generation_ctx.generate_text(
                    prompt,
                    Some(grammar::ANALYZE_GBNF),
                    max_tokens.saturating_add(96),
                )?;
                parse_analyze_output(&retry_raw).map_err(|retry_err| {
                    RuntimeError::Inference(format!(
                        "analyze parse failed with grammar: primary={parse_err}; retry={retry_err}"
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
                    prompt,
                    Some(grammar::ANALYZE_GBNF),
                    max_tokens.saturating_add(96),
                )?;
                parse_analyze_output(&retry_raw).map_err(|retry_err| {
                    RuntimeError::Inference(format!(
                        "analyze grammar pass failed ({err}); grammar retry parse failed ({retry_err})"
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
    if analyze_output_has_dangling_fragment(&output) {
        return Err(RuntimeError::Inference(
            "analyze output has dangling fragment likely caused by token truncation".to_string(),
        ));
    }

    Ok(output)
}

fn analyze_output_has_dangling_fragment(output: &AnalyzeModelOutput) -> bool {
    ends_with_dangling_joiner(&output.intent)
        || ends_with_dangling_joiner(&output.summary)
        || ends_with_dangling_joiner(&output.title)
}

fn ends_with_dangling_joiner(value: &str) -> bool {
    let cleaned = sanitize_sentence(value).trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        return false;
    }

    let cleaned = cleaned.trim_end_matches(|ch: char| {
        matches!(
            ch,
            '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | ')' | ']' | '}'
        )
    });
    let word_count = cleaned.split_whitespace().count();
    if word_count < 3 {
        return false;
    }

    let dangling_two_word_suffixes = ["based on", "as a", "such as", "in order", "up to"];
    if dangling_two_word_suffixes
        .iter()
        .any(|suffix| cleaned.ends_with(suffix))
    {
        return true;
    }

    let last = cleaned
        .split_whitespace()
        .last()
        .unwrap_or_default()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
    matches!(
        last,
        "a" | "an"
            | "the"
            | "to"
            | "for"
            | "with"
            | "without"
            | "from"
            | "into"
            | "on"
            | "in"
            | "of"
            | "and"
            | "or"
            | "but"
            | "via"
            | "by"
            | "based"
    )
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

fn analyze_token_budget(chunk: &DiffChunk, context_window_tokens: usize) -> usize {
    if let Some(override_tokens) = env_budget_override("AUTOCOMMIT_ANALYZE_MAX_TOKENS") {
        return override_tokens.clamp(96, 1_024);
    }

    let density = (chunk.estimated_tokens as f32 / 4_000.0)
        .clamp(0.0, 1.0)
        .sqrt();
    let context_bonus = ((context_window_tokens as f32 - 2_048.0) / 4_096.0).clamp(0.0, 1.0);
    let base = 344.0 - (72.0 * density) + (24.0 * context_bonus);
    base.round().clamp(224.0, 384.0) as usize
}

fn build_analyze_prompt_capped(chunk: &DiffChunk, context_window_tokens: usize) -> String {
    let max_chars = analyze_char_budget(chunk, context_window_tokens);

    if chunk.text.len() <= max_chars {
        return prompts::build_analyze_prompt(chunk);
    }

    let truncated = compact_diff_for_prompt(&chunk.text, max_chars);

    format!(
        "/no_think\n\
Task: Analyze one diff chunk.\n\
Return ONLY JSON with keys: summary, bucket, type_tag, title, intent.\n\
Allowed bucket values: Feature, Patch, Addition, Other.\n\
Allowed type_tag values: Feat, Fix, Refactor, Docs, Test, Chore, Perf, Style, Mixed.\n\
summary: <= 18 words, concrete change outcome.\n\
title: <= 10 words, action-first phrase.\n\
intent: <= 16 words, concise rationale phrase (not a long sentence).\n\
Do not end `summary`, `title`, or `intent` with dangling filler words.\n\
No markdown, no prose outside JSON.\n\
Path: {}\n\
Diff:\n```diff\n{}\n```",
        chunk.path, truncated
    )
}

fn analyze_char_budget(chunk: &DiffChunk, context_window_tokens: usize) -> usize {
    if let Some(override_chars) = env_budget_override("AUTOCOMMIT_ANALYZE_MAX_PROMPT_CHARS") {
        return override_chars.clamp(2_000, 16_000);
    }

    let context_cap = context_window_tokens.saturating_mul(3).clamp(2_000, 14_000);
    let density = (chunk.estimated_tokens as f32 / 6_000.0)
        .clamp(0.0, 1.0)
        .sqrt();
    let raw = 5_400.0 - (2_000.0 * density);
    let contextual = raw.min((context_cap as f32) * 0.92);
    contextual.round().clamp(2_400.0, context_cap as f32) as usize
}

fn compact_diff_for_prompt(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return String::new();
    }
    let mut used = vec![false; lines.len()];
    let mut out = String::with_capacity(max_chars + 128);
    let mut header_lines = 0usize;
    let mut change_lines = 0usize;
    let mut semantic_lines = 0usize;

    for (idx, line) in lines.iter().copied().enumerate() {
        if is_file_header_line(line) {
            if !push_line_index_with_limit(&mut out, &lines, &mut used, idx, max_chars) {
                break;
            }
            header_lines += 1;
        }
    }

    let hunk_ranges = extract_hunk_ranges(&lines);
    let selected_hunks = if hunk_ranges.is_empty() {
        vec![0usize]
    } else {
        let max_hunks = (max_chars / 900).clamp(2, 10);
        sample_stratified_indices(hunk_ranges.len(), max_hunks)
    };

    let mut scoped_hunks = Vec::with_capacity(selected_hunks.len());
    if hunk_ranges.is_empty() {
        scoped_hunks.push((0usize, lines.len()));
    } else {
        for idx in selected_hunks {
            if let Some(range) = hunk_ranges.get(idx).copied() {
                scoped_hunks.push(range);
            }
        }
    }

    for (start, _) in scoped_hunks.iter().copied() {
        if start < lines.len()
            && lines[start].starts_with("@@ ")
            && push_line_index_with_limit(&mut out, &lines, &mut used, start, max_chars)
        {
            header_lines += 1;
        }
    }

    let per_hunk_change_budget = ((max_chars / scoped_hunks.len().max(1)) / 95).clamp(3, 10);
    for (start, end) in scoped_hunks.iter().copied() {
        let candidates = (start..end)
            .filter(|&idx| is_change_line(lines[idx]))
            .collect::<Vec<_>>();
        for relative_idx in sample_stratified_indices(candidates.len(), per_hunk_change_budget) {
            let idx = candidates[relative_idx];
            if !push_line_index_with_limit(&mut out, &lines, &mut used, idx, max_chars) {
                break;
            }
            change_lines += 1;
        }
    }

    let mut change_buckets = scoped_hunks
        .iter()
        .map(|(start, end)| {
            (*start..*end)
                .filter(|&idx| is_change_line(lines[idx]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    change_lines += round_robin_fill(&mut out, &lines, &mut used, &mut change_buckets, max_chars);

    let per_hunk_semantic_budget = ((max_chars / scoped_hunks.len().max(1)) / 260).clamp(1, 2);
    for (start, end) in scoped_hunks.iter().copied() {
        let candidates = (start..end)
            .filter(|&idx| is_semantic_context_line(lines[idx]))
            .collect::<Vec<_>>();
        for relative_idx in sample_stratified_indices(candidates.len(), per_hunk_semantic_budget) {
            let idx = candidates[relative_idx];
            if !push_line_index_with_limit(&mut out, &lines, &mut used, idx, max_chars) {
                break;
            }
            semantic_lines += 1;
        }
    }

    if out.len() < max_chars / 2 {
        let mut semantic_buckets = scoped_hunks
            .iter()
            .map(|(start, end)| {
                (*start..*end)
                    .filter(|&idx| is_semantic_context_line(lines[idx]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        semantic_lines += round_robin_fill(
            &mut out,
            &lines,
            &mut used,
            &mut semantic_buckets,
            max_chars,
        );
    }

    if out.len() < max_chars / 3 {
        out.clear();
        used.fill(false);
        header_lines = 0;
        change_lines = 0;
        semantic_lines = 0;
        append_head_tail_with_limit(text, &mut out, max_chars);
        for line in out.lines() {
            if is_file_header_line(line) || line.starts_with("@@ ") {
                header_lines += 1;
            } else if is_change_line(line) {
                change_lines += 1;
            } else if is_semantic_context_line(line) {
                semantic_lines += 1;
            }
        }
    }

    out.push_str(&format!(
        "\n... [diff truncated for budget; kept {} headers, {} change lines, {} semantic lines]",
        header_lines, change_lines, semantic_lines
    ));
    out
}

fn extract_hunk_ranges(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut current_start: Option<usize> = None;

    for (idx, line) in lines.iter().copied().enumerate() {
        if line.starts_with("@@ ") {
            if let Some(start) = current_start.replace(idx) {
                out.push((start, idx));
            }
        } else if line.starts_with("diff --git ") {
            if let Some(start) = current_start.take() {
                out.push((start, idx));
            }
        }
    }

    if let Some(start) = current_start {
        out.push((start, lines.len()));
    }

    out
}

fn sample_stratified_indices(total: usize, limit: usize) -> Vec<usize> {
    if total == 0 || limit == 0 {
        return Vec::new();
    }
    if total <= limit {
        return (0..total).collect();
    }
    if limit == 1 {
        return vec![0];
    }

    let mut selected = std::collections::BTreeSet::new();
    let last = total.saturating_sub(1);
    let denom = limit.saturating_sub(1);

    for slot in 0..limit {
        let numerator = slot.saturating_mul(last);
        let idx = (numerator + (denom / 2)) / denom;
        selected.insert(idx.min(last));
    }

    let mut next = 0usize;
    while selected.len() < limit {
        if selected.insert(next) {
            next = next.saturating_add(1);
            continue;
        }
        next = next.saturating_add(1);
    }

    selected.into_iter().collect()
}

fn round_robin_fill(
    out: &mut String,
    lines: &[&str],
    used: &mut [bool],
    buckets: &mut [Vec<usize>],
    max_chars: usize,
) -> usize {
    if buckets.is_empty() {
        return 0;
    }

    let mut cursors = vec![0usize; buckets.len()];
    let mut added = 0usize;

    loop {
        let mut progressed = false;
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            let cursor = &mut cursors[bucket_idx];
            while *cursor < bucket.len() && used[bucket[*cursor]] {
                *cursor += 1;
            }
            if *cursor >= bucket.len() {
                continue;
            }

            let idx = bucket[*cursor];
            if !push_line_index_with_limit(out, lines, used, idx, max_chars) {
                return added;
            }
            *cursor += 1;
            added += 1;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    added
}

fn push_line_index_with_limit(
    out: &mut String,
    lines: &[&str],
    used: &mut [bool],
    idx: usize,
    max_chars: usize,
) -> bool {
    if idx >= lines.len() || used[idx] {
        return true;
    }
    if !push_line_with_limit(out, lines[idx], max_chars) {
        return false;
    }
    used[idx] = true;
    true
}

fn is_file_header_line(line: &str) -> bool {
    line.starts_with("diff --git ")
        || line.starts_with("index ")
        || line.starts_with("--- ")
        || line.starts_with("+++ ")
}

fn push_line_with_limit(out: &mut String, line: &str, max_chars: usize) -> bool {
    let required = line.len().saturating_add(1);
    if out.len().saturating_add(required) > max_chars {
        return false;
    }
    out.push_str(line);
    out.push('\n');
    true
}

fn append_head_tail_with_limit(text: &str, out: &mut String, max_chars: usize) {
    let lines = text.lines().collect::<Vec<_>>();
    let head = lines.len().min(24);
    let tail = lines.len().saturating_sub(head).min(12);

    for line in lines.iter().copied().take(head) {
        if !push_line_with_limit(out, line, max_chars) {
            return;
        }
    }
    if head + tail < lines.len() {
        if !push_line_with_limit(out, "...", max_chars) {
            return;
        }
    }
    for line in lines.iter().copied().skip(lines.len().saturating_sub(tail)) {
        if !push_line_with_limit(out, line, max_chars) {
            return;
        }
    }
}

fn is_change_line(line: &str) -> bool {
    (line.starts_with('+') || line.starts_with('-'))
        && !line.starts_with("+++ ")
        && !line.starts_with("--- ")
}

fn is_semantic_context_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("trait ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("interface ")
        || trimmed.starts_with("mod ")
        || trimmed.starts_with("const ")
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
    let (bucket, type_tag) =
        reconcile_model_labels(&model.bucket, &model.type_tag, &title, &intent);

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
            bucket,
            type_tag,
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

fn reconcile_model_labels(
    bucket: &ChangeBucket,
    type_tag: &TypeTag,
    title: &str,
    intent: &str,
) -> (ChangeBucket, TypeTag) {
    let combined = format!(
        "{} {}",
        title.to_ascii_lowercase(),
        intent.to_ascii_lowercase()
    );
    let inferred_type = infer_type_tag_from_text(&combined);
    let inferred_bucket = infer_bucket_from_text(&combined);

    let mut final_type = type_tag.clone();
    if should_override_type_tag(type_tag, &inferred_type) {
        final_type = inferred_type;
    }

    let mut final_bucket = bucket.clone();
    if matches!(final_bucket, ChangeBucket::Other)
        && !matches!(inferred_bucket, ChangeBucket::Other)
    {
        final_bucket = inferred_bucket;
    }
    final_bucket = normalize_bucket_for_type(final_bucket, &final_type);

    (final_bucket, final_type)
}

fn should_override_type_tag(current: &TypeTag, inferred: &TypeTag) -> bool {
    if matches!(inferred, TypeTag::Mixed) {
        return false;
    }

    match current {
        TypeTag::Feat => matches!(
            inferred,
            TypeTag::Fix
                | TypeTag::Refactor
                | TypeTag::Docs
                | TypeTag::Test
                | TypeTag::Perf
                | TypeTag::Style
        ),
        TypeTag::Mixed | TypeTag::Chore => true,
        _ => false,
    }
}

fn normalize_bucket_for_type(bucket: ChangeBucket, type_tag: &TypeTag) -> ChangeBucket {
    match type_tag {
        TypeTag::Refactor
        | TypeTag::Fix
        | TypeTag::Test
        | TypeTag::Perf
        | TypeTag::Style
        | TypeTag::Chore => {
            if matches!(bucket, ChangeBucket::Feature) {
                ChangeBucket::Patch
            } else {
                bucket
            }
        }
        TypeTag::Docs => {
            if matches!(bucket, ChangeBucket::Feature | ChangeBucket::Other) {
                ChangeBucket::Addition
            } else {
                bucket
            }
        }
        TypeTag::Feat => {
            if matches!(bucket, ChangeBucket::Other) {
                ChangeBucket::Feature
            } else {
                bucket
            }
        }
        TypeTag::Mixed => bucket,
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

fn dominant_commit_type_by_confidence(items: &[ChangeItem]) -> Option<&'static str> {
    let mut feat = 0.0f32;
    let mut fix = 0.0f32;
    let mut refactor = 0.0f32;
    let mut docs = 0.0f32;
    let mut test = 0.0f32;
    let mut chore = 0.0f32;

    for item in items {
        let weight = item.confidence.clamp(0.05, 1.0);
        match item.type_tag {
            TypeTag::Feat => feat += weight,
            TypeTag::Fix => fix += weight,
            TypeTag::Refactor => refactor += weight,
            TypeTag::Docs => docs += weight,
            TypeTag::Test => test += weight,
            TypeTag::Chore | TypeTag::Perf | TypeTag::Style | TypeTag::Mixed => chore += weight,
        }
    }

    let ranked = [
        (feat, "feat"),
        (fix, "fix"),
        (refactor, "refactor"),
        (docs, "docs"),
        (test, "test"),
        (chore, "chore"),
    ];

    ranked
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|(weight, kind)| {
            if weight > f32::EPSILON {
                Some(kind)
            } else {
                None
            }
        })
}

fn reconcile_commit_type_with_items(
    candidate_type: &'static str,
    items: &[ChangeItem],
) -> &'static str {
    let dominant_type = dominant_commit_type_by_confidence(items)
        .or_else(|| dominant_type_tag(items).map(type_tag_prefix))
        .unwrap_or("chore");
    if dominant_type == candidate_type || items.is_empty() {
        return candidate_type;
    }

    let mut total_weight = 0.0f32;
    let mut candidate_weight = 0.0f32;
    let mut dominant_weight = 0.0f32;

    for item in items {
        let weight = item.confidence.clamp(0.05, 1.0);
        total_weight += weight;

        let item_type = type_tag_prefix(item.type_tag.clone());
        if item_type == candidate_type {
            candidate_weight += weight;
        }
        if item_type == dominant_type {
            dominant_weight += weight;
        }
    }

    if total_weight <= f32::EPSILON {
        return dominant_type;
    }

    let candidate_ratio = candidate_weight / total_weight;
    let dominant_ratio = dominant_weight / total_weight;
    if candidate_ratio >= 0.45 {
        return candidate_type;
    }

    if candidate_type == "feat" && candidate_ratio < 0.34 {
        return dominant_type;
    }

    if dominant_ratio >= candidate_ratio + 0.2 {
        dominant_type
    } else {
        candidate_type
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
        let description =
            trim_redundant_commit_type_prefix(commit_type, &decapitalize_first(&best.subject));

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

    let merged_second = merge_summary_subject(subjects[0].as_str(), subjects[1].as_str());
    format!(
        "{} and {} across {file_count} {noun}.",
        capitalize_first(&subjects[0]),
        merged_second
    )
}

fn trim_redundant_commit_type_prefix(commit_type: &str, description: &str) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let lower = trimmed.to_ascii_lowercase();
    let mut out = match commit_type {
        "refactor" if lower.starts_with("refactor ") => trimmed["refactor ".len()..].trim(),
        "fix" if lower.starts_with("fix ") => trimmed["fix ".len()..].trim(),
        "docs" if lower.starts_with("document ") => trimmed["document ".len()..].trim(),
        "test" if lower.starts_with("test ") => trimmed["test ".len()..].trim(),
        _ => trimmed,
    };

    if out.is_empty() {
        out = trimmed;
    }

    out.to_string()
}

fn merge_summary_subject(first: &str, second: &str) -> String {
    let first_token = leading_action_token(first);
    let second_token = leading_action_token(second);

    if let (Some(a), Some(b)) = (first_token.as_deref(), second_token.as_deref()) {
        if a == b && is_mergeable_action_token(a) {
            let suffix = drop_leading_action_token(second, a).trim();
            if !suffix.is_empty() {
                return decapitalize_first(suffix);
            }
        }
    }

    decapitalize_first(second)
}

fn leading_action_token(subject: &str) -> Option<String> {
    let token = subject
        .split_whitespace()
        .next()?
        .trim()
        .to_ascii_lowercase();
    if token.is_empty() { None } else { Some(token) }
}

fn drop_leading_action_token<'a>(subject: &'a str, token: &str) -> &'a str {
    let trimmed = subject.trim_start();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let first = parts.next().unwrap_or_default();
    if first.eq_ignore_ascii_case(token) {
        parts.next().unwrap_or_default()
    } else {
        trimmed
    }
}

fn is_mergeable_action_token(token: &str) -> bool {
    token.contains('/')
        || matches!(
            token,
            "add"
                | "build"
                | "fix"
                | "refactor"
                | "update"
                | "improve"
                | "guard"
                | "prevent"
                | "remove"
                | "support"
                | "enable"
                | "disable"
                | "migrate"
                | "introduce"
                | "extract"
                | "simplify"
                | "reorganize"
                | "document"
                | "test"
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
        let primary_path = item.files.first().map(|file| file.path.as_str());
        let item_scope = item
            .files
            .first()
            .and_then(|file| scope_from_path(&file.path));

        for (raw, bonus) in [(&item.title, 0.02f32), (&item.intent, 0.0f32)] {
            let Some(candidate_raw) = sanitize_commit_subject(raw) else {
                continue;
            };
            if is_low_information_subject(&candidate_raw) {
                continue;
            }
            if looks_like_reducer_meta(&candidate_raw) {
                continue;
            }
            let candidate = clamp_chars(&candidate_raw, 72);
            if candidate.is_empty() {
                continue;
            }

            let score = item.confidence
                + bonus
                + subject_quality_bonus(&candidate_raw)
                + subject_context_bonus(&candidate_raw, primary_path);
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

    if scope.is_empty() {
        return None;
    }

    if matches!(
        scope.as_str(),
        "scope"
            | "module"
            | "file"
            | "files"
            | "project"
            | "repo"
            | "repository"
            | "code"
            | "default"
            | "misc"
            | "mixed"
            | "unknown"
            | "change"
            | "changes"
            | "autocommit"
    ) {
        return None;
    }

    Some(scope)
}

fn subject_quality_bonus(subject: &str) -> f32 {
    let lower = subject.to_ascii_lowercase();
    let mut score = 0.0f32;

    if lower.starts_with("add ")
        || lower.starts_with("build ")
        || lower.starts_with("compose ")
        || lower.starts_with("consolidate ")
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
        if lower.contains("synthesi") || lower.contains("composition") {
            score -= 0.35;
        } else {
            score -= 0.08;
        }
    }
    if lower.contains("synthesis") {
        score -= 0.25;
    }
    if lower.contains("misc") || lower.contains("various") {
        score -= 0.20;
    }
    if lower.starts_with("implement a feature that ")
        || lower.starts_with("implement feature that ")
        || lower.starts_with("add a feature that ")
        || lower.starts_with("add feature that ")
        || lower.starts_with("create a feature that ")
        || lower.starts_with("create feature that ")
    {
        score -= 0.25;
    }

    score
}

fn subject_context_bonus(subject: &str, file_path: Option<&str>) -> f32 {
    let lower = subject.to_ascii_lowercase();
    let mut score = 0.0f32;

    if lower == "add version number"
        || lower == "add version"
        || lower == "update version"
        || lower == "version bump"
        || lower == "bump version"
    {
        score -= 0.40;
    }
    if lower.contains("update cargo.lock versions")
        || lower.contains("update cargo lock versions")
        || lower.contains("update lockfile")
        || lower.contains("update dependencies")
    {
        score -= 0.28;
    }
    if lower.contains("version number") {
        score -= 0.22;
    }

    if let Some(path) = file_path {
        let path_lower = path.to_ascii_lowercase();
        if path_lower.ends_with("cargo.lock")
            || path_lower.ends_with("package-lock.json")
            || path_lower.ends_with("yarn.lock")
            || path_lower.ends_with("pnpm-lock.yaml")
            || path_lower.ends_with("go.sum")
            || path_lower.ends_with("composer.lock")
            || path_lower.ends_with("gemfile.lock")
        {
            score -= 0.45;
        } else if path_lower.ends_with("cargo.toml")
            || path_lower.ends_with("package.json")
            || path_lower.ends_with("pyproject.toml")
            || path_lower.ends_with("go.mod")
            || path_lower.ends_with("pom.xml")
            || path_lower.ends_with("build.gradle")
            || path_lower.ends_with("build.gradle.kts")
            || path_lower.ends_with("pubspec.yaml")
        {
            if lower.contains("version")
                || lower.contains("dependencies")
                || lower.contains("dependency")
            {
                score -= 0.25;
            } else {
                score -= 0.08;
            }
        } else if path_lower.contains("/src/") || path_lower.ends_with(".rs") {
            score += 0.05;
        }
    }

    score
}

fn is_low_information_subject(subject: &str) -> bool {
    let lower = subject.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return true;
    }
    if matches!(
        lower.as_str(),
        "add version number"
            | "add version"
            | "update version"
            | "update versions"
            | "version bump"
            | "bump version"
            | "update dependencies"
            | "mixed"
    ) {
        return true;
    }
    if lower.starts_with("update cargo.lock")
        || lower.starts_with("update cargo lock")
        || lower.starts_with("update lockfile")
    {
        return true;
    }

    let words = lower.split_whitespace().collect::<Vec<_>>();
    if words.len() == 1 {
        return matches!(
            words[0],
            "refactor"
                | "fix"
                | "update"
                | "improve"
                | "add"
                | "build"
                | "implement"
                | "feature"
                | "chore"
                | "changes"
                | "cleanup"
        );
    }

    false
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
    subject = simplify_subject_phrase(&subject);

    if subject.is_empty() {
        return None;
    }

    Some(subject)
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
    let value = value.trim();
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut out = String::new();
    for token in value.split_whitespace() {
        let next = if out.is_empty() {
            token.to_string()
        } else {
            format!("{out} {token}")
        };
        if next.chars().count() > max_chars {
            break;
        }
        out = next;
    }

    if !out.is_empty() {
        return out;
    }

    value
        .chars()
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_string()
}

fn simplify_subject_phrase(subject: &str) -> String {
    let mut out = subject.trim().to_string();
    if out.is_empty() {
        return out;
    }

    for prefix in [
        "implement a feature that ",
        "implement feature that ",
        "add a feature that ",
        "add feature that ",
        "create a feature that ",
        "create feature that ",
    ] {
        if out.to_ascii_lowercase().starts_with(prefix) {
            out = out[prefix.len()..].trim_start().to_string();
            break;
        }
    }

    for (from, to) in [
        ("composes ", "compose "),
        ("creates ", "create "),
        ("adds ", "add "),
        ("builds ", "build "),
        ("updates ", "update "),
        ("fixes ", "fix "),
        ("refactors ", "refactor "),
        ("improves ", "improve "),
        ("supports ", "support "),
        ("prevents ", "prevent "),
        ("guards ", "guard "),
        ("implements ", "implement "),
    ] {
        if out.to_ascii_lowercase().starts_with(from) {
            out = format!("{to}{}", out[from.len()..].trim_start());
            break;
        }
    }

    for prefix in [
        "extract/simplify/reorganize ",
        "extract, simplify, reorganize ",
        "extract simplify reorganize ",
        "extract/reorganize ",
    ] {
        if out.to_ascii_lowercase().starts_with(prefix) {
            out = format!("refactor {}", out[prefix.len()..].trim_start());
            break;
        }
    }

    for (from, to) in [
        (" and creates ", " and create "),
        (" and adds ", " and add "),
        (" and builds ", " and build "),
        (" and updates ", " and update "),
        (" and fixes ", " and fix "),
        (" and refactors ", " and refactor "),
        (" and improves ", " and improve "),
        (" and supports ", " and support "),
        (" and prevents ", " and prevent "),
        (" and guards ", " and guard "),
    ] {
        out = out.replace(from, to);
    }

    out
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
    let tokens = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let contains_word = |needle: &str| tokens.iter().any(|token| *token == needle);

    let has_analysis = contains_word("analysis") || contains_word("analyses");
    let has_report = contains_word("report") || contains_word("reports");
    let has_chunk = contains_word("chunk") || contains_word("chunks");
    let has_partial = contains_word("partial") || contains_word("partials");
    let has_reduce = contains_word("reduce")
        || contains_word("reducer")
        || contains_word("reducers")
        || contains_word("reduction")
        || contains_word("reductions");
    let has_synthesis = contains_word("synthesis")
        || contains_word("synthesize")
        || contains_word("synthesized")
        || contains_word("synthesizes");

    (has_partial && has_analysis)
        || (has_chunk && has_analysis)
        || (lower.contains("consolidated report"))
        || (lower.contains("adaptive reducer"))
        || (has_reduce && (has_analysis || has_report || has_chunk || has_partial))
        || (has_synthesis && (has_analysis || has_report || has_chunk))
}

fn normalize_reduce_summary(raw: &str) -> Option<String> {
    let summary = sanitize_sentence(raw).trim().to_string();
    if summary.is_empty() {
        return None;
    }
    if looks_like_reducer_meta(&summary) {
        return None;
    }

    let summary = summary
        .trim_end_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .trim()
        .to_string();
    if summary.len() < 8 {
        return None;
    }

    Some(summary)
}

fn recover_commit_message_from_reduce(
    generated: &ReduceModelOutput,
    items: &[ChangeItem],
) -> Option<String> {
    let description_candidate = sanitize_commit_subject(&generated.commit_message)
        .or_else(|| sanitize_commit_subject(&generated.summary))?;
    if looks_like_reducer_meta(&description_candidate)
        || is_low_information_subject(&description_candidate)
    {
        return None;
    }

    let (commit_type_candidate, scope_from_header) =
        parse_conventional_commit(&generated.commit_message)
            .map(|(kind, scope, _)| (kind, scope.and_then(sanitize_scope)))
            .or_else(|| {
                parse_type_prefixed_header(&generated.commit_message).map(|(kind, _)| (kind, None))
            })
            .or_else(|| {
                parse_type_prefixed_header(&generated.summary).map(|(kind, _)| (kind, None))
            })
            .unwrap_or_else(|| {
                (
                    dominant_type_tag(items).map_or("chore", type_tag_prefix),
                    None,
                )
            });
    let commit_type = reconcile_commit_type_with_items(commit_type_candidate, items);

    let description =
        trim_redundant_commit_type_prefix(commit_type, &decapitalize_first(&description_candidate));
    if description.is_empty() {
        return None;
    }
    let scope = if commit_type != commit_type_candidate {
        dominant_scope(items).or(scope_from_header)
    } else {
        scope_from_header.or_else(|| dominant_scope(items))
    };

    Some(match scope {
        Some(scope) => format!("{commit_type}({scope}): {description}"),
        None => format!("{commit_type}: {description}"),
    })
}

fn recover_summary_from_reduce(
    generated: &ReduceModelOutput,
    commit_message: &str,
    stats: &DiffStats,
) -> Option<String> {
    if let Some(summary) = normalize_reduce_summary(&generated.summary) {
        return Some(summary);
    }

    let cleaned = sanitize_sentence(&generated.summary).trim().to_string();
    if !cleaned.is_empty() && cleaned.len() >= 8 && !looks_like_reducer_meta(&cleaned) {
        return Some(ensure_terminal_punctuation(cleaned));
    }

    summary_from_commit_message(commit_message, stats)
}

fn summary_from_commit_message(commit_message: &str, stats: &DiffStats) -> Option<String> {
    let (_, _, desc) = parse_conventional_commit(commit_message)?;
    let description = desc.trim();
    if description.is_empty() {
        return None;
    }

    let file_count = stats.files_changed.max(1);
    let noun = if file_count == 1 { "file" } else { "files" };
    Some(format!(
        "{} across {file_count} {noun}.",
        capitalize_first(description)
    ))
}

fn ensure_terminal_punctuation(mut value: String) -> String {
    let trimmed = value.trim_end();
    if trimmed.ends_with(['.', '!', '?']) {
        return trimmed.to_string();
    }
    value = trimmed.to_string();
    value.push('.');
    value
}

fn normalize_commit_message(raw: &str) -> Option<String> {
    let header = raw.lines().next()?.trim();
    if header.is_empty() {
        return None;
    }

    if let Some((kind, scope, desc)) = parse_conventional_commit(header) {
        let normalized_desc = sanitize_commit_subject(desc)?;
        if looks_like_reducer_meta(&normalized_desc) || is_low_information_subject(&normalized_desc)
        {
            return None;
        }
        let description =
            trim_redundant_commit_type_prefix(kind, &decapitalize_first(&normalized_desc));
        if description.is_empty() {
            return None;
        }
        let mut out = kind.to_string();
        if let Some(scope) = scope.and_then(sanitize_scope) {
            out.push('(');
            out.push_str(&scope);
            out.push(')');
        }
        out.push_str(": ");
        out.push_str(&description);
        return Some(out);
    }

    if let Some((kind, desc)) = parse_type_prefixed_header(header) {
        let normalized_desc = sanitize_commit_subject(desc)?;
        if looks_like_reducer_meta(&normalized_desc) || is_low_information_subject(&normalized_desc)
        {
            return None;
        }
        let description =
            trim_redundant_commit_type_prefix(kind, &decapitalize_first(&normalized_desc));
        if description.is_empty() {
            return None;
        }
        return Some(format!("{kind}(autocommit): {description}"));
    }

    None
}

fn env_budget_override(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        trimmed.parse::<usize>().ok().filter(|value| *value > 0)
    })
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
    fn parse_analyze_output_rejects_dangling_fragment() {
        let raw = r#"{"summary":"Route embeddings based on cosine similarity","bucket":"Patch","type_tag":"Refactor","title":"Update embedding gate","intent":"Determine dispatch route based on the"}"#;
        let err = parse_analyze_output(raw).expect_err("dangling fragment should fail");
        assert!(
            err.to_string()
                .contains("analyze output has dangling fragment")
        );
    }

    #[test]
    fn ends_with_dangling_joiner_detects_common_cutoff_suffixes() {
        assert!(ends_with_dangling_joiner(
            "Determine dispatch route based on the"
        ));
        assert!(!ends_with_dangling_joiner(
            "Determine dispatch route using cosine similarity."
        ));
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
    fn normalize_commit_message_rewrites_extract_simplify_reorganize_subject() {
        let raw = "refactor: extract/simplify/reorganize model reduction logic";
        let normalized = normalize_commit_message(raw).expect("commit should normalize");
        assert_eq!(normalized, "refactor: model reduction logic");
    }

    #[test]
    fn normalize_commit_message_drops_placeholder_scope() {
        let raw = "feat(scope): add cache key with no version";
        let normalized = normalize_commit_message(raw).expect("commit should normalize");
        assert_eq!(normalized, "feat: add cache key with no version");
    }

    #[test]
    fn normalize_reduce_summary_rejects_meta_phrasing() {
        let raw = "Consolidate 9 partial analyses into one consolidated report.";
        assert!(normalize_reduce_summary(raw).is_none());
    }

    #[test]
    fn normalize_reduce_summary_keeps_plain_change_summary() {
        let raw = "Improve commit message quality across 4 files.";
        assert_eq!(
            normalize_reduce_summary(raw).as_deref(),
            Some("Improve commit message quality across 4 files.")
        );
    }

    #[test]
    fn normalize_reduce_summary_allows_non_meta_reduce_wording() {
        let raw = "Preserve valid commit text when reduce JSON is partially malformed.";
        assert_eq!(
            normalize_reduce_summary(raw).as_deref(),
            Some("Preserve valid commit text when reduce JSON is partially malformed.")
        );
    }

    #[test]
    fn parse_reduce_output_keeps_commit_when_summary_missing() {
        let raw = r#"{"commit_message":"refactor(cli): tighten reduce output acceptance","risk_level":"low"}"#;
        let parsed = parse_reduce_output(raw).expect("reduce parse should keep commit");
        assert_eq!(
            normalize_commit_message(&parsed.commit_message).as_deref(),
            Some("refactor(cli): tighten reduce output acceptance")
        );
        assert!(parsed.summary.is_empty());
    }

    #[test]
    fn parse_reduce_output_salvages_freeform_commit_and_summary() {
        let raw = "\
Commit: feat(core): preserve valid reduce commit output
Summary: Preserve valid commit text when reduce JSON is partially malformed.
";
        let parsed = parse_reduce_output(raw).expect("freeform reduce output should salvage");
        assert_eq!(
            normalize_commit_message(&parsed.commit_message).as_deref(),
            Some("feat(core): preserve valid reduce commit output")
        );
        assert_eq!(
            normalize_reduce_summary(&parsed.summary).as_deref(),
            Some("Preserve valid commit text when reduce JSON is partially malformed.")
        );
    }

    #[test]
    fn recover_commit_message_from_reduce_repairs_non_conventional_subject() {
        let generated = ReduceModelOutput {
            commit_message: "extract/simplify/reorganize model reduction logic".to_string(),
            summary: String::new(),
            risk_level: "low".to_string(),
            risk_notes: Vec::new(),
        };
        let items = vec![sample_item(
            TypeTag::Refactor,
            "Refactor model reduction logic",
            "Refactor model reduction logic",
            "crates/llama-runtime/src/model.rs",
            0.9,
        )];

        let repaired =
            recover_commit_message_from_reduce(&generated, &items).expect("should repair commit");
        assert_eq!(repaired, "refactor(llama-runtime): model reduction logic");
    }

    #[test]
    fn recover_commit_message_from_reduce_ignores_placeholder_scope() {
        let generated = ReduceModelOutput {
            commit_message: "feat(scope): add cache key with no version".to_string(),
            summary: String::new(),
            risk_level: "low".to_string(),
            risk_notes: Vec::new(),
        };
        let items = vec![sample_item(
            TypeTag::Feat,
            "Add cache key with no version",
            "Share cache keys between commands",
            "crates/cli/src/cmd/report_cache.rs",
            0.88,
        )];

        let repaired =
            recover_commit_message_from_reduce(&generated, &items).expect("should repair commit");
        assert_eq!(repaired, "feat(cli): add cache key with no version");
    }

    #[test]
    fn recover_commit_message_from_reduce_reconciles_misaligned_type() {
        let generated = ReduceModelOutput {
            commit_message: "feat(cli): improve model reduction stability".to_string(),
            summary: String::new(),
            risk_level: "low".to_string(),
            risk_notes: Vec::new(),
        };
        let items = vec![
            sample_item(
                TypeTag::Refactor,
                "Refactor model reduction logic",
                "Refactor model reduction flow",
                "crates/llama-runtime/src/model.rs",
                0.93,
            ),
            sample_item(
                TypeTag::Fix,
                "Fix reduce item sampling",
                "Fix change-line sampling behavior",
                "crates/llama-runtime/src/model.rs",
                0.81,
            ),
        ];

        let repaired = recover_commit_message_from_reduce(&generated, &items)
            .expect("should reconcile commit type");
        assert!(repaired.starts_with("refactor(llama-runtime):"));
    }

    #[test]
    fn recover_summary_from_reduce_uses_commit_when_summary_is_meta() {
        let generated = ReduceModelOutput {
            commit_message: "refactor(llama-runtime): model reduction logic".to_string(),
            summary: "Consolidate partial analyses into one report".to_string(),
            risk_level: "medium".to_string(),
            risk_notes: Vec::new(),
        };
        let stats = DiffStats {
            files_changed: 3,
            lines_changed: 120,
            hunks: 10,
            binary_files: 0,
        };

        let summary = recover_summary_from_reduce(
            &generated,
            "refactor(llama-runtime): model reduction logic",
            &stats,
        )
        .expect("should recover summary");
        assert_eq!(summary, "Model reduction logic across 3 files.");
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
    fn synthesize_fallback_commit_message_strips_redundant_refactor_prefix() {
        let items = vec![sample_item(
            TypeTag::Refactor,
            "Refactor model labels logic",
            "Refactor model labels logic",
            "crates/llama-runtime/src/model.rs",
            0.9,
        )];

        let commit = synthesize_fallback_commit_message(&items, 1);
        assert_eq!(commit, "refactor(llama-runtime): model labels logic");
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
        let acceptable = [
            "Guard backend config export and Consolidate runtime prompt handling across 2 files.",
            "Guard backend config export and consolidate runtime prompt handling across 2 files.",
            "Consolidate runtime prompt handling and Guard backend config export across 2 files.",
            "Consolidate runtime prompt handling and guard backend config export across 2 files.",
        ];
        assert!(
            acceptable.contains(&summary.as_str()),
            "unexpected summary: {summary}"
        );
    }

    #[test]
    fn synthesize_fallback_summary_merges_shared_refactor_prefix() {
        let items = vec![
            sample_item(
                TypeTag::Refactor,
                "Refactor model labels logic",
                "Refactor model labels logic",
                "crates/llama-runtime/src/model.rs",
                0.91,
            ),
            sample_item(
                TypeTag::Refactor,
                "Refactor commit message formatting",
                "Refactor commit message formatting",
                "crates/cli/src/cmd/commit.rs",
                0.89,
            ),
        ];
        let stats = DiffStats {
            files_changed: 2,
            lines_changed: 50,
            hunks: 4,
            binary_files: 0,
        };

        let summary = synthesize_fallback_summary(&items, &stats);
        assert_eq!(
            summary,
            "Refactor model labels logic and commit message formatting across 2 files."
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
    fn commit_synthesis_downranks_manifest_version_noise() {
        let items = vec![
            sample_item(
                TypeTag::Feat,
                "Add version number",
                "Update package version",
                "crates/llama-runtime/Cargo.toml",
                0.95,
            ),
            sample_item(
                TypeTag::Refactor,
                "Refactor model reduction logic",
                "Simplify reduce candidate handling",
                "crates/llama-runtime/src/model.rs",
                0.82,
            ),
        ];

        let commit = synthesize_fallback_commit_message(&items, 2);
        assert_eq!(commit, "refactor(llama-runtime): model reduction logic");
    }

    #[test]
    fn reconcile_model_labels_demotes_feature_to_refactor_when_text_says_refactor() {
        let (bucket, tag) = reconcile_model_labels(
            &ChangeBucket::Feature,
            &TypeTag::Feat,
            "Refactor commit message generation and creation",
            "Improve commit message composition",
        );
        assert_eq!(tag, TypeTag::Refactor);
        assert_eq!(bucket, ChangeBucket::Patch);
    }

    #[test]
    fn clamp_chars_keeps_word_boundaries() {
        assert_eq!(clamp_chars("alpha beta gamma", 11), "alpha beta");
    }

    #[test]
    fn simplify_subject_phrase_strips_feature_boilerplate() {
        assert_eq!(
            simplify_subject_phrase("Implement a feature that composes commit messages"),
            "compose commit messages"
        );
    }

    #[test]
    fn low_information_subject_detection_flags_single_generic_verbs() {
        assert!(is_low_information_subject("Refactor"));
        assert!(!is_low_information_subject(
            "Refactor ChangeItem formatting"
        ));
    }

    #[test]
    fn mapped_change_item_id_uses_file_path_shape() {
        assert_eq!(
            mapped_change_item_id("crates/core/src/llm/prompts.rs", 0),
            "crates/core/src/llm/prompts.rs#chunk-1"
        );
    }

    #[test]
    fn sampled_indices_prefers_confident_diverse_partials() {
        let partials = vec![
            PartialReport {
                summary: "low confidence core".to_string(),
                items: vec![sample_item(
                    TypeTag::Refactor,
                    "Refactor core plumbing",
                    "Refactor core plumbing",
                    "crates/core/src/lib.rs",
                    0.20,
                )],
            },
            PartialReport {
                summary: "high confidence cli".to_string(),
                items: vec![sample_item(
                    TypeTag::Feat,
                    "Add interactive commit flow",
                    "Add interactive commit flow",
                    "crates/cli/src/cmd/commit.rs",
                    0.95,
                )],
            },
            PartialReport {
                summary: "high confidence runtime".to_string(),
                items: vec![sample_item(
                    TypeTag::Fix,
                    "Fix runtime reduce selection",
                    "Fix runtime reduce selection",
                    "crates/llama-runtime/src/model.rs",
                    0.91,
                )],
            },
            PartialReport {
                summary: "mid confidence cli".to_string(),
                items: vec![sample_item(
                    TypeTag::Refactor,
                    "Refactor commit preview styles",
                    "Refactor commit preview styles",
                    "crates/cli/src/cmd/commit.rs",
                    0.62,
                )],
            },
        ];

        let picked = sampled_indices(&partials, 2);
        assert_eq!(picked, vec![1, 2]);
    }

    #[test]
    fn compact_diff_for_prompt_keeps_structure_and_changes() {
        let diff = "\
diff --git a/crates/cli/src/cmd/commit.rs b/crates/cli/src/cmd/commit.rs\n\
index 111..222 100644\n\
--- a/crates/cli/src/cmd/commit.rs\n\
+++ b/crates/cli/src/cmd/commit.rs\n\
@@ -1,4 +1,8 @@\n\
-fn old_name() {}\n\
+fn new_name() {}\n\
 context line one\n\
 context line two\n\
@@ -20,2 +24,6 @@\n\
-let a = 1;\n\
+let a = 2;\n\
+let b = 3;\n\
 impl CommitComposer {}\n\
";

        let compacted = compact_diff_for_prompt(diff, 320);
        assert!(compacted.contains("diff --git"));
        assert!(compacted.contains("@@ -1,4 +1,8 @@"));
        assert!(compacted.contains("+fn new_name() {}") || compacted.contains("+let b = 3;"));
        assert!(compacted.contains("diff truncated for budget"));
    }
}
