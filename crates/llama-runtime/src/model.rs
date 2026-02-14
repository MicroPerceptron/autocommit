use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use autocommit_core::CoreError;
use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};
use llama_sys::ffi;

use crate::context::RuntimeContext;
use crate::context_handle::ContextHandle;
use crate::embed::{EMBEDDING_MODEL_ENV, FALLBACK_MODEL_ENV, resolve_embedding_model_path};
use crate::error::RuntimeError;
use crate::model_handle::ModelHandle;

static BACKEND_REFCOUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct BackendGuard;

impl BackendGuard {
    fn acquire() -> Self {
        if BACKEND_REFCOUNT.fetch_add(1, Ordering::SeqCst) == 0 {
            // SAFETY: llama backend init is process-global and intended to be called before runtime usage.
            unsafe { ffi::llama_backend_init() };
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
    embedding_ctx: ContextHandle,
    model: Arc<ModelHandle>,
}

impl LoadedRuntime {
    fn load(model_path: &Path, profile: &str) -> Result<Self, RuntimeError> {
        let model = Arc::new(ModelHandle::load(model_path, profile)?);
        let embedding_ctx = ContextHandle::new_embedding(Arc::clone(&model))?;

        Ok(Self {
            generation_ctx: None,
            embedding_ctx,
            model,
        })
    }

    #[allow(dead_code)]
    fn generation_ctx(&mut self) -> Result<&mut ContextHandle, RuntimeError> {
        if self.generation_ctx.is_none() {
            self.generation_ctx = Some(ContextHandle::new_generation(Arc::clone(&self.model))?);
        }

        Ok(self
            .generation_ctx
            .as_mut()
            .expect("generation context just initialized"))
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        self.embedding_ctx.embed(text)
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

    fn embed_with_runtime(&self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        let model_path = self.runtime_model_path.as_deref().ok_or_else(|| {
            RuntimeError::Embed(format!(
                "embedding model path is not configured; set {EMBEDDING_MODEL_ENV} or {FALLBACK_MODEL_ENV}"
            ))
        })?;

        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| RuntimeError::Embed("runtime lock poisoned".to_string()))?;

        if guard.is_none() {
            *guard = Some(LoadedRuntime::load(model_path, &self.context.profile)?);
        }

        guard
            .as_mut()
            .expect("runtime just initialized")
            .embed(text)
    }
}

impl LlmEngine for Engine {
    fn analyze_chunk(&self, chunk: &DiffChunk) -> Result<PartialReport, CoreError> {
        let item = ChangeItem {
            id: format!("rt-{}-{}", self.context.id, chunk.path.replace('/', "_")),
            bucket: ChangeBucket::Patch,
            type_tag: TypeTag::Fix,
            title: format!("Analyze {}", chunk.path),
            intent: format!("Runtime analysis in profile {}", self.context.profile),
            files: vec![FileRef {
                path: chunk.path.clone(),
                status: FileStatus::Modified,
                ranges: chunk.ranges.clone(),
            }],
            confidence: 0.75,
        };

        Ok(PartialReport {
            summary: format!("{} lines scanned", chunk.text.lines().count()),
            items: vec![item],
        })
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

        Ok(AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "fix(runtime): reduce partial reports".to_string(),
            summary: format!(
                "Reduced {} partial reports (gpu_offload={})",
                partials.len(),
                Self::supports_gpu_offload()
            ),
            items,
            risk: RiskReport {
                level: "low".to_string(),
                notes: vec!["runtime engine scaffold".to_string()],
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
