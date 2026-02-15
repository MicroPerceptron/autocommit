use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::pipeline::fanout;
use autocommit_core::types::{
    AnalysisReport, ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};
use autocommit_core::{CoreError, DispatchRoute};

struct BatchedOnlyEngine;

impl LlmEngine for BatchedOnlyEngine {
    fn analyze_chunk(&self, _chunk: &DiffChunk) -> Result<PartialReport, CoreError> {
        panic!("fanout should use batched hook before per-chunk fallback")
    }

    fn analyze_chunks_batched(
        &self,
        chunks: &[DiffChunk],
    ) -> Option<Result<Vec<PartialReport>, CoreError>> {
        let reports = chunks
            .iter()
            .map(|chunk| PartialReport {
                summary: format!("batched {}", chunk.path),
                items: vec![ChangeItem {
                    id: chunk.path.clone(),
                    bucket: ChangeBucket::Patch,
                    type_tag: TypeTag::Fix,
                    title: "batched".to_string(),
                    intent: "batched path".to_string(),
                    files: vec![FileRef {
                        path: chunk.path.clone(),
                        status: FileStatus::Modified,
                        ranges: chunk.ranges.clone(),
                    }],
                    confidence: 0.8,
                }],
            })
            .collect();

        Some(Ok(reports))
    }

    fn reduce_report(
        &self,
        _partials: &[PartialReport],
        _decision: &DispatchDecision,
        _stats: &DiffStats,
    ) -> Result<AnalysisReport, CoreError> {
        Ok(AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "fix: test".to_string(),
            summary: "summary".to_string(),
            items: Vec::new(),
            risk: RiskReport {
                level: "low".to_string(),
                notes: Vec::new(),
            },
            stats: DiffStats::default(),
            dispatch: DispatchDecision {
                route: DispatchRoute::DraftOnly,
                reason_codes: vec!["test".to_string()],
                estimated_cost_tokens: 0,
            },
        })
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, CoreError> {
        Ok(vec![0.1, 0.2])
    }
}

#[test]
fn fanout_prefers_batched_hook_when_available() {
    let chunks = vec![
        DiffChunk {
            path: "src/a.rs".to_string(),
            text: "@@ -1 +1 @@".to_string(),
            ranges: Vec::new(),
            estimated_tokens: 4,
        },
        DiffChunk {
            path: "src/b.rs".to_string(),
            text: "@@ -1 +1 @@".to_string(),
            ranges: Vec::new(),
            estimated_tokens: 4,
        },
    ];

    let reports = fanout::analyze_chunks(&BatchedOnlyEngine, &chunks).expect("fanout result");
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].summary, "batched src/a.rs");
    assert_eq!(reports[1].summary, "batched src/b.rs");
}
