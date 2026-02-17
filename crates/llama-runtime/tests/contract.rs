use autocommit_core::llm::traits::LlmEngine;
use autocommit_core::types::{
    ChangeBucket, ChangeItem, DiffChunk, DiffStats, DispatchDecision, DispatchRoute, FileRef,
    FileStatus, PartialReport, RiskReport, TypeTag,
};
use autocommit_core::{AnalysisReport, CoreError};
use llama_runtime::Engine;

struct MockEngine;

impl LlmEngine for MockEngine {
    fn analyze_chunk(&self, chunk: &DiffChunk) -> Result<PartialReport, CoreError> {
        Ok(PartialReport {
            summary: "mock".to_string(),
            items: vec![ChangeItem {
                id: "mock-item".to_string(),
                bucket: ChangeBucket::Patch,
                type_tag: TypeTag::Fix,
                title: "mock title".to_string(),
                intent: "mock intent".to_string(),
                files: vec![FileRef {
                    path: chunk.path.clone(),
                    status: FileStatus::Modified,
                    ranges: chunk.ranges.clone(),
                }],
                confidence: 0.9,
            }],
        })
    }

    fn reduce_report(
        &self,
        partials: &[PartialReport],
        decision: &DispatchDecision,
        stats: &DiffStats,
    ) -> Result<AnalysisReport, CoreError> {
        Ok(AnalysisReport {
            schema_version: "1.0".to_string(),
            commit_message: "fix: mock".to_string(),
            summary: format!("{} partials", partials.len()),
            items: partials.iter().flat_map(|p| p.items.clone()).collect(),
            risk: RiskReport {
                level: "low".to_string(),
                notes: vec![],
            },
            stats: stats.clone(),
            dispatch: decision.clone(),
        })
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, CoreError> {
        Ok(vec![0.9, 0.1])
    }
}

#[test]
fn mock_and_runtime_satisfy_llm_contract() {
    let chunk = DiffChunk {
        path: "src/lib.rs".to_string(),
        text: "@@ -1 +1 @@\n-a\n+b\n".to_string(),
        ranges: vec![],
        estimated_tokens: 8,
    };

    let decision = DispatchDecision {
        route: DispatchRoute::DraftOnly,
        reason_codes: vec!["test".to_string()],
        estimated_cost_tokens: 10,
    };
    let stats = DiffStats::default();

    let mock = MockEngine;
    for engine in [&mock as &dyn LlmEngine] {
        let partial = engine.analyze_chunk(&chunk).expect("partial");
        assert!(!partial.items.is_empty());
        let report = engine
            .reduce_report(&[partial], &decision, &stats)
            .expect("report");
        assert_eq!(report.schema_version, "1.0");
        assert!(!report.commit_message.is_empty());
    }

    let runtime = Engine::new("test").expect("runtime init");
    match runtime.analyze_chunk(&chunk) {
        Ok(partial) => {
            assert!(!partial.items.is_empty());
            let report = runtime
                .reduce_report(&[partial], &decision, &stats)
                .expect("runtime report");
            assert_eq!(report.schema_version, "1.0");
            assert!(!report.commit_message.is_empty());
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("runtime model")
                    || msg.contains("download")
                    || msg.contains("Hugging Face")
                    || msg.contains("not enabled")
                    || msg.contains("embedding model"),
                "unexpected runtime error: {err}"
            );
        }
    }
}
