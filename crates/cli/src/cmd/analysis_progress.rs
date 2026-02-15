use indicatif::{ProgressBar, ProgressStyle};

#[cfg(feature = "llama-native")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "llama-native")]
use llama_runtime::progress::{ProgressCallback, ProgressEvent, ProgressStage};

pub struct AnalysisProgress {
    bar: ProgressBar,
    #[cfg(feature = "llama-native")]
    callback: ProgressCallback,
}

impl AnalysisProgress {
    pub fn new(diff_text: &str) -> Self {
        let total_chunks = autocommit_core::diff::collect::collect(diff_text).len();
        let total_steps = (total_chunks as u64).saturating_add(2).max(1);
        let bar = ProgressBar::new(total_steps);
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg} {pos}/{len}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(80));
        bar.set_message("Preparing analysis".to_string());

        #[cfg(feature = "llama-native")]
        let callback: ProgressCallback = {
            let bar = bar.clone();
            let state = Arc::new(Mutex::new(ProgressState {
                last_pos: 0,
                total_chunks,
                total_steps,
            }));
            Arc::new(move |event: ProgressEvent| {
                if let Ok(mut guard) = state.lock() {
                    update_progress(&bar, &mut guard, event);
                }
            }) as ProgressCallback
        };

        #[cfg(feature = "llama-native")]
        {
            Self { bar, callback }
        }

        #[cfg(not(feature = "llama-native"))]
        {
            let _ = total_chunks;
            let _ = total_steps;
            Self { bar }
        }
    }

    #[cfg(feature = "llama-native")]
    pub fn callback(&self) -> ProgressCallback {
        self.callback.clone()
    }

    pub fn finish(self) {
        self.bar
            .finish_with_message("[ok] Generating analysis");
    }
}

#[cfg(feature = "llama-native")]
struct ProgressState {
    last_pos: u64,
    total_chunks: usize,
    total_steps: u64,
}

#[cfg(feature = "llama-native")]
fn update_progress(bar: &ProgressBar, state: &mut ProgressState, event: ProgressEvent) {
    match event.stage {
        ProgressStage::Embedding => {
            bar.set_message("Embedding".to_string());
            advance_to(bar, state, 1);
        }
        ProgressStage::Analyze { completed, total } => {
            let total = total.max(state.total_chunks);
            bar.set_message(format!("Analyzing {}/{}", completed.min(total), total));
            let pos = 1 + completed.min(total) as u64;
            advance_to(bar, state, pos);
        }
        ProgressStage::Reduce => {
            bar.set_message("Reducing".to_string());
            advance_to(bar, state, state.total_steps);
        }
    }
}

#[cfg(feature = "llama-native")]
fn advance_to(bar: &ProgressBar, state: &mut ProgressState, pos: u64) {
    let pos = pos.min(state.total_steps);
    if pos >= state.last_pos {
        state.last_pos = pos;
        bar.set_position(pos);
    }
}
