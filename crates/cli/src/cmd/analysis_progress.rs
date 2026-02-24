use indicatif::{ProgressBar, ProgressStyle};

use autocommit_core::progress::{ProgressCallback, ProgressEvent, ProgressStage};
use std::sync::{Arc, Mutex};

pub struct AnalysisProgress {
    bar: ProgressBar,
    label: String,
    callback: ProgressCallback,
}

impl AnalysisProgress {
    pub fn new(diff_text: &str, label: &str) -> Self {
        let total_chunks = autocommit_core::diff::collect::collect(diff_text).len();
        // Steps: dispatch(1) + merge(1) + chunks(N) + reduce(1) = N + 3
        let total_steps = (total_chunks as u64).saturating_add(3).max(1);
        let bar = ProgressBar::new(total_steps);
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner())
                .tick_strings(&["-", "\\", "|", "/"]),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(80));
        let label = label.to_string();
        bar.set_message(render_message(&label, "preparing", 0, total_steps));

        let callback: ProgressCallback = {
            let bar = bar.clone();
            let callback_label = label.clone();
            let state = Arc::new(Mutex::new(ProgressState {
                last_pos: 0,
                total_chunks,
                total_steps,
            }));
            Arc::new(move |event: ProgressEvent| {
                if let Ok(mut guard) = state.lock() {
                    update_progress(&bar, &mut guard, &callback_label, event);
                }
            }) as ProgressCallback
        };

        Self {
            bar,
            label,
            callback,
        }
    }

    pub fn callback(&self) -> ProgressCallback {
        self.callback.clone()
    }

    pub fn finish(self) {
        self.bar.finish_with_message(format!("[ok] {}", self.label));
    }
}

struct ProgressState {
    last_pos: u64,
    total_chunks: usize,
    total_steps: u64,
}

fn update_progress(
    bar: &ProgressBar,
    state: &mut ProgressState,
    label: &str,
    event: ProgressEvent,
) {
    match event.stage {
        ProgressStage::Dispatch => {
            advance_to(bar, state, 1);
            bar.set_message(render_message(
                label,
                "dispatching",
                state.last_pos,
                state.total_steps,
            ));
        }
        ProgressStage::Embedding => {
            // Embedding events arrive during dispatch; keep position at 1.
            bar.set_message(render_message(
                label,
                "embedding",
                state.last_pos,
                state.total_steps,
            ));
        }
        ProgressStage::Merging { from, to } => {
            // Recalculate total steps based on the merged chunk count.
            // Steps: dispatch(1) + merge(1) + merged_chunks(to) + reduce(1)
            let new_total = (to as u64).saturating_add(3).max(1);
            state.total_steps = new_total;
            state.total_chunks = to;
            bar.set_length(new_total);
            advance_to(bar, state, 2);
            bar.set_message(render_message(
                label,
                &format!("merged {from}\u{2192}{to} chunks"),
                state.last_pos,
                state.total_steps,
            ));
        }
        ProgressStage::Analyze { completed, total } => {
            let total = total.max(state.total_chunks);
            // Position: dispatch(1) + merge(1) + completed
            let pos = 2 + completed.min(total) as u64;
            advance_to(bar, state, pos);
            bar.set_message(render_message(
                label,
                &format!("analyzing chunks {}/{}", completed.min(total), total),
                state.last_pos,
                state.total_steps,
            ));
        }
        ProgressStage::Reduce => {
            advance_to(bar, state, state.total_steps);
            bar.set_message(render_message(
                label,
                "reducing",
                state.last_pos,
                state.total_steps,
            ));
        }
        ProgressStage::DraftSynthesis => {
            advance_to(bar, state, state.total_steps);
            bar.set_message(render_message(
                label,
                "synthesizing",
                state.last_pos,
                state.total_steps,
            ));
        }
    }
}

fn advance_to(bar: &ProgressBar, state: &mut ProgressState, pos: u64) {
    let pos = pos.min(state.total_steps);
    if pos >= state.last_pos {
        state.last_pos = pos;
        bar.set_position(pos);
    }
}

fn render_message(label: &str, stage: &str, pos: u64, total: u64) -> String {
    let pct = if total == 0 {
        0
    } else {
        (((pos as f64) / (total as f64)) * 100.0).round() as u64
    };
    format!("{label} [{pct}%] {stage}")
}
