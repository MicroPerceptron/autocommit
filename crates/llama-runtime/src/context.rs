use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_CONTEXT_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Debug, Clone)]
pub struct RuntimeContext {
    pub id: usize,
    pub profile: String,
}

impl RuntimeContext {
    pub fn new(profile: &str) -> Self {
        Self {
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::SeqCst),
            profile: profile.to_string(),
        }
    }
}
