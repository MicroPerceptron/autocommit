use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DispatchRoute {
    FormatOnly,
    DraftOnly,
    DraftThenReduce,
    FullPipeline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DispatchDecision {
    pub route: DispatchRoute,
    pub reason_codes: Vec<String>,
    pub estimated_cost_tokens: u32,
}
