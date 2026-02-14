use autocommit_core::diff::features::DiffFeatures;
use autocommit_core::dispatch::policy;

pub fn run(_args: &[String]) -> String {
    let sample = DiffFeatures {
        files_changed: 4,
        lines_changed: 220,
        hunks: 8,
        binary_files: 0,
        risky_paths: 0,
    };

    let decision = policy::decide(&sample, Some(0.81));
    format!(
        "route={:?} reasons={} estimated_tokens={}\n",
        decision.route,
        decision.reason_codes.join(","),
        decision.estimated_cost_tokens
    )
}
