use std::fs;

use llama_runtime::Engine;
use llama_runtime::state;

#[test]
fn backend_raii_refcount_balances() {
    assert_eq!(Engine::backend_refcount(), 0);
    let engine_a = Engine::new("a").expect("engine a");
    assert_eq!(Engine::backend_refcount(), 1);

    {
        let engine_b = Engine::new("b").expect("engine b");
        assert_eq!(Engine::backend_refcount(), 2);
        drop(engine_b);
    }

    assert_eq!(Engine::backend_refcount(), 1);
    drop(engine_a);
    assert_eq!(Engine::backend_refcount(), 0);
}

#[test]
fn state_load_save_error_paths_are_exposed() {
    let missing = std::env::temp_dir().join("autocommit-missing-state.bin");
    let _ = fs::remove_file(&missing);
    assert!(state::load_state(&missing).is_err());

    let saved = std::env::temp_dir().join(format!(
        "autocommit-state-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_millis()
    ));

    state::save_state(&saved, b"state-bytes").expect("save state");
    let loaded = state::load_state(&saved).expect("load state");
    assert_eq!(loaded, b"state-bytes");

    let _ = fs::remove_file(saved);
}
