use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn repo_key(repo_root: &str, origin_url: &str) -> String {
    let mut hasher = DefaultHasher::new();
    repo_root.hash(&mut hasher);
    origin_url.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_changes_with_origin() {
        let a = repo_key("/tmp/repo", "git@example/a.git");
        let b = repo_key("/tmp/repo", "git@example/b.git");
        assert_ne!(a, b);
    }
}
