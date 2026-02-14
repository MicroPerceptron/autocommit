pub fn cmake_build_dir() -> &'static str {
    env!("LLAMA_CPP_BUILD_DIR")
}

pub fn cmake_install_dir() -> &'static str {
    env!("LLAMA_CPP_INSTALL_DIR")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmake_paths_are_populated() {
        assert!(!cmake_build_dir().is_empty());
        assert!(!cmake_install_dir().is_empty());
    }
}
