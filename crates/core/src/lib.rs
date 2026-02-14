pub const APP_NAME: &str = "autocommit";

pub fn llama_native_build_info() -> Option<(&'static str, &'static str)> {
    #[cfg(feature = "llama-native")]
    {
        return Some((llama_sys::cmake_build_dir(), llama_sys::cmake_install_dir()));
    }

    #[cfg(not(feature = "llama-native"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_name_is_set() {
        assert_eq!(APP_NAME, "autocommit");
    }

    #[test]
    fn llama_info_is_absent_without_feature() {
        if !cfg!(feature = "llama-native") {
            assert_eq!(llama_native_build_info(), None);
        }
    }
}
