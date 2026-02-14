fn main() {
    println!("{} CLI scaffold is ready.", autocommit_core::APP_NAME);
    if let Some((build_dir, install_dir)) = autocommit_core::llama_native_build_info() {
        println!("llama native build: {build_dir}");
        println!("llama native install: {install_dir}");
    }
}
