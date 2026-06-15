use std::env;
use std::path::PathBuf;

fn main() {
    // Explicit override via environment variable
    if let Ok(lib_dir) = env::var("SDX_RUNTIME_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        return;
    }

    // Fall back to SDK layout: ../../lib relative to Cargo.toml
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let sdk_lib = PathBuf::from(&manifest_dir).join("../../lib");
    if sdk_lib.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            sdk_lib.canonicalize().unwrap().display()
        );
    }
}
