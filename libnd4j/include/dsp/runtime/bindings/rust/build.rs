use std::env;
use std::path::PathBuf;

fn main() {
    // ── SDX graph-runtime library (libnd4jcpu / libsdx_cpu / …) ─────────────
    // Explicit override via environment variable
    if let Ok(lib_dir) = env::var("SDX_RUNTIME_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
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

    // ── SDX LLM library (libsdx_llm.so) — only when `llm` feature is active ──
    // `CARGO_FEATURE_LLM` is set by Cargo when `features = ["llm"]` is active.
    if env::var_os("CARGO_FEATURE_LLM").is_some() {
        let llm_lib_dir = env::var("SDX_LLM_LIB_DIR")
            .map(PathBuf::from)
            .or_else(|_| {
                env::var("SDX_LLM_AOT_HOME").map(|h| PathBuf::from(h).join("lib"))
            })
            .unwrap_or_else(|_| {
                let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
                PathBuf::from(manifest).join("../../lib")
            });

        if llm_lib_dir.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                llm_lib_dir.canonicalize()
                    .unwrap_or(llm_lib_dir.clone())
                    .display()
            );
        }

        println!("cargo:rerun-if-env-changed=SDX_LLM_LIB_DIR");
        println!("cargo:rerun-if-env-changed=SDX_LLM_AOT_HOME");
    }

    println!("cargo:rerun-if-env-changed=SDX_RUNTIME_LIB_DIR");
}
