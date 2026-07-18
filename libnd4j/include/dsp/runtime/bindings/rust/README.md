# SDX Rust Binding

This crate wraps `dsp_runtime_c.h` with idiomatic Rust ownership for:

- `Runtime`
- `Model`
- `Context`

Linking mode:

- Default links `nd4jcpu`
- `--features cuda` links `nd4jcuda`
- `--features amd` links `nd4jamd`
- `--features standalone` links `libsdx_cpu` / `libsdx_cuda` (JVM-free AOT)

Usage sketch:

```rust
use sdx_runtime::{Runtime, sdx_model_options_t, sdx_run_options_t};

let runtime = Runtime::create(None)?;
let model = runtime.load_model("/path/to/model.sdz", Some(sdx_model_options_t::default()))?;
let context = model.create_context(None)?;
context.run(&[], &[], Some(sdx_run_options_t::default()))?;
let report = context.execution_report()?;
println!("{}", report.execution_time_ns);
```

## LLM / VLM / STT surface (`features = ["llm"]`)

The `llm` feature gates the `sdx_runtime::llm` module, which wraps
`libsdx_llm.so` — the AOT (GraalVM native-image) LLM surface defined in
`nd4j/sdx-aot/include/sdx_llm_c.h` (see ADR 0109).  No JVM required.

Add to `Cargo.toml`:

```toml
[dependencies]
sdx-runtime = { path = "…", features = ["llm"] }
```

Set `SDX_LLM_AOT_HOME` at build time so `build.rs` can locate `libsdx_llm.so`:

```bash
SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8 cargo build --release --features llm
```

Quick start:

```rust
use sdx_runtime::llm::{ensure_native_lib_dir, LlmRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ensure_native_lib_dir();  // sets SDX_NATIVE_LIB_DIR from SDX_LLM_AOT_HOME
    let rt = LlmRuntime::new()?;
    println!("ABI version: {}", rt.abi_version());

    let model = rt.load_model(
        "/path/to/model.gguf",
        Some("/path/to/tokenizer.json"),
        Some(r#"{"maxNewTokens":64,"sampling":{"preset":"greedy"}}"#),
    )?;

    // Tokenize → detokenize round-trip
    let ids = model.tokenize("Hello, world!", false)?;
    let text = model.detokenize(&ids, true)?;
    println!("round-trip: {text}");

    // Text generation
    let response = model.generate("The capital of France is", None)?;
    println!("generated: {response}");

    let stats = model.last_result()?;
    println!("{:.1} tok/s", stats.tokens_per_sec);
    Ok(())
}
```

### Environment variables (LLM)

| Variable | Purpose |
|---|---|
| `SDX_LLM_AOT_HOME` | Root of the unpacked AOT SDK (`$HOME/lib/libsdx_llm.so`). |
| `SDX_LLM_LIB_DIR` | Override for the lib directory (takes precedence over `SDX_LLM_AOT_HOME/lib`). |
| `SDX_NATIVE_LIB_DIR` | Directory where `libsdx_llm.so` resolves bundled native libs; set automatically by `ensure_native_lib_dir()`. |

### Thread affinity

A `LlmRuntime` is **bound to the OS thread that created it**.  Create, use, and
drop from the same thread.  For concurrent generation create one `LlmRuntime`
per thread.
