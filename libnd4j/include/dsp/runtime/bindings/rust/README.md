# SDX Rust Binding

This crate wraps `dsp_runtime_c.h` with idiomatic Rust ownership for:

- `Runtime`
- `Model`
- `Context`

Linking mode:

- Default links `nd4jcpu`
- `--features cuda` links `nd4jcuda`
- `--features amd` links `nd4jamd`

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
