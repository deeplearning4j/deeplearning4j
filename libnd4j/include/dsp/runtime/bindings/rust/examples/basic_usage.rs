use sdx_runtime::{runtime_abi_version, Runtime};

fn main() {
    println!("ABI version: {}", runtime_abi_version());

    let runtime = Runtime::create(None).expect("Failed to create runtime");
    let _model = runtime
        .load_model("path/to/model.sdx", None)
        .expect("Failed to load model");

    println!("Runtime created successfully.");
}
