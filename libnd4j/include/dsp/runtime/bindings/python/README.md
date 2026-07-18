# SDX Python Binding (`ctypes`)

Module:

- `sdx_runtime.py`

Quick start:

```python
from sdx_runtime import SdxRuntime, ModelOptions, RunOptions

with SdxRuntime("libnd4jcpu.so") as rt:
    with rt.load_model("/path/to/model.sdz", ModelOptions()) as model:
        with model.create_context() as ctx:
            inputs = []
            outputs = []
            ctx.run(inputs, outputs, RunOptions())
            report = ctx.execution_report()
            print(report.execution_time_ns)
```

`ctx.run(...)` also accepts `numpy.ndarray` directly:

```python
import numpy as np
from sdx_runtime import SdxRuntime

x = np.random.rand(1, 128).astype(np.float32)
y = np.empty((1, 64), dtype=np.float32)

with SdxRuntime() as rt:
    with rt.load_model("/path/to/model.sdz") as model:
        with model.create_context() as ctx:
            ctx.run([x], [y])  # NumPy arrays are converted to TensorView automatically
```

Autodetection now uses host platform detection (`linux|windows|macos|android + arch`) and probes SDK-style layouts first:

- `<SDX_RUNTIME_HOME>/bindings/<platform>/<variant>/lib`
- `<SDX_RUNTIME_HOME>/sdx-runtime-sdk/bindings/<platform>/<variant>/lib`
- `<module_dir>/bindings/<platform>/<variant>/lib`

Then it falls back to system linker names (`nd4jcpu`, `nd4jcuda`, `nd4jamd`).

Loader environment overrides:

- `SDX_RUNTIME_HOME`: root folder of packaged SDK bindings.
- `SDX_RUNTIME_LIBRARY_DIR`: one or more library directories (use platform path separator).
- `SDX_RUNTIME_BACKEND`: backend priority seed (`cpu`, `cuda`, `amd`).
- `SDX_RUNTIME_PLATFORM_ID`: explicit platform id override (for non-standard layouts).

You can always pass an explicit runtime library path/name to `SdxRuntime(...)`.

## LLM / VLM / STT surface (`sdx_llm`)

Module: `sdx_llm.py` — ctypes wrapper for `libsdx_llm.so` (the AOT GraalVM
native-image LLM surface, defined in `nd4j/sdx-aot/include/sdx_llm_c.h`).
No JVM required.

### API table

| Class / function | Purpose |
|---|---|
| `SdxLlmRuntime` | Context manager wrapping `sdxLlmCreateRuntime` / `sdxLlmDestroyRuntime` |
| `SdxLlmRuntime.load_model(model_path, tokenizer_path, options_json)` | Returns `SdxLlmModel`; loads GGUF/sdz and builds pipeline |
| `SdxLlmRuntime.vlm_extract(model, tok, image, opts)` | Stateless SmolDocling image/PDF → doctags/text |
| `SdxLlmRuntime.audio_transcribe(model, audio, opts)` | Stateless Whisper STT |
| `SdxLlmModel` | Context manager for a loaded model; reuses warm plan/KV across calls |
| `SdxLlmModel.generate(prompt, options_json)` | Blocking text generation |
| `SdxLlmModel.last_result()` | Returns `GenerateStats` dataclass after `generate` |
| `SdxLlmModel.info()` | Dict: vocab size, chat-template flag, input/output names |
| `SdxLlmModel.tokenize(text, add_special_tokens)` | Text → `List[int]` token IDs |
| `SdxLlmModel.detokenize(ids, skip_special_tokens)` | `List[int]` → text |
| `GenerateStats` | Frozen dataclass: `prompt_tokens`, `generated_tokens`, `total_tokens`, `generation_time_ms`, `first_token_latency_ms`, `tokens_per_sec`, `finish_reason` |
| `SdxLlmError` | Raised on any non-OK ABI status |
| `load_library(path=None)` | Explicitly load `libsdx_llm.so`; called automatically by `SdxLlmRuntime.__init__` |
| `vlm_extract(runtime, ...)` | Module-level stateless VLM helper |
| `audio_transcribe(runtime, ...)` | Module-level stateless STT helper |

### Environment variables

| Variable | Purpose |
|---|---|
| `SDX_LLM_AOT_HOME` | Root of the unpacked AOT SDK (`$HOME/lib/libsdx_llm.so`). When not set, falls back to system `dlopen("libsdx_llm.so")`. |
| `SDX_NATIVE_LIB_DIR` | Override for the directory where `libsdx_llm.so` resolves its bundled native libs (ND4J, BLAS, tokenizers). Set automatically to `$SDX_LLM_AOT_HOME/lib` if unset. |

### Thread affinity

A `SdxLlmRuntime` is **bound to the OS thread that created it**.  Create, use,
and destroy each runtime from one thread.  For concurrent generation create one
`SdxLlmRuntime` per thread — the per-thread GraalVM isolate handles isolation.

### Quick start

```python
import os
os.environ["SDX_LLM_AOT_HOME"] = "/tmp/sdx-cpu-v8"

from sdx_llm import SdxLlmRuntime

with SdxLlmRuntime() as rt:
    print("ABI version:", rt.abi_version())
    with rt.load_model(
        "/path/to/model.gguf",
        "/path/to/tokenizer.json",
        '{"maxNewTokens":64,"sampling":{"preset":"greedy"}}',
    ) as model:
        ids = model.tokenize("Hello, world!")
        print("tokens:", ids)
        print("recovered:", model.detokenize(ids))
        text = model.generate("The capital of France is")
        print("generated:", text)
        stats = model.last_result()
        print(f"  {stats.tokens_per_sec:.1f} tok/s")
```

---

## SDK Runner (REST + gRPC)

Files:

- `sdx_sdk_runner.py` - dual-protocol server
- `sdx_serving.proto` - gRPC contract for tensor binary payloads
- `sdx_serving_pb2.py`, `sdx_serving_pb2_grpc.py` - generated Python stubs
- `sdx_tensor_transport.py` - shared ndarray transport and dtype validation

Start both servers:

```bash
python sdx_sdk_runner.py --library libnd4jcpu.so --rest-port 8080 --grpc-port 50051
```

Packaged SDK loading example (no explicit library file):

```bash
python sdx_sdk_runner.py \
  --runtime-home /opt/sdx-runtime-sdk \
  --runtime-platform-id linux-x86_64 \
  --runtime-backend cuda
```

REST APIs:

- `GET /healthz`
- `POST /v1/models:load`
- `POST /v1/models/{model_id}:unload`
- `POST /v1/models/{model_id}:run` (JSON tensors as base64)
- `POST /v1/models/{model_id}:run-npz` (binary NPZ request/response)

JSON run request shape:

```json
{
  "inputs": [
    {"name": "input_0", "dtype": 5, "shape": [1, 4], "data_base64": "..."}
  ],
  "outputs": [
    {"name": "output_0", "dtype": 5, "shape": [1, 2]}
  ],
  "run_options": {"backend": 0, "strict_signature": true, "gpu_target": 0}
}
```

NPZ run request:

- Body: `application/x-sdx-npz` (or `application/octet-stream`) with `input_0`, `input_1`, ... arrays.
- Optional header `X-SDX-Input-Order`: JSON array to explicitly define input ordering, e.g. `["token_ids","mask"]`.
- Header `X-SDX-Output-Specs`: JSON array of output specs, e.g.
  `[{"name":"output_0","dtype":5,"shape":[1,2]}]`
- Optional header `X-SDX-Run-Options`: JSON object matching run options.
- Response: NPZ output tensors, with execution report in `X-SDX-Execution-Report`.

Generate protobuf stubs (if proto changes):

```bash
python generate_proto.py --out .
```

Run SDK runner tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
