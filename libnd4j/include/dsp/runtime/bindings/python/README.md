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
