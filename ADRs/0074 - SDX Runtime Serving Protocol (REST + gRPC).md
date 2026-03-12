# ADR: SDX Runtime Serving Protocol (REST + gRPC)

## Status

Accepted (March 4, 2026)

Proposed by: Adam Gibson

Discussed with: Runtime/SDK maintainers

## Context

The SDX C runtime now supports direct `.sdz`/`.sdnb` model loading and multi-backend execution (CPU/CUDA/AMD/MLX/NNAPI/ARM). We need a serving layer that:

1. handles large binary tensor payloads efficiently,
2. preserves exact dtype/shape semantics from `dsp_runtime_c.h`,
3. is usable from both low-level and high-level client SDKs,
4. works consistently across Linux, Windows, macOS, iOS, Android, CUDA, and AMD target deployments.

The runtime ABI requires caller-provided output buffers (`sdxRun(...)`), so service requests must carry output specs (`dtype` + `shape`) unless another allocation API is introduced.

## Decision

We standardize on a dual-protocol serving contract:

### 1) gRPC as the primary typed binary protocol

- Contract: `libnd4j/include/dsp/runtime/bindings/python/sdx_serving.proto`
- Tensor message carries:
  - `bytes data`
  - `repeated int64 shape`
  - SDX dtype code (`int32 dtype`)
- `RunRequest` requires explicit output specs (`TensorSpec`) so server can allocate outputs before `sdxRun(...)`.
- Default gRPC max message size must be raised beyond the common 4 MiB default for ndarray workloads.

### 2) REST with binary-first NPZ payloads and JSON control plane

- Binary inference endpoint: `POST /v1/models/{model_id}:run-npz`
  - request body: NPZ with input ndarrays,
  - optional input order: `X-SDX-Input-Order` JSON header,
  - output specs: `X-SDX-Output-Specs` JSON header,
  - response body: NPZ output ndarrays,
  - execution metadata: `X-SDX-Execution-Report` JSON header.
- Compatibility endpoint: `POST /v1/models/{model_id}:run` with JSON/base64 tensors for smaller or debugging workloads.

### 3) Single execution core for both protocols

- Both transports use the same tensor codec (`sdx_tensor_transport.py`) and runtime registry (`sdx_sdk_runner.py`) to avoid behavioral drift.
- Runtime lifecycle is centralized:
  - load model,
  - create context,
  - execute,
  - unload model.

## Consequences

### Advantages

- One transport model for all backends and platforms.
- gRPC stays strongly typed while carrying raw binary tensors.
- REST supports efficient binary transfers without forcing base64 overhead.
- Shared codec minimizes protocol-specific bugs/regressions.

### Tradeoffs

- Output specs are required per run (due current C ABI contract).
- NPZ requires NumPy-compatible clients for the binary REST path.
- gRPC clients must manage message-size settings for large tensors.

## Implementation

- `libnd4j/include/dsp/runtime/bindings/python/sdx_sdk_runner.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_tensor_transport.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_serving.proto`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_serving_pb2.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_serving_pb2_grpc.py`

## References

- gRPC Python basics: https://grpc.io/docs/languages/python/basics/
- gRPC Java API default max inbound message size (4 MiB): https://grpc.github.io/grpc-java/javadoc/io/grpc/ManagedChannelBuilder.html
- Protocol Buffers scalar types (`bytes`): https://protobuf.dev/programming-guides/proto3/
- NumPy NPY/NPZ format docs: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
- HTTP semantics and content types: https://www.rfc-editor.org/rfc/rfc9110.html
