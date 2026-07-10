# SDX Swift Binding

Swift wrapper for the SDX C runtime ABI (`dsp_runtime_c.h`).

## Package structure

```
Sources/
  CSdxRuntime/          — system library target; imports dsp_runtime_c.h via shim.h
    shim.h              — resolves the header across SDK and source-tree layouts
    module.modulemap
  SdxRuntime/           — Swift wrapper target (extend here, not in generated files)
    SdxRuntime.swift    — public API
```

## Public API overview

### Value types

| Type | Description |
|------|-------------|
| `SdxTensor` | f32 tensor with copy-value semantics (`shape: [Int]`, `scalars: [Float]`) |
| `SdxExecutionReport` | Telemetry snapshot with Swifty names and typed enums |

### Typed enums

| Enum | Values |
|------|--------|
| `SdxBackend` | `.auto` `.slotBySlot` `.cudaGraphs` `.nvrtc` `.ptx` `.triton` `.mlx` `.armHybrid` `.nnapi` |
| `SdxDevice` | `.host` `.cuda` `.amd` |
| `SdxGpuTarget` | `.auto` `.cuda` `.amd` |
| `SdxPlanPhase` | `.slotBySlot` `.shapesFrozen` `.replaying` `.replayBlocked` |
| `SdxError` | `.nativeStatus(code:message:)` |

### Reference types

| Type | Description |
|------|-------------|
| `SdxRuntime` | Root runtime object; `deinit` releases the handle |
| `SdxModel` | Loaded model bundle; `deinit` unloads |
| `SdxContext` | Per-stream inference context; `deinit` destroys |
| `SdxTensorViewLease` | Zero-copy lease into a caller-owned buffer (power users) |

### Named-tensor inference API (idiomatic)

```swift
// Discovery
let names = ctx.inputNames()   // ["w1", "b1", "w2", "b2", "x"]

// Run
let outputs = try ctx.run(
    inputs:       ["w1": w1, "b1": b1, "w2": w2, "b2": b2, "x": x],
    outputShapes: ["probs": [2, 3]]
)
let probs: [Float] = outputs["probs"]!.scalars
```

### Low-level lease API (zero-copy, power users)

```swift
// Pre-existing buffer → lease → raw view array → run
let lease = SdxTensorViewLease(data: myBuf, shape: [2, 4], dtype: 5, bytes: 32)
try ctx.run(inputs: [lease.view], outputs: [outLease.view])
```

## Lifecycle

```
SdxRuntime.init()
  └─ SdxModel = runtime.loadModel(path:)
       └─ SdxContext = model.createContext(requestedOutputs:)
            ├─ ctx.markInputPlaceholder(_:)   // for value+shape-varying inputs
            ├─ ctx.run(inputs:outputShapes:)  // warmup (phase 0 → 1)
            ├─ ctx.freezeShapes()             // → phase 1/2
            └─ ctx.run(inputs:outputShapes:)  // replay fast path (phase 2)
```

## Notes

- `SdxRuntime`, `SdxModel`, and `SdxContext` release their native handles in
  `deinit`.  Call `close()` explicitly for deterministic teardown order.
- `SdxContext` is **not thread-safe**.  Create one context per concurrent stream.
- `SdxTensor` uses plain `[Float]` storage (copy-value semantics).  For
  zero-copy access to Metal buffers, Core ML arrays, or other existing
  allocations, use `SdxTensorViewLease` directly and call the low-level
  `run(inputs:outputs:)` overload.
- Ensure the SDX runtime shared library (`libnd4jcpu.so`, `libnd4jcuda.so`,
  or the `.xcframework` on Apple platforms) is on the library search path at
  link time and at runtime.
