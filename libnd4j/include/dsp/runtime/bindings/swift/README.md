# SDX Swift Binding

Swift wrappers for both SDX C ABIs.

## Package products

| Product | C ABI | Description |
|---------|-------|-------------|
| `SdxRuntime` | `dsp_runtime_c.h` | General DSP inference runtime |
| `SdxLlm`     | `sdx_llm_c.h`     | LLM / VLM / STT AOT runtime (`libsdx_llm`) |

## Package structure

```
Sources/
  CSdxRuntime/          — system library; imports dsp_runtime_c.h via shim.h
    shim.h              — resolves header across SDK and source-tree layouts
    module.modulemap    — links "nd4jcpu"
  SdxRuntime/           — Swift wrapper for dsp_runtime_c.h
    SdxRuntime.swift
  CSdxLlm/              — system library; imports sdx_llm_c.h via shim.h
    shim.h              — resolves header across SDK and source-tree layouts
    module.modulemap    — links "sdx_llm"
  SdxLlm/               — Swift wrapper for sdx_llm_c.h
    SdxLlm.swift
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

## Notes (SdxRuntime)

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

---

## SdxLlm — LLM / VLM / STT AOT runtime

`SdxLlm.swift` wraps `sdx_llm_c.h` — the AOT-compiled (GraalVM native-image)
LLM/VLM/STT library (`libsdx_llm`). **No JVM is embedded in the library.**

### Quick start

```swift
// Optionally configure SDX_NATIVE_LIB_DIR before the first call.
// SdxLlmRuntime.init() does this automatically from SDX_LLM_AOT_HOME.
let rt = try SdxLlmRuntime()

let model = try rt.loadModel(
    modelPath:     "/path/to/model.gguf",
    tokenizerPath: "/path/to/tokenizer.json",
    optionsJson:   #"{"maxNewTokens":8,"sampling":{"preset":"greedy"}}"#
)

let text = try model.generate("The capital of France is")
print(text)                          // " Paris."

if let stats = try model.lastResultStats() {
    print(stats.tokensPerSecond ?? 0, "tok/s")
}

// Tokenize / detokenize
let ids  = try model.tokenize("Hello world")
let back = try model.detokenize(ids)

// VLM extraction (stateless)
let extracted = try rt.vlmExtract(modelPath: vlmPath, inputPath: imagePath)

// STT (stateless)
let transcript = try rt.audioTranscribe(modelPath: whisperPath, audioPath: wavPath)
```

### Public API

| Type | Description |
|------|-------------|
| `SdxLlmRuntime` | Root object; `deinit` tears down the GraalVM isolate |
| `SdxLlmModel` | Loaded model; `deinit` unloads |
| `SdxLlmStats` | Parsed generation stats |
| `SdxLlmStatusCode` | Typed status enum |
| `SdxLlmError` | Error type |

### Build command

```bash
swift build \
  -Xcc -I$SDX_LLM_AOT_HOME/include \
  -Xlinker -L$SDX_LLM_AOT_HOME/lib
```

The `CSdxLlm/module.modulemap` declares `link "sdx_llm"` — no `-lsdx_llm` flag needed.

### Side-loaded natives

`libsdx_llm` resolves companion natives using `SDX_NATIVE_LIB_DIR`. Set it
**before** the process starts or call `setenv` before `SdxLlmRuntime.init()`:

```swift
setenv("SDX_NATIVE_LIB_DIR", "\(aotHome)/lib", 0)
let rt = try SdxLlmRuntime()
```

`SdxLlmRuntime.init()` performs this automatically when `SDX_LLM_AOT_HOME` is
set and `SDX_NATIVE_LIB_DIR` is not.

### Threading

Each `SdxLlmRuntime` handle is bound to the OS thread that created it. Create,
use, and destroy each runtime from the same thread. Use one runtime per thread
for concurrent generation.
