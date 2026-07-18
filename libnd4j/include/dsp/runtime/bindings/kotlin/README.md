# SDX Kotlin Binding

Idiomatic Kotlin facades over the JNA-based Java wrappers for both SDX C ABIs:

| Kotlin class | Java delegate | C ABI |
|---|---|---|
| `KotlinSdxRuntime` | `SdxRuntime` | `dsp_runtime_c.h` — general DSP inference |
| `KotlinSdxLlmRuntime` | `SdxLlm.Runtime` | `sdx_llm_c.h` — LLM/VLM/STT AOT runtime |

## Primary types

| Type | Role |
|------|------|
| `KotlinSdxRuntime` | Runtime factory; `AutoCloseable` |
| `KotlinSdxRuntime.KotlinSdxModel` | Loaded model bundle; `AutoCloseable` |
| `KotlinSdxRuntime.KotlinSdxContext` | Execution context + DSP lifecycle; `AutoCloseable` |
| `FloatTensor` | Host float tensor backed by JNA `Memory` |
| `ExecutionReport` | `data class` mirror of the C execution-report struct |
| `SdxBackend` | `enum class` — typed back-end identifiers |
| `PlanPhase` | `enum class` — DSP lifecycle phases |

## Quick start

```kotlin
KotlinSdxRuntime.create().use { runtime ->
    runtime.loadModel("model.sdz").use { model ->
        model.createContext(outputs = listOf("probs")).use { ctx ->

            // Discover input contract
            val names: List<String> = ctx.inputNames()   // ["w1","b1","w2","b2","x"]

            // Mark runtime-varying inputs
            ctx.markPlaceholders("x")

            // Build named inputs — order resolved automatically
            val xTensor = FloatTensor(xData, longArrayOf(2, 4))
            val output  = FloatTensor.zeros(longArrayOf(2, 3))

            // Warmup (SLOT_BY_SLOT)
            repeat(3) {
                ctx.runNamed(
                    inputs  = mapOf("x" to xTensor),
                    weights = weightsMap,
                    outputs = listOf(output),
                )
            }

            // Freeze → CUDA graph capture → fast-path replay
            ctx.freezeShapes()
            println(ctx.phaseLabel)   // "REPLAYING"

            // Typed execution report
            println(ctx.executionReport().summary())

            // Read results
            val probs: FloatArray = output.readBack()
        }
    }
}
```

## Key APIs

### `KotlinSdxContext`

| API | Description |
|-----|-------------|
| `inputNames(): List<String>` | All external input names in plan-binding order |
| `numInputs: Int` | Number of external inputs |
| `numOutputs: Int` | Number of outputs |
| `markPlaceholders(vararg names)` | Mark named inputs as placeholders; returns marked indices |
| `markInputVariable(index)` | Mark a single input as a variable |
| `markInputPlaceholder(index)` | Mark a single input as a placeholder |
| `freezeShapes()` | Signal shape stability; triggers CUDA graph capture |
| `phase: PlanPhase?` | Current DSP lifecycle phase (typed enum) |
| `phaseLabel: String` | Human-readable phase name; never throws |
| `isReplaying: Boolean` | `true` when fast-path graph replay is active |
| `isWarmup: Boolean` | `true` when still in slot-by-slot warmup |
| `executionCount: Int` | Total runs completed on this context |
| `executionReport(): ExecutionReport` | Typed data-class snapshot of the last run |
| `run(inputs, outputs)` | Positional run with `FloatTensor` lists |
| `runNamed(inputs, weights, outputs)` | Named-input run; order resolved via `inputNames()` |

### `FloatTensor`

```kotlin
FloatTensor(data, shape)          // wrap existing FloatArray
FloatTensor.zeros(shape)          // zero-filled output tensor
FloatTensor.of(data, *shape)      // vararg shape convenience
tensor.readBack()                  // copy native memory → FloatArray
tensor.view                        // SdxRuntime.TensorView for raw run() calls
```

### `ExecutionReport` (data class)

```kotlin
data class ExecutionReport(
    val statusCode: Int,
    val requestedBackend: SdxBackend?,
    val appliedBackend: SdxBackend?,
    val usedFallback: Boolean?,
    val phase: PlanPhase?,
    val executionCount: Int,
    val executionTimeNs: Long,
    val requestedGpuTarget: Int,
    val appliedGpuTarget: Int,
) {
    val executionTimeMs: Double     // convenience: ns → ms
    fun summary(): String           // formatted multi-line string
}
```

### `SdxBackend` / `PlanPhase` enums

```kotlin
SdxBackend.AUTO            // 0  — runtime chooses
SdxBackend.SLOT_BY_SLOT    // 1
SdxBackend.CUDA_GRAPHS     // 2
SdxBackend.TRITON          // 5
// … plus NVRTC, PTX, MLX, ARM_HYBRID, NNAPI

PlanPhase.SLOT_BY_SLOT     // 0 — warmup; every run re-traces
PlanPhase.SHAPES_FROZEN    // 1 — capture pending
PlanPhase.REPLAYING        // 2 — graph is captured and replaying
PlanPhase.REPLAY_BLOCKED   // 3 — replay requested but unavailable
```

## Library loading — DSP runtime

```kotlin
KotlinSdxRuntime.create()                    // auto-discover sdx_cpu/sdx_cuda/nd4jcpu/nd4jcuda
KotlinSdxRuntime.create("sdx_cpu")          // explicit library name
KotlinSdxRuntime.create(sdkRootPath)         // Path to SDK root; sets jna.library.path to lib/
```

The Java wrapper also auto-detects an SDK layout by walking up from the JAR
location and checking for `binding.json`, then setting `jna.library.path` to
the adjacent `lib/` directory.

---

## KotlinSdxLlmRuntime — LLM / VLM / STT AOT runtime

`KotlinSdxLlmRuntime` is an idiomatic Kotlin facade over `SdxLlm.Runtime` (the
JNA wrapper for `sdx_llm_c.h`). It binds `libsdx_llm.so` — the AOT-compiled
(GraalVM native-image) LLM/VLM/STT library — via JNA. **No JVM is embedded in
the library.**

### Quick start

```kotlin
KotlinSdxLlmRuntime.create().use { runtime ->
    runtime.loadModel(modelPath, tokenizerPath).use { model ->
        val text = model.generate(
            prompt = "The capital of France is",
            optionsJson = """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}""",
        )
        println(text)                       // " Paris."
        println(model.lastResultStats().summary())

        val ids  = model.tokenize("Hello world")
        val back = model.detokenize(ids)
    }
    // Stateless VLM extraction and STT
    val extracted = runtime.vlmExtract(vlmModelPath, null, imagePath, null)
    val transcript = runtime.audioTranscribe(whisperModelPath, audioPath)
}
```

### Primary types

| Type | Role |
|------|------|
| `KotlinSdxLlmRuntime` | Runtime factory + VLM/STT helpers; `AutoCloseable` |
| `KotlinSdxLlmRuntime.KotlinSdxLlmModel` | Loaded LLM model; `AutoCloseable` |
| `LlmResultStats` | `data class` — generation stats (tok/s, token counts, finish reason) |
| `SdxLlmStatus` | `enum class` — typed status codes |

### Key APIs

| API | Description |
|-----|-------------|
| `KotlinSdxLlmRuntime.create()` | Auto-detect `libsdx_llm.so` |
| `runtime.loadModel(path, tokenizer?, options?)` | Load GGUF/SDZ/SameDiff model |
| `model.generate(prompt, optionsJson?)` | Blocking text generation |
| `model.lastResultStats()` | Typed `LlmResultStats` data class |
| `model.lastResultJson()` | Raw stats JSON string |
| `model.infoJson()` | Model/tokenizer summary JSON |
| `model.tokenize(text, addSpecialTokens)` | Text → `IntArray` of token IDs |
| `model.detokenize(ids, skipSpecialTokens)` | Token IDs → text |
| `runtime.vlmExtract(modelPath, tokenizer?, inputPath, options?)` | SmolDocling VLM extraction |
| `runtime.audioTranscribe(modelPath, audioPath, options?)` | Whisper STT |

### Library loading — LLM runtime

```kotlin
KotlinSdxLlmRuntime.create()               // auto-detect libsdx_llm.so
KotlinSdxLlmRuntime.create("sdx_llm")      // explicit bare name
KotlinSdxLlmRuntime.create("/path/to/libsdx_llm.so")  // absolute path
```

Resolution order (first match wins):
1. JVM property `sdx.llm.library` or env `SDX_LLM_LIBRARY`.
2. `$SDX_LLM_AOT_HOME/lib/libsdx_llm.so`.
3. Bare name `sdx_llm` via JNA default search.

### Side-loaded natives — CRITICAL

`libsdx_llm.so` resolves its companion natives using `SDX_NATIVE_LIB_DIR`. A JVM
**cannot** set process environment after start. Export before launching:

```bash
export SDX_LLM_AOT_HOME=/path/to/sdx-sdk
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
gradle run --args="<model> <tokenizer>"
```
