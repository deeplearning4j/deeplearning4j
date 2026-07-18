# SDX Runtime — Java Binding

JNA-based Java wrappers for the SDX C ABIs. Two classes live in the same
`org.nd4j.dsp.runtime` package:

| Class | ABI | Description |
|-------|-----|-------------|
| `SdxRuntime` | `dsp_runtime_c.h` | General DSP inference runtime (tensor-level) |
| `SdxLlm`     | `sdx_llm_c.h`     | LLM/VLM/STT AOT runtime (`libsdx_llm.so`) |

## Source layout

Both canonical sources live in the `nd4j-sdx` Maven module:

```
nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java/org/nd4j/dsp/runtime/
  SdxRuntime.java   — DSP general inference runtime
  SdxLlm.java       — LLM / VLM / STT AOT runtime
```

They are intentionally **not** duplicated in this directory in the source tree.
When the SDK is staged (`cmake --build <build-dir> --target sdx_runtime_bindings`
or `libnd4j/tools/sdx-generate-bindings.sh`), both sources are copied into this
package under `src/main/java/`, making it self-contained and buildable with the
bundled `pom.xml`:

```bash
cd wrappers/java
mvn package
```

In the deeplearning4j repository, build the module instead:

```bash
mvn -Psdx install -pl :nd4j-sdx-preset,:nd4j-sdx -DskipTests
```

## SdxRuntime — DSP general inference runtime

Covers the full `dsp_runtime_c.h` surface: runtime/model/context lifecycle,
`sdxRun`, execution reports, input marking (`markInputVariable` /
`markInputPlaceholder`), shape freezing, plan phase and execution count queries,
plus an optional reflection-based ND4J `INDArray` interop (`runNd4j`).

### Library resolution

`SdxRuntime.create()` tries, in order: the standalone runtimes `sdx_cpu` /
`sdx_cuda` (JVM-free, built with `-DSD_BUILD_SDX_STANDALONE=ON`), then the
monolithic backend libraries `nd4jcpu` / `nd4jcuda` / `nd4jamd` — all export
the same `sdx*` C ABI. Resolution honors `jna.library.path`; when running from
an unpacked SDK package the `lib/` directory next to `binding.json` is detected
automatically, or pass the SDK root explicitly via `SdxRuntime.create(Path)`.

### Minimal usage

```java
try (SdxRuntime runtime = SdxRuntime.create()) {
    try (SdxRuntime.SdxModel model = runtime.loadModel("model.sdz", null);
         SdxRuntime.SdxContext ctx = model.createContext(null)) {
        SdxRuntime.TensorView in = SdxRuntime.TensorView.hostTensor(dataPtr, new long[]{1, 4}, /*dtype=*/5, /*bytes=*/16);
        SdxRuntime.TensorView out = SdxRuntime.TensorView.hostTensor(outPtr, new long[]{1, 2}, 5, 8);
        ctx.run(new SdxRuntime.TensorView[]{in}, new SdxRuntime.TensorView[]{out}, null);
        SdxRuntime.ExecutionReport report = ctx.executionReport();
    }
}
```

## SdxLlm — LLM / VLM / STT AOT runtime

Covers the `sdx_llm_c.h` surface: runtime/model lifecycle, text generation,
tokenize/detokenize, model info JSON, last-result stats JSON, stateless VLM
document extraction (`vlmExtract`), and Whisper speech-to-text
(`audioTranscribe`).

### Library resolution

`SdxLlm.Runtime.create()` tries, in order:
1. JVM system property `sdx.llm.library` or env-var `SDX_LLM_LIBRARY`.
2. `$SDX_LLM_AOT_HOME/lib/libsdx_llm.so` (also extends `jna.library.path`).
3. Bare name `sdx_llm` via JNA default search.

### Side-loaded natives — CRITICAL

`libsdx_llm.so` resolves its companion natives relative to the host executable
using `SDX_NATIVE_LIB_DIR`. A JVM **cannot** set process environment after
start. Export the variable before launching the JVM:

```bash
export SDX_LLM_AOT_HOME=/path/to/sdx-sdk
export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
java -jar my-app.jar
```

### Minimal usage

```java
try (SdxLlm.Runtime rt = SdxLlm.Runtime.create()) {
    try (SdxLlm.Model model = rt.loadModel(modelPath, tokenizerPath, null)) {
        String text = model.generate("The capital of France is",
                "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
        System.out.println(text);       // " Paris."
        System.out.println(model.lastResultJson());
        int[] ids  = model.tokenize("Hello world", false);
        String back = model.detokenize(ids, true);
    }
    // VLM extraction (stateless)
    String extracted = rt.vlmExtract(vlmModelPath, null, imagePath, null);
}
```

## Kotlin binding

The Kotlin binding in `../kotlin` provides idiomatic facades over **both**
`SdxRuntime` (`KotlinSdxRuntime`) and `SdxLlm` (`KotlinSdxLlmRuntime`).
