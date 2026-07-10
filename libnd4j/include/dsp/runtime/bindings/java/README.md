# SDX Runtime — Java Binding

JNA-based Java wrapper for the SDX runtime C ABI (`dsp_runtime_c.h`). Covers the
full ABI surface: runtime/model/context lifecycle, `sdxRun`, execution reports,
input marking (`markInputVariable` / `markInputPlaceholder`), shape freezing,
plan phase and execution count queries, plus an optional reflection-based ND4J
`INDArray` interop (`runNd4j`).

## Source layout

The canonical wrapper source lives in the `nd4j-sdx` Maven module of the
deeplearning4j repository
(`nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java/org/nd4j/dsp/runtime/SdxRuntime.java`).
It is intentionally **not** duplicated in this directory in the source tree.

When the SDK is staged (`cmake --build <build-dir> --target sdx_runtime_bindings`
or `libnd4j/tools/sdx-generate-bindings.sh`), the source is copied into this
package under `src/main/java/`, making it self-contained and buildable with the
bundled `pom.xml`:

```bash
cd wrappers/java
mvn package
```

In the deeplearning4j repository, build the module instead:

```bash
mvn -Psdx install -pl :nd4j-sdx -DskipTests
```

## Library resolution

`SdxRuntime.create()` tries, in order: the standalone runtimes `sdx_cpu` /
`sdx_cuda` (JVM-free, built with `-DSD_BUILD_SDX_STANDALONE=ON`), then the
monolithic backend libraries `nd4jcpu` / `nd4jcuda` / `nd4jamd` — all export
the same `sdx*` C ABI. Resolution honors `jna.library.path`; when running from
an unpacked SDK package the `lib/` directory next to `binding.json` is detected
automatically, or pass the SDK root explicitly via `SdxRuntime.create(Path)`.

## Minimal usage

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

The Kotlin binding in `../kotlin` is a thin facade over this class.
