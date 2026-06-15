# SDX Java Binding (JNA)

This module wraps `dsp_runtime_c.h` using JNA.

Main entrypoint:

- `org.nd4j.dsp.runtime.SdxRuntime`

Example:

```java
try (SdxRuntime runtime = SdxRuntime.create()) {
    SdxRuntime.ModelOptions modelOptions = new SdxRuntime.ModelOptions();
    modelOptions.backend = SdxRuntime.SDX_BACKEND_AUTO;

    try (SdxRuntime.SdxModel model = runtime.loadModel("/path/to/model.sdz", modelOptions);
         SdxRuntime.SdxContext context = model.createContext(null)) {

        SdxRuntime.RunOptions runOptions = new SdxRuntime.RunOptions();
        runOptions.backend = SdxRuntime.SDX_BACKEND_AUTO;

        SdxRuntime.TensorView[] inputs = new SdxRuntime.TensorView[] { /* fill */ };
        SdxRuntime.TensorView[] outputs = new SdxRuntime.TensorView[] { /* fill */ };

        context.run(inputs, outputs, runOptions);
        SdxRuntime.ExecutionReport report = context.executionReport();
    }
}
```

ND4J `INDArray` integration (runtime reflection, no hard compile dependency in this wrapper):

```java
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

try (SdxRuntime runtime = SdxRuntime.create();
     SdxRuntime.SdxModel model = runtime.loadModel("/path/to/model.sdz", new SdxRuntime.ModelOptions());
     SdxRuntime.SdxContext context = model.createContext(null)) {

    INDArray input = Nd4j.rand(new long[]{1, 128});
    INDArray output = Nd4j.create(input.dataType(), 1, 64);
    context.runNd4j(new INDArray[]{input}, new INDArray[]{output});
}
```

Library loading:

- `SdxRuntime.create()` tries `nd4jcpu`, then `nd4jcuda`, then `nd4jamd`.
- You can pass an explicit path or runtime library name to `SdxRuntime.create(...)`.
