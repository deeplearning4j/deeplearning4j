# SDX Kotlin Binding

Kotlin facade for the Java SDX wrapper.

Primary class:

- `org.nd4j.dsp.runtime.KotlinSdxRuntime`

Example:

```kotlin
KotlinSdxRuntime.create().use { runtime ->
    runtime.loadModel("/path/to/model.sdz").use { model ->
        model.createContext().use { ctx ->
            val inputs = arrayOf<SdxRuntime.TensorView>()
            val outputs = arrayOf<SdxRuntime.TensorView>()
            ctx.run(inputs, outputs)
            val report = ctx.executionReport()
            println("exec ns = ${report.execution_time_ns}")
        }
    }
}
```

Library loading:

- `KotlinSdxRuntime.create()` defers to Java auto-discovery (`nd4jcpu`, `nd4jcuda`, `nd4jamd`).
- You can pass an explicit path/name with `KotlinSdxRuntime.create("...")`.
