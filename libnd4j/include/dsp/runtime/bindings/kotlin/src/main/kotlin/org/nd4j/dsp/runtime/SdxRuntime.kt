package org.nd4j.dsp.runtime

/**
 * Kotlin facade over the Java SDX wrapper.
 */
class KotlinSdxRuntime private constructor(
    private val delegate: SdxRuntime
) : AutoCloseable {

    companion object {
        @JvmStatic
        fun create(libraryNameOrPath: String? = null): KotlinSdxRuntime {
            val runtime = if (libraryNameOrPath.isNullOrBlank()) {
                SdxRuntime.create()
            } else {
                SdxRuntime.create(libraryNameOrPath)
            }
            return KotlinSdxRuntime(runtime)
        }
    }

    fun abiVersion(): Int = delegate.abiVersion()

    fun loadModel(
        bundlePath: String,
        backend: Int = SdxRuntime.SDX_BACKEND_AUTO,
        strictBackend: Boolean = false,
        allowRuntimeJit: Boolean = false,
        gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO
    ): KotlinSdxModel {
        val opts = SdxRuntime.ModelOptions().apply {
            this.backend = backend
            this.strict_backend = if (strictBackend) 1 else 0
            this.allow_runtime_jit = if (allowRuntimeJit) 1 else 0
            this.gpu_target = gpuTarget
            write()
        }
        return KotlinSdxModel(delegate.loadModel(bundlePath, opts))
    }

    fun lastError(): String = delegate.lastError()

    override fun close() {
        delegate.close()
    }

    class KotlinSdxModel internal constructor(
        private val delegate: SdxRuntime.SdxModel
    ) : AutoCloseable {

        fun createContext(requestedOutputs: Array<String>? = null): KotlinSdxContext {
            return KotlinSdxContext(delegate.createContext(requestedOutputs))
        }

        override fun close() {
            delegate.close()
        }
    }

    class KotlinSdxContext internal constructor(
        private val delegate: SdxRuntime.SdxContext
    ) : AutoCloseable {

        fun run(
            inputs: Array<SdxRuntime.TensorView>,
            outputs: Array<SdxRuntime.TensorView>,
            backend: Int = SdxRuntime.SDX_BACKEND_AUTO,
            strictSignature: Boolean = true,
            gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO
        ) {
            val opts = SdxRuntime.RunOptions().apply {
                this.backend = backend
                this.strict_signature = if (strictSignature) 1 else 0
                this.gpu_target = gpuTarget
                write()
            }
            delegate.run(inputs, outputs, opts)
        }

        fun executionReport(): SdxRuntime.ExecutionReport = delegate.executionReport()

        override fun close() {
            delegate.close()
        }
    }
}
