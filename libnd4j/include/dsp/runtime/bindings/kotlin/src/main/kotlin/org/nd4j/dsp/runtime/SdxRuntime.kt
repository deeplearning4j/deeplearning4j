package org.nd4j.dsp.runtime

import com.sun.jna.Memory
import com.sun.jna.Pointer

// ─────────────────────────────────────────────────────────────────────────────
// Domain enumerations — idiomatic Kotlin alternatives to bare Int constants.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * DSP execution back-end requested or applied for a run.
 *
 * Values match the `SDX_BACKEND_*` C constants in `dsp_runtime_c.h`.
 */
enum class SdxBackend(val code: Int) {
    AUTO(SdxRuntime.SDX_BACKEND_AUTO),
    SLOT_BY_SLOT(SdxRuntime.SDX_BACKEND_SLOT_BY_SLOT),
    CUDA_GRAPHS(SdxRuntime.SDX_BACKEND_CUDA_GRAPHS),
    NVRTC(SdxRuntime.SDX_BACKEND_NVRTC),
    PTX(SdxRuntime.SDX_BACKEND_PTX),
    TRITON(SdxRuntime.SDX_BACKEND_TRITON),
    MLX(SdxRuntime.SDX_BACKEND_MLX),
    ARM_HYBRID(SdxRuntime.SDX_BACKEND_ARM_HYBRID),
    NNAPI(SdxRuntime.SDX_BACKEND_NNAPI),
    HIP_GRAPHS(SdxRuntime.SDX_BACKEND_HIP_GRAPHS),
    LEVEL_ZERO(SdxRuntime.SDX_BACKEND_LEVEL_ZERO),
    VULKAN(SdxRuntime.SDX_BACKEND_VULKAN),
    METAL(SdxRuntime.SDX_BACKEND_METAL),
    TPU(SdxRuntime.SDX_BACKEND_TPU),
    HEXAGON(SdxRuntime.SDX_BACKEND_HEXAGON);

    companion object {
        /** Returns the enum constant for [code], or null if unrecognised. */
        fun fromCode(code: Int): SdxBackend? = entries.firstOrNull { it.code == code }
    }
}

/**
 * DSP plan lifecycle phase.
 *
 * Phases progress monotonically: [SLOT_BY_SLOT] → [SHAPES_FROZEN] → [REPLAYING].
 * [REPLAY_BLOCKED] signals that replay was requested but is unavailable.
 */
enum class PlanPhase(val code: Int) {
    /** Warmup: every run re-traces shapes through the plan. */
    SLOT_BY_SLOT(0),

    /** Shapes have been frozen; next execution will attempt graph capture. */
    SHAPES_FROZEN(1),

    /** CUDA / native graph is captured and replaying at full speed. */
    REPLAYING(2),

    /** Replay was requested but capture is unavailable for this configuration. */
    REPLAY_BLOCKED(3);

    companion object {
        /** Returns the enum constant for [code], or null if unrecognised. */
        fun fromCode(code: Int): PlanPhase? = entries.firstOrNull { it.code == code }

        /** Human-readable label for display; never throws. */
        fun label(code: Int): String = fromCode(code)?.name ?: "UNKNOWN($code)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Value types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Kotlin data class mirror of [SdxRuntime.ExecutionReport].
 *
 * All fields have idiomatic Kotlin names (camelCase). The [phase] and
 * [requestedBackend] / [appliedBackend] fields expose typed enums in addition
 * to their raw [Int] codes so callers need not look up constants manually.
 *
 * @param statusCode      `SDX_STATUS_OK` (0) on success.
 * @param requestedBackend Back-end the caller requested.
 * @param appliedBackend  Back-end the runtime actually used.
 * @param usedFallback    `true` when [appliedBackend] != [requestedBackend].
 *                        `null` when the runtime does not know.
 * @param phase           DSP lifecycle phase at the end of the last run.
 * @param executionCount  Total runs completed on this context.
 * @param executionTimeNs Wall-clock time of the last run in nanoseconds.
 * @param requestedGpuTarget GPU target the caller requested.
 * @param appliedGpuTarget   GPU target the runtime actually used.
 */
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
    /** [executionTimeNs] expressed in milliseconds for convenient printing. */
    val executionTimeMs: Double get() = executionTimeNs / 1.0e6

    /**
     * Multi-line summary suitable for logging or printing.
     *
     * Example:
     * ```
     * ExecutionReport {
     *   status          = 0 (OK)
     *   requestedBackend= AUTO
     *   appliedBackend  = SLOT_BY_SLOT
     *   usedFallback    = false
     *   phase           = REPLAYING
     *   executionCount  = 6
     *   executionTime   = 0.123 ms
     * }
     * ```
     */
    fun summary(): String = buildString {
        appendLine("ExecutionReport {")
        appendLine("  status           = $statusCode${if (statusCode == 0) " (OK)" else ""}")
        appendLine("  requestedBackend = ${requestedBackend?.name ?: "UNKNOWN($requestedBackend)"}")
        appendLine("  appliedBackend   = ${appliedBackend?.name ?: "UNKNOWN($appliedBackend)"}")
        appendLine("  usedFallback     = ${usedFallback ?: "unknown"}")
        appendLine("  phase            = ${phase?.name ?: "UNKNOWN"}")
        appendLine("  executionCount   = $executionCount")
        append("  executionTime    = %.3f ms".format(executionTimeMs))
        append("\n}")
    }
}

/** Converts the JNA [SdxRuntime.ExecutionReport] struct to the idiomatic [ExecutionReport]. */
fun SdxRuntime.ExecutionReport.toKotlin(): ExecutionReport = ExecutionReport(
    statusCode = status_code,
    requestedBackend = SdxBackend.fromCode(requested_backend),
    appliedBackend = SdxBackend.fromCode(applied_backend),
    usedFallback = when (used_fallback) {
        -1 -> null
        0  -> false
        else -> true
    },
    phase = PlanPhase.fromCode(plan_phase),
    executionCount = execution_count,
    executionTimeNs = execution_time_ns,
    requestedGpuTarget = requested_gpu_target,
    appliedGpuTarget = applied_gpu_target,
)

// ─────────────────────────────────────────────────────────────────────────────
// FloatTensor — lightweight JNA-backed host tensor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A host-resident float tensor backed by a pinned JNA [Memory] buffer.
 *
 * Wraps a [FloatArray] in a [SdxRuntime.TensorView] so it can be passed
 * directly to [KotlinSdxContext.run] without manual memory management at
 * the call site.
 *
 * Keep the [FloatTensor] alive for the duration of any [KotlinSdxContext.run]
 * call that references it — JNA reads the [memory] pointer during execution.
 *
 * @param data  Initial float values (copied into native memory immediately).
 * @param shape Tensor dimensions; product must equal [data].size.
 */
class FloatTensor(val data: FloatArray, val shape: LongArray) {

    /** Pinned native memory holding a copy of [data]. */
    val memory: Memory = Memory(data.size.toLong() * java.lang.Float.BYTES).also {
        it.write(0, data, 0, data.size)
    }

    /** JNA [SdxRuntime.TensorView] pointing into [memory]. Ready for [KotlinSdxContext.run]. */
    val view: SdxRuntime.TensorView = SdxRuntime.TensorView.tensor(
        memory, shape,
        /* dtype = */ SDX_DTYPE_FLOAT32,
        data.size.toLong() * java.lang.Float.BYTES,
        /* deviceType = */ SdxRuntime.SDX_DEVICE_HOST,
        /* deviceId = */ -1,
    )

    /** Reads the current contents of [memory] back into a new [FloatArray]. */
    fun readBack(): FloatArray {
        val out = FloatArray(data.size)
        memory.read(0, out, 0, out.size)
        return out
    }

    /** Number of elements. */
    val size: Int get() = data.size

    /** Total bytes occupied in native memory. */
    val byteSize: Long get() = memory.size()

    companion object {
        /** `sd::DataType::FLOAT32` — the only dtype used by SDX host tensors in this facade. */
        const val SDX_DTYPE_FLOAT32: Int = 5

        /** Creates a zero-filled output tensor of the given [shape]. */
        fun zeros(shape: LongArray): FloatTensor {
            val n = shape.fold(1L, Long::times).toInt()
            return FloatTensor(FloatArray(n), shape)
        }

        /** Creates a [FloatTensor] from an explicit [FloatArray] and a vararg [shape]. */
        fun of(data: FloatArray, vararg shape: Long): FloatTensor = FloatTensor(data, shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kotlin facade
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Idiomatic Kotlin facade over [SdxRuntime] (the JNA-based Java wrapper for
 * the SDX C ABI).
 *
 * All three lifecycle objects ([KotlinSdxRuntime], [KotlinSdxModel],
 * [KotlinSdxContext]) implement [AutoCloseable] and are designed for Kotlin's
 * `use {}` scoping idiom:
 *
 * ```kotlin
 * KotlinSdxRuntime.create().use { runtime ->
 *     runtime.loadModel("model.sdz").use { model ->
 *         model.createContext(outputs = listOf("probs")).use { ctx ->
 *             val inputOrder = ctx.inputNames()
 *             val report = ctx.runNamed(
 *                 inputs = mapOf("x" to FloatTensor.of(xData, 2, 4)),
 *                 weights = weightsFor(inputOrder),
 *             )
 *             println(report.summary())
 *         }
 *     }
 * }
 * ```
 *
 * The facade is **dependency-light**: it only requires JNA (already needed by
 * the Java wrapper). No ND4J on the classpath is required.
 *
 * @see SdxRuntime — the underlying JNA Java wrapper (read-only reference).
 */
class KotlinSdxRuntime private constructor(
    private val delegate: SdxRuntime,
) : AutoCloseable {

    // ── Factory ──────────────────────────────────────────────────────────────

    companion object {
        /**
         * Creates a runtime using automatic library discovery.
         *
         * The Java wrapper searches standalone and monolithic CPU, CUDA, Vulkan,
         * and Metal library names in the JNA library path. Set
         * `-Djna.library.path` or call
         * [create(libraryNameOrPath)] to override.
         */
        @JvmStatic
        fun create(): KotlinSdxRuntime = KotlinSdxRuntime(SdxRuntime.create())

        /**
         * Creates a runtime loading the library named or found at [libraryNameOrPath].
         *
         * @param libraryNameOrPath Base name (e.g. `"sdx_cpu"`) or absolute path to the
         *                          shared library. When blank or null, falls back to auto-discovery.
         */
        @JvmStatic
        fun create(libraryNameOrPath: String?): KotlinSdxRuntime {
            val runtime = if (libraryNameOrPath.isNullOrBlank()) {
                SdxRuntime.create()
            } else {
                SdxRuntime.create(libraryNameOrPath)
            }
            return KotlinSdxRuntime(runtime)
        }

        /**
         * Creates a runtime with an explicit SDK root directory.
         *
         * Sets `jna.library.path` to `sdkRoot/lib/` before loading.
         */
        @JvmStatic
        fun create(sdkRoot: java.nio.file.Path): KotlinSdxRuntime =
            KotlinSdxRuntime(SdxRuntime.create(sdkRoot))
    }

    // ── Runtime API ──────────────────────────────────────────────────────────

    /** ABI version reported by the native library. */
    fun abiVersion(): Int = delegate.abiVersion()

    /** Last error message from the native runtime, or an empty string. */
    fun lastError(): String = delegate.lastError()

    /**
     * Loads a model bundle (`.sdz`) and returns an [KotlinSdxModel].
     *
     * The returned model must be closed when no longer needed; prefer `use {}`.
     *
     * @param bundlePath       Absolute or relative path to the `.sdz` file.
     * @param backend          Preferred execution back-end (default [SdxBackend.AUTO]).
     * @param strictBackend    When `true`, fail if the requested back-end is unavailable.
     * @param allowRuntimeJit  When `true`, permit on-the-fly JIT compilation during runs.
     * @param gpuTarget        GPU device preference (default [SdxRuntime.SDX_GPU_TARGET_AUTO]).
     */
    fun loadModel(
        bundlePath: String,
        backend: SdxBackend = SdxBackend.AUTO,
        strictBackend: Boolean = false,
        allowRuntimeJit: Boolean = false,
        gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO,
    ): KotlinSdxModel {
        val opts = SdxRuntime.ModelOptions().apply {
            this.backend = backend.code
            this.strict_backend = if (strictBackend) 1 else 0
            this.allow_runtime_jit = if (allowRuntimeJit) 1 else 0
            this.gpu_target = gpuTarget
            write()
        }
        return KotlinSdxModel(delegate.loadModel(bundlePath, opts))
    }

    override fun close(): Unit = delegate.close()

    // ─────────────────────────────────────────────────────────────────────────
    // Model
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A loaded SDX model bundle.
     *
     * Obtain via [KotlinSdxRuntime.loadModel]; close via `use {}` or [close].
     */
    class KotlinSdxModel internal constructor(
        private val delegate: SdxRuntime.SdxModel,
    ) : AutoCloseable {

        /** Resolved offline tokenizer path declared by the bundle, if present. */
        val tokenizerPath: String? get() = delegate.tokenizerPath()

        /** Resolved text-generation metadata path declared by the bundle. */
        val textGenerationConfigPath: String?
            get() = delegate.textGenerationConfigPath()

        /**
         * Creates an execution context for this model.
         *
         * @param outputs Names of the plan variables to materialise as outputs.
         *                When empty the runtime returns all terminal outputs.
         */
        fun createContext(vararg outputs: String): KotlinSdxContext =
            KotlinSdxContext(delegate.createContext(outputs.takeIf { it.isNotEmpty() }))

        /**
         * Creates an execution context, selecting outputs from a [List].
         *
         * @see createContext
         */
        fun createContext(outputs: List<String> = emptyList()): KotlinSdxContext =
            KotlinSdxContext(delegate.createContext(outputs.takeIf { it.isNotEmpty() }?.toTypedArray()))

        /**
         * Creates the recommended mobile inference context with model parameters
         * bound internally and only placeholders exposed as public inputs.
         */
        fun createInferenceContext(vararg outputs: String): KotlinSdxContext =
            KotlinSdxContext(
                delegate.createInferenceContext(outputs.takeIf { it.isNotEmpty() })
            )

        fun createInferenceContext(
            outputs: List<String> = emptyList(),
        ): KotlinSdxContext = KotlinSdxContext(
            delegate.createInferenceContext(
                outputs.takeIf { it.isNotEmpty() }?.toTypedArray()
            )
        )

        override fun close(): Unit = delegate.close()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Context
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * An execution context bound to a loaded model.
     *
     * The context owns the DSP plan and its shape/phase lifecycle. Obtain via
     * [KotlinSdxModel.createContext]; close via `use {}` or [close].
     *
     * ## Typical lifecycle
     *
     * ```kotlin
     * ctx.markPlaceholders()         // mark "x" as placeholder
     * repeat(3) { ctx.run(...) }     // warmup — SLOT_BY_SLOT
     * ctx.freezeShapes()             // transition → SHAPES_FROZEN → REPLAYING
     * repeat(3) { ctx.run(...) }     // fast-path replay
     * println(ctx.executionReport().summary())
     * ```
     */
    class KotlinSdxContext internal constructor(
        private val delegate: SdxRuntime.SdxContext,
    ) : AutoCloseable {

        // ── Input contract discovery ──────────────────────────────────────────

        /**
         * Number of external inputs the plan expects per run.
         *
         * External inputs cover the model's *constants*, *variables* (weights),
         * and *placeholders* (runtime data), bound **positionally** in plan order.
         * Use [inputNames] to discover the full contract.
         */
        val numInputs: Int get() = delegate.numInputs()

        /** Number of outputs the plan produces per run. */
        val numOutputs: Int get() = delegate.numOutputs()

        /** All external input names in plan-binding order. */
        fun inputNames(): List<String> = delegate.inputNames().toList()

        /** Name of the external input at [index], or `null` if out of range. */
        fun inputName(index: Int): String? = delegate.inputName(index)

        /** Explicit requested output names in plan order. */
        fun outputNames(): List<String?> = delegate.outputNames().toList()

        /** Explicit requested output name at [index], if one was supplied. */
        fun outputName(index: Int): String? = delegate.outputName(index)

        /** Borrow the host-readable output from the most recent successful run. */
        fun outputTensor(index: Int): SdxRuntime.TensorView =
            delegate.outputTensor(index)

        // ── Lifecycle control ─────────────────────────────────────────────────

        /**
         * Marks input at [index] as a **variable** (value changes between runs;
         * shape is fixed). Enables D2D staging for this input. Call before the
         * first [run].
         */
        fun markInputVariable(index: Int): Unit = delegate.markInputVariable(index)

        /**
         * Marks input at [index] as a **placeholder** (value — and potentially
         * shape — may change between runs). Call before the first [run].
         */
        fun markInputPlaceholder(index: Int): Unit = delegate.markInputPlaceholder(index)

        /**
         * Convenience: marks every input whose name appears in [names] as a
         * placeholder. Returns the indices that were marked.
         */
        fun markPlaceholders(vararg names: String): List<Int> {
            val nameSet = names.toSet()
            return inputNames()
                .mapIndexedNotNull { i, name -> i.takeIf { name in nameSet } }
                .onEach { markInputPlaceholder(it) }
        }

        /**
         * Signals that input shapes are stable and will not change between future
         * runs. Triggers CUDA graph capture on the next execution, enabling the
         * [PlanPhase.REPLAYING] fast path.
         */
        fun freezeShapes(): Unit = delegate.freezeShapes()

        // ── Telemetry ─────────────────────────────────────────────────────────

        /**
         * Current DSP plan phase.
         *
         * Returns the [PlanPhase] enum when the code is recognised, or `null`
         * when the runtime returns an unrecognised value.
         */
        val phase: PlanPhase? get() = PlanPhase.fromCode(delegate.planPhase())

        /** Raw phase code (useful for logging when [phase] is null). */
        val phaseCode: Int get() = delegate.planPhase()

        /** Total number of executions completed on this context. -1 on error. */
        val executionCount: Int get() = delegate.executionCount()

        /**
         * Snapshot of the last execution's performance and configuration.
         *
         * Returns the idiomatic Kotlin [ExecutionReport] data class.
         */
        fun executionReport(): ExecutionReport = delegate.executionReport().toKotlin()

        // ── Execution ─────────────────────────────────────────────────────────

        /**
         * Runs the plan with the provided [inputs] and [outputs].
         *
         * Inputs and outputs are arrays of [SdxRuntime.TensorView]; use
         * [FloatTensor.view] to obtain views from host float data.
         *
         * @param inputs         Positional input tensors in plan-binding order.
         * @param outputs        Pre-allocated output tensors.
         * @param backend        Back-end override for this run.
         * @param strictSignature When `true`, validates the input/output signature.
         * @param gpuTarget      GPU device override for this run.
         */
        fun run(
            inputs: Array<SdxRuntime.TensorView>,
            outputs: Array<SdxRuntime.TensorView>,
            backend: SdxBackend = SdxBackend.AUTO,
            strictSignature: Boolean = true,
            gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO,
        ) {
            val opts = SdxRuntime.RunOptions().apply {
                this.backend = backend.code
                this.strict_signature = if (strictSignature) 1 else 0
                this.gpu_target = gpuTarget
                write()
            }
            delegate.run(inputs, outputs, opts)
        }

        /**
         * Runs with runtime-owned dynamic outputs. Returned tensor views borrow
         * native memory and remain valid only until the next run on this context.
         */
        fun runAllocating(
            inputs: Array<SdxRuntime.TensorView>,
            backend: SdxBackend = SdxBackend.AUTO,
            strictSignature: Boolean = true,
            gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO,
        ): Array<SdxRuntime.TensorView> {
            val opts = SdxRuntime.RunOptions().apply {
                this.backend = backend.code
                this.strict_signature = if (strictSignature) 1 else 0
                this.gpu_target = gpuTarget
                write()
            }
            return delegate.runAllocating(inputs, opts)
        }

        fun runAllocating(
            inputs: List<FloatTensor>,
            backend: SdxBackend = SdxBackend.AUTO,
            strictSignature: Boolean = true,
            gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO,
        ): List<SdxRuntime.TensorView> = runAllocating(
            inputs = inputs.map { it.view }.toTypedArray(),
            backend = backend,
            strictSignature = strictSignature,
            gpuTarget = gpuTarget,
        ).toList()

        /**
         * Runs the plan with [FloatTensor] inputs and outputs.
         *
         * This overload lets callers work entirely with [FloatTensor] and avoids
         * manual extraction of [SdxRuntime.TensorView].
         *
         * @param inputs  Positional input tensors in plan-binding order.
         * @param outputs Pre-allocated output tensors; values are updated in-place.
         */
        fun run(
            inputs: List<FloatTensor>,
            outputs: List<FloatTensor>,
            backend: SdxBackend = SdxBackend.AUTO,
            strictSignature: Boolean = true,
            gpuTarget: Int = SdxRuntime.SDX_GPU_TARGET_AUTO,
        ) = run(
            inputs = inputs.map { it.view }.toTypedArray(),
            outputs = outputs.map { it.view }.toTypedArray(),
            backend = backend,
            strictSignature = strictSignature,
            gpuTarget = gpuTarget,
        )

        /**
         * Runs the plan with a named input map; positional order is resolved
         * automatically via [inputNames].
         *
         * [weights] is a convenience parameter for the model's constants and
         * variables: supply them here to keep [inputs] limited to the
         * runtime-varying placeholders.
         *
         * ```kotlin
         * val result = FloatTensor.zeros(longArrayOf(2, 3))
         * ctx.runNamed(
         *     inputs  = mapOf("x" to xTensor),
         *     weights = weightsMap,
         *     outputs = listOf(result),
         * )
         * val probs = result.readBack()
         * ```
         *
         * @param inputs  Named tensors for placeholder inputs.
         * @param weights Named tensors for weight/constant inputs (merged with [inputs]).
         * @param outputs Pre-allocated output tensors.
         * @param backend Back-end override for this run.
         * @throws IllegalArgumentException if a required input name is missing.
         */
        fun runNamed(
            inputs: Map<String, FloatTensor>,
            weights: Map<String, FloatTensor> = emptyMap(),
            outputs: List<FloatTensor>,
            backend: SdxBackend = SdxBackend.AUTO,
            strictSignature: Boolean = true,
        ) {
            val all = weights + inputs
            val ordered = inputNames().map { name ->
                requireNotNull(all[name]) {
                    "Missing input '$name' — plan expects: ${inputNames().joinToString()}"
                }
            }
            run(
                inputs = ordered,
                outputs = outputs,
                backend = backend,
                strictSignature = strictSignature,
            )
        }

        override fun close(): Unit = delegate.close()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns `true` when the plan is actively replaying a captured CUDA / native
 * graph. Use this to confirm the fast-path is engaged.
 */
val KotlinSdxRuntime.KotlinSdxContext.isReplaying: Boolean
    get() = phase == PlanPhase.REPLAYING

/**
 * Returns `true` when the plan is still in warmup (slot-by-slot) mode.
 *
 * During warmup every run re-traces shapes through the plan interpreter.
 * After [KotlinSdxRuntime.KotlinSdxContext.freezeShapes] the plan transitions
 * to [PlanPhase.SHAPES_FROZEN] and then [PlanPhase.REPLAYING].
 */
val KotlinSdxRuntime.KotlinSdxContext.isWarmup: Boolean
    get() = phase == PlanPhase.SLOT_BY_SLOT

/**
 * Returns a display label for this phase, e.g. `"REPLAYING"` or `"UNKNOWN(99)"`.
 *
 * Useful when [KotlinSdxRuntime.KotlinSdxContext.phase] might be null (future
 * runtime versions with new phase codes).
 */
val KotlinSdxRuntime.KotlinSdxContext.phaseLabel: String
    get() = PlanPhase.label(phaseCode)
