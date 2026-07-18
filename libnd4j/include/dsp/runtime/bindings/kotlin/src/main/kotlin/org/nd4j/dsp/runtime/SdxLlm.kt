/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ******************************************************************************
 */
package org.nd4j.dsp.runtime

import java.nio.file.Path

// ─────────────────────────────────────────────────────────────────────────────
// Status codes — idiomatic Kotlin alternative to bare Int constants
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Status codes returned by the SDX LLM C ABI (`sdx_llm_c.h`).
 *
 * Values match the `sdx_llm_status_t` C enum.
 */
enum class SdxLlmStatus(val code: Int) {
    OK(SdxLlm.SDX_LLM_STATUS_OK),
    INVALID_ARGUMENT(SdxLlm.SDX_LLM_STATUS_INVALID_ARGUMENT),
    MODEL_LOAD_FAILED(SdxLlm.SDX_LLM_STATUS_MODEL_LOAD_FAILED),
    EXECUTION_FAILED(SdxLlm.SDX_LLM_STATUS_EXECUTION_FAILED),
    IO_ERROR(SdxLlm.SDX_LLM_STATUS_IO_ERROR);

    companion object {
        /** Returns the enum constant for [code], or null if unrecognised. */
        fun fromCode(code: Int): SdxLlmStatus? = entries.firstOrNull { it.code == code }
        /** Human-readable label; never throws. */
        fun label(code: Int): String = fromCode(code)?.name ?: "UNKNOWN($code)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation stats — typed data class parsed from lastResultJson
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parsed subset of the JSON returned by `sdxLlmLastResultJson`.
 *
 * All fields are nullable because the JSON schema may evolve across SDK versions.
 * The full JSON string is available in [rawJson] for forward-compatibility.
 *
 * @param rawJson        Full JSON string from `sdxLlmLastResultJson`.
 * @param promptTokens   Number of input (prompt) tokens processed.
 * @param newTokens      Number of tokens generated.
 * @param tokensPerSec   Generation throughput in tokens per second.
 * @param finishReason   Why generation stopped (e.g. `"max_tokens"`, `"eos"`).
 */
data class LlmResultStats(
    val rawJson: String,
    val promptTokens: Int?,
    val newTokens: Int?,
    val tokensPerSec: Double?,
    val finishReason: String?,
) {
    fun summary(): String = buildString {
        appendLine("LlmResultStats {")
        appendLine("  promptTokens = $promptTokens")
        appendLine("  newTokens    = $newTokens")
        appendLine("  tokensPerSec = ${tokensPerSec?.let { "%.2f".format(it) } ?: "n/a"}")
        append("  finishReason = $finishReason")
        append("\n}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kotlin facade — KotlinSdxLlmRuntime
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Idiomatic Kotlin facade over [SdxLlm] (the JNA-based Java wrapper for
 * the SDX LLM C ABI, `sdx_llm_c.h`).
 *
 * This wrapper binds `libsdx_llm.so` — the AOT-compiled (GraalVM native-image)
 * LLM/VLM/STT library — via JNA.  **No JVM is embedded in the library.**
 *
 * All lifecycle objects ([KotlinSdxLlmRuntime], [KotlinSdxLlmModel]) implement
 * [AutoCloseable] and are designed for Kotlin's `use {}` idiom:
 *
 * ```kotlin
 * KotlinSdxLlmRuntime.create().use { runtime ->
 *     runtime.loadModel(modelPath, tokenizerPath).use { model ->
 *         val text = model.generate(
 *             prompt = "The capital of France is",
 *             optionsJson = """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}""",
 *         )
 *         println(text)            // " Paris."
 *         println(model.lastResultStats().summary())
 *     }
 * }
 * ```
 *
 * ## Threading (v1)
 *
 * The underlying runtime handle is bound to the OS thread that created it.
 * Create, use, and destroy the runtime from **one thread**. For concurrent
 * generation, create one [KotlinSdxLlmRuntime] per thread — models are not
 * shared across runtimes.
 *
 * ## Side-loaded natives (CRITICAL)
 *
 * `libsdx_llm.so` resolves its companion natives (ND4J CPU kernels, tokenizers,
 * etc.) relative to the host executable using `SDX_NATIVE_LIB_DIR`. A JVM
 * **cannot** set process environment after start. The runner must export the
 * variable before launching:
 * ```bash
 * export SDX_LLM_AOT_HOME=/tmp/sdx-sdk
 * export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 * gradle run --args="<model> <tokenizer>"
 * ```
 *
 * @see SdxLlm — the underlying JNA Java wrapper.
 */
class KotlinSdxLlmRuntime private constructor(
    private val delegate: SdxLlm.Runtime,
) : AutoCloseable {

    // ── Factory ──────────────────────────────────────────────────────────────

    companion object {
        /** ABI version this wrapper targets. */
        const val SDX_LLM_ABI_VERSION: Int = SdxLlm.SDX_LLM_ABI_VERSION

        /**
         * Creates a runtime by auto-detecting `libsdx_llm.so`.
         *
         * Resolution order (first match wins):
         * 1. JVM property `sdx.llm.library` or env `SDX_LLM_LIBRARY`.
         * 2. `SDX_LLM_AOT_HOME/lib/libsdx_llm.so`.
         * 3. Bare names `sdx_llm` / `sdx_llm_cpu` via JNA search.
         *
         * @throws IllegalStateException if the library or runtime cannot be loaded.
         */
        @JvmStatic
        fun create(): KotlinSdxLlmRuntime = KotlinSdxLlmRuntime(SdxLlm.Runtime.create())

        /**
         * Creates a runtime loading the library at [libraryPath].
         *
         * @param libraryPath Absolute path or bare JNA library name (e.g. `"sdx_llm"`).
         *                    Pass `null` or blank to fall back to auto-discovery.
         */
        @JvmStatic
        fun create(libraryPath: String?): KotlinSdxLlmRuntime {
            val rt = if (libraryPath.isNullOrBlank()) {
                SdxLlm.Runtime.create()
            } else {
                SdxLlm.Runtime.create(libraryPath)
            }
            return KotlinSdxLlmRuntime(rt)
        }
    }

    // ── Runtime API ──────────────────────────────────────────────────────────

    /**
     * ABI version reported by the loaded library.
     * Compare with [SDX_LLM_ABI_VERSION] at startup.
     */
    fun abiVersion(): Int = delegate.abiVersion()

    /** Last error message from the native runtime, or empty string. */
    fun lastError(): String = delegate.lastError()

    /**
     * Loads a GGUF/SDZ model and builds the generation pipeline.
     *
     * First-load DSP plan compilation takes 30–90 s on CPU; subsequent loads
     * reuse the plan cache.
     *
     * @param modelPath     Path to the model file (`.gguf` / `.sdz` / `.sdnb`).
     * @param tokenizerPath Path to `tokenizer.json` or its directory, or `null`
     *                      to probe the model directory.
     * @param optionsJson   Optional JSON defaults, e.g.
     *                      `{"maxNewTokens":128,"sampling":{"preset":"greedy"}}`.
     *                      `null` uses library defaults.
     * @return A new [KotlinSdxLlmModel]; close it when done.
     * @throws IllegalStateException if loading fails.
     */
    fun loadModel(
        modelPath: String,
        tokenizerPath: String? = null,
        optionsJson: String? = null,
    ): KotlinSdxLlmModel = KotlinSdxLlmModel(
        delegate.loadModel(modelPath, tokenizerPath, optionsJson)
    )

    /**
     * VLM document extraction (SmolDocling) — stateless, loads and releases per call.
     *
     * @param modelPath     Path to the SmolDocling model directory.
     * @param tokenizerPath Path to `tokenizer.json` or `null` for auto-discovery.
     * @param inputPath     Path to an image (PNG/JPEG) or PDF file.
     * @param optionsJson   Optional JSON, e.g. `{"maxNewTokens":512,"format":"doctags"}`.
     * @return Extracted text in the requested format.
     * @throws IllegalStateException if extraction fails.
     */
    fun vlmExtract(
        modelPath: String,
        tokenizerPath: String? = null,
        inputPath: String,
        optionsJson: String? = null,
    ): String = delegate.vlmExtract(modelPath, tokenizerPath, inputPath, optionsJson)

    /**
     * Whisper speech-to-text — stateless, loads and releases per call.
     *
     * @param modelPath    Path to a Whisper ONNX model directory.
     * @param audioPath    Path to a WAV file (16 kHz mono recommended).
     * @param optionsJson  Optional JSON, e.g. `{"language":"en","maxNewTokens":448}`.
     * @return Transcription text (UTF-8).
     * @throws IllegalStateException if transcription fails.
     */
    fun audioTranscribe(
        modelPath: String,
        audioPath: String,
        optionsJson: String? = null,
    ): String = delegate.audioTranscribe(modelPath, audioPath, optionsJson)

    override fun close(): Unit = delegate.close()

    // ─────────────────────────────────────────────────────────────────────────
    // Model
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A loaded LLM/VLM model.
     *
     * Obtain via [KotlinSdxLlmRuntime.loadModel]; always close via `use {}`.
     * Thread-bound to the same OS thread as the parent [KotlinSdxLlmRuntime].
     */
    class KotlinSdxLlmModel internal constructor(
        private val delegate: SdxLlm.Model,
    ) : AutoCloseable {

        // ── Generation ────────────────────────────────────────────────────────

        /**
         * Generates a text continuation for [prompt].
         *
         * The pipeline's compiled plan and KV state are reused across calls on
         * the same model (stateful decode). Do not create a new model for each
         * conversation turn — reuse the same model.
         *
         * @param prompt      Input text (UTF-8).
         * @param optionsJson Per-call overrides, e.g.
         *                    `{"maxNewTokens":64,"sampling":{"temperature":0.8}}`,
         *                    or `null` for load-time defaults.
         * @return The generated continuation (not the prompt).
         * @throws IllegalStateException on generation failure.
         */
        fun generate(prompt: String, optionsJson: String? = null): String =
            delegate.generate(prompt, optionsJson)

        // ── Stats + info ──────────────────────────────────────────────────────

        /**
         * Returns a [LlmResultStats] parsed from the stats JSON of the most
         * recent [generate] call (token counts, tok/s, finish reason).
         *
         * @throws IllegalStateException if the ABI call fails.
         */
        fun lastResultStats(): LlmResultStats {
            val json = lastResultJson()
            return LlmResultStats(
                rawJson      = json,
                promptTokens = extractInt(json, "promptTokens"),
                newTokens    = extractInt(json, "newTokens"),
                tokensPerSec = extractDouble(json, "tokensPerSec"),
                finishReason = extractString(json, "finishReason"),
            )
        }

        /**
         * Raw JSON string from `sdxLlmLastResultJson`.
         * Callers normally use [lastResultStats].
         *
         * @throws IllegalStateException if the ABI call fails.
         */
        fun lastResultJson(): String = delegate.lastResultJson()

        /**
         * JSON summary of this model and its tokenizer: inputs, outputs,
         * vocab size, and whether a chat template is present.
         *
         * @throws IllegalStateException if the ABI call fails.
         */
        fun infoJson(): String = delegate.infoJson()

        // ── Tokenization ──────────────────────────────────────────────────────

        /**
         * Encodes [text] to token IDs.
         *
         * @param text             Input text (UTF-8).
         * @param addSpecialTokens `true` to prepend/append BOS/EOS tokens.
         * @return [IntArray] of token IDs.
         * @throws IllegalStateException if tokenization fails.
         */
        fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray =
            delegate.tokenize(text, addSpecialTokens)

        /**
         * Decodes [ids] back to text.
         *
         * @param ids               Token ID array.
         * @param skipSpecialTokens `true` to omit BOS/EOS/pad from output.
         * @return Decoded text (UTF-8).
         * @throws IllegalStateException if detokenization fails.
         */
        fun detokenize(ids: IntArray, skipSpecialTokens: Boolean = true): String =
            delegate.detokenize(ids, skipSpecialTokens)

        override fun close(): Unit = delegate.close()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON extraction helpers (no external JSON library dependency)
// ─────────────────────────────────────────────────────────────────────────────

private val LLM_INT_PATTERN    = Regex(""""(\w+)"\s*:\s*(-?\d+)""")
private val LLM_DOUBLE_PATTERN = Regex(""""(\w+)"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)""")
private val LLM_STRING_PATTERN = Regex(""""(\w+)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"""")

internal fun extractInt(json: String, key: String): Int? =
    LLM_INT_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)?.toIntOrNull()

internal fun extractDouble(json: String, key: String): Double? =
    LLM_DOUBLE_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)?.toDoubleOrNull()

internal fun extractString(json: String, key: String): String? =
    LLM_STRING_PATTERN.findAll(json).firstOrNull { it.groupValues[1] == key }
        ?.groupValues?.get(2)
