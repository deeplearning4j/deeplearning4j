/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ******************************************************************************
 */
package org.nd4j.dsp.runtime;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

import java.io.File;
import java.nio.charset.StandardCharsets;

/**
 * Java wrapper for the SDX LLM C ABI ({@code sdx_llm_c.h}).
 *
 * <p>This class uses JNA to bind {@code libsdx_llm.so} — the AOT-compiled
 * (GraalVM native-image) LLM/VLM/STT library — from a standard JVM process.
 * <b>No JVM is embedded in the library.</b> This wrapper is the canonical way
 * to call the library from Java without any {@code samediff-llm} or ND4J
 * dependencies on the classpath.
 *
 * <p>The outer class ({@code SdxLlm}) acts as the entry-point / factory.
 * Inner classes {@link SdxLlm.Runtime} and {@link SdxLlm.Model} own the
 * native lifecycle (both implement {@link AutoCloseable} for try-with-resources).
 *
 * <h2>Minimal usage</h2>
 * <pre>{@code
 * try (SdxLlm.Runtime rt = SdxLlm.Runtime.create()) {
 *     try (SdxLlm.Model model = rt.loadModel(modelPath, tokenizerPath, null)) {
 *         String text = model.generate("The capital of France is",
 *                 "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
 *         System.out.println(text);  // " Paris."
 *         System.out.println(model.lastResultJson());
 *     }
 * }
 * }</pre>
 *
 * <h2>Threading (v1)</h2>
 * <p>The runtime handle is bound to the OS thread that created it. Create, use,
 * and destroy a runtime from <em>one thread</em>. For concurrent generation,
 * create one {@code Runtime} per thread — models are not shared across runtimes.
 *
 * <h2>Library resolution</h2>
 * <p>Resolution order (first match wins):
 * <ol>
 *   <li>JVM system property {@code sdx.llm.library} or env-var
 *       {@code SDX_LLM_LIBRARY} — explicit path or bare name.</li>
 *   <li>{@code $SDX_LLM_AOT_HOME/lib/libsdx_llm.so} — preferred; mirrors the
 *       unpacked AOT SDK ZIP layout. Also sets {@code jna.library.path} to
 *       {@code $SDX_LLM_AOT_HOME/lib} so that side-loaded dependencies are
 *       found by the OS loader.</li>
 *   <li>Bare names {@code sdx_llm} / {@code sdx_llm_cpu} via JNA's default
 *       library search ({@code LD_LIBRARY_PATH}, {@code java.library.path},
 *       etc.).</li>
 * </ol>
 *
 * <h2>Side-loaded natives — CRITICAL</h2>
 * <p>{@code libsdx_llm.so} resolves its companion natives (ND4J CPU kernels,
 * tokenizer shared libraries, etc.) relative to the <em>host executable</em>
 * using the environment variable {@code SDX_NATIVE_LIB_DIR}. A JVM process
 * <b>cannot set process environment after start</b> — the runner must export
 * the variable <em>before</em> launching the JVM:
 * <pre>{@code
 *   export SDX_LLM_AOT_HOME=/path/to/sdx-sdk
 *   export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
 *   java -jar my-app.jar
 * }</pre>
 * A loud warning (including corrective instructions) is printed at runtime when
 * {@code SDX_LLM_AOT_HOME} is set but {@code SDX_NATIVE_LIB_DIR} is missing.
 * An {@link IllegalStateException} with a descriptive message is thrown when
 * {@code sdxLlmCreateRuntime()} returns null, which almost always indicates
 * that {@code SDX_NATIVE_LIB_DIR} was not set before JVM launch.
 *
 * <h2>ABI version</h2>
 * <p>This wrapper targets ABI version {@link #SDX_LLM_ABI_VERSION}. Compare
 * with {@link Runtime#abiVersion()} at startup; a mismatch indicates a
 * header / library version skew.
 */
public final class SdxLlm {

    /** ABI version this wrapper was written against ({@code sdx_llm_c.h}). */
    public static final int SDX_LLM_ABI_VERSION = 1;

    // ── Status codes (sdx_llm_status_t) ──────────────────────────────────────

    /** Successful completion. */
    public static final int SDX_LLM_STATUS_OK               = 0;
    /** Caller supplied a null or invalid argument. */
    public static final int SDX_LLM_STATUS_INVALID_ARGUMENT  = 1;
    /** Model file not found or not parseable. */
    public static final int SDX_LLM_STATUS_MODEL_LOAD_FAILED = 3;
    /** Generation / tokenization failed at runtime. */
    public static final int SDX_LLM_STATUS_EXECUTION_FAILED  = 4;
    /** I/O error (file read, audio decode, …). */
    public static final int SDX_LLM_STATUS_IO_ERROR          = 6;

    // ── Low-level JNA ABI binding ─────────────────────────────────────────────

    /**
     * Package-private JNA binding for {@code sdx_llm_c.h}.
     *
     * <p>Application code uses the higher-level {@link Runtime} / {@link Model}
     * API instead of this interface.
     */
    interface NativeApi extends Library {

        /** Creates a GraalVM isolate (runtime). Returns {@code null} on failure. */
        Pointer sdxLlmCreateRuntime();

        /** Tears down the runtime. All model handles become invalid. Returns 0 on success. */
        int sdxLlmDestroyRuntime(Pointer runtime);

        /** Returns the ABI version compiled into the library. */
        int sdxLlmAbiVersion(Pointer runtime);

        /**
         * Loads a model and builds the generation pipeline. Returns {@code null} on failure.
         *
         * @param runtime        Runtime handle.
         * @param model_path     Path to {@code .gguf}/{@code .ggml}/{@code .sdz}/{@code .sdnb}.
         * @param tokenizer_path Path to {@code tokenizer.json} or directory; {@code null} to probe.
         * @param options_json   Optional JSON, e.g.
         *                       {@code {"maxNewTokens":128,"sampling":{"preset":"greedy"}}};
         *                       {@code null} for library defaults.
         */
        Pointer sdxLlmLoadModel(Pointer runtime,
                                 String model_path,
                                 String tokenizer_path,
                                 String options_json);

        /** Unloads a model and releases associated resources. Returns 0 on success. */
        int sdxLlmUnloadModel(Pointer runtime, Pointer model);

        /**
         * Blocking text generation.
         *
         * <p>{@code *out_text} receives a malloc'd UTF-8 string owned by the caller;
         * release with {@link #sdxLlmFree}. The pipeline's compiled plan and KV state
         * are reused across calls.
         */
        int sdxLlmGenerate(Pointer runtime,
                           Pointer model,
                           String prompt,
                           String options_json,
                           PointerByReference out_text);

        /**
         * Returns a JSON string with stats for the most recent
         * {@link #sdxLlmGenerate}: token counts, timings, tok/s, finish reason.
         * Caller frees {@code *out_json} with {@link #sdxLlmFree}.
         */
        int sdxLlmLastResultJson(Pointer runtime,
                                  Pointer model,
                                  PointerByReference out_json);

        /**
         * Returns a model/tokenizer summary JSON (inputs, outputs, vocab size,
         * chat-template flag). Caller frees {@code *out_json} with {@link #sdxLlmFree}.
         */
        int sdxLlmInfoJson(Pointer runtime,
                           Pointer model,
                           PointerByReference out_json);

        /**
         * Encodes {@code text} to token IDs. {@code *out_ids} receives a malloc'd
         * {@code int32} array; {@code *out_count} its length. Caller frees
         * {@code *out_ids} with {@link #sdxLlmFree}.
         *
         * @param add_special_tokens Non-zero to prepend/append BOS/EOS tokens.
         */
        int sdxLlmTokenize(Pointer runtime,
                           Pointer model,
                           String text,
                           int add_special_tokens,
                           PointerByReference out_ids,
                           IntByReference out_count);

        /**
         * Decodes {@code count} token IDs to text. {@code *out_text} receives a
         * malloc'd UTF-8 string. Caller frees with {@link #sdxLlmFree}.
         *
         * @param skip_special_tokens Non-zero to omit BOS/EOS/pad from output.
         */
        int sdxLlmDetokenize(Pointer runtime,
                             Pointer model,
                             int[] ids,
                             int count,
                             int skip_special_tokens,
                             PointerByReference out_text);

        /**
         * VLM document extraction (SmolDocling) — stateless, loads/releases per call.
         * Caller frees {@code *out_text} with {@link #sdxLlmFree}.
         */
        int sdxVlmExtract(Pointer runtime,
                          String model_path,
                          String tokenizer_path,
                          String input_path,
                          String options_json,
                          PointerByReference out_text);

        /**
         * Whisper speech-to-text — stateless, loads/releases per call.
         * Caller frees {@code *out_text} with {@link #sdxLlmFree}.
         */
        int sdxAudioTranscribe(Pointer runtime,
                               String model_path,
                               String audio_path,
                               String options_json,
                               PointerByReference out_text);

        /**
         * Releases any pointer returned through an out-parameter of this ABI.
         * Always prefer this over {@code free()} — the library manages its own heap.
         */
        void sdxLlmFree(Pointer runtime, Pointer pointer);

        /**
         * Copies the last error message into {@code buffer}.
         *
         * @param buffer   Caller-allocated buffer (may be {@code null} if capacity is 0).
         * @param capacity Size of {@code buffer} in bytes.
         * @return Full message length in bytes (excluding NUL), or 0 if no error.
         */
        int sdxLlmGetLastError(Pointer runtime, byte[] buffer, int capacity);
    }

    // Private constructor — this class is a namespace only.
    private SdxLlm() {}

    // ── Library resolution ────────────────────────────────────────────────────

    private static void warnIfNativeLibDirMissing() {
        String aotHome    = System.getenv("SDX_LLM_AOT_HOME");
        String nativeLibDir = System.getenv("SDX_NATIVE_LIB_DIR");
        if (aotHome != null && !aotHome.isEmpty()
                && (nativeLibDir == null || nativeLibDir.isEmpty())) {
            System.err.println(
                "[SdxLlm] WARNING: SDX_LLM_AOT_HOME is set but SDX_NATIVE_LIB_DIR is not.\n" +
                "  libsdx_llm.so resolves its side-loaded natives relative to the host executable.\n" +
                "  Export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib BEFORE starting the JVM, e.g.:\n" +
                "    export SDX_NATIVE_LIB_DIR=" + aotHome + "/lib");
        }
    }

    private static String resolveLibrary() {
        // 1) Explicit override via system property or env-var.
        String explicit = System.getProperty("sdx.llm.library",
                System.getenv("SDX_LLM_LIBRARY"));
        if (explicit != null && !explicit.isEmpty()) {
            return explicit;
        }

        // 2) AOT SDK unpacked at SDX_LLM_AOT_HOME.
        String aotHome = System.getenv("SDX_LLM_AOT_HOME");
        if (aotHome != null && !aotHome.isEmpty()) {
            for (String name : new String[]{"libsdx_llm.so", "sdx_llm.so", "libsdx_llm.dylib"}) {
                File candidate = new File(aotHome, "lib/" + name);
                if (candidate.exists()) {
                    // Extend jna.library.path so side-loaded deps are found.
                    String jnaPath = System.getProperty("jna.library.path", "");
                    String libDir  = new File(aotHome, "lib").getAbsolutePath();
                    if (!jnaPath.contains(libDir)) {
                        System.setProperty("jna.library.path",
                            jnaPath.isEmpty() ? libDir : libDir + File.pathSeparator + jnaPath);
                    }
                    return candidate.getAbsolutePath();
                }
            }
        }

        // 3) JNA bare-name fallback.
        return "sdx_llm";
    }

    // ── Runtime ───────────────────────────────────────────────────────────────

    /**
     * Root object for the SDX LLM runtime — analogous to {@code OrtEnvironment}
     * in the ONNX Runtime Java API.
     *
     * <p>There is typically one {@code Runtime} per process (or per dedicated
     * generation thread). It owns the native {@code sdx_llm_runtime_t*} and is
     * the factory for {@link Model} objects.
     *
     * <p><b>Threading contract (v1):</b> the runtime handle is bound to the OS
     * thread that created it. Create, use, and destroy the runtime on a single
     * thread. For concurrent generation, create one {@code Runtime} per thread.
     *
     * <p>Always close the runtime after all models created from it have been
     * closed; prefer try-with-resources.
     */
    public static final class Runtime implements AutoCloseable {

        final NativeApi api;
        final Pointer handle;
        private volatile boolean closed = false;

        private Runtime(NativeApi api, Pointer handle) {
            this.api    = api;
            this.handle = handle;
        }

        // ── Factory ───────────────────────────────────────────────────────────

        /**
         * Creates a runtime by auto-detecting {@code libsdx_llm.so}.
         *
         * @throws IllegalStateException if the library cannot be found or the
         *                               runtime cannot be initialized.
         */
        public static Runtime create() {
            warnIfNativeLibDirMissing();
            return create(resolveLibrary());
        }

        /**
         * Creates a runtime using an explicit library path or bare JNA name.
         *
         * @param libraryPath Absolute path to {@code libsdx_llm.so} or a bare
         *                    JNA library name, e.g. {@code "sdx_llm"}.
         * @throws IllegalStateException if the runtime cannot be initialized.
         */
        public static Runtime create(String libraryPath) {
            warnIfNativeLibDirMissing();
            NativeApi api = Native.load(libraryPath, NativeApi.class);
            Pointer   rt  = api.sdxLlmCreateRuntime();
            if (rt == null) {
                throw new IllegalStateException(
                    "sdxLlmCreateRuntime() returned null — check that " +
                    "SDX_NATIVE_LIB_DIR is set before starting the JVM " +
                    "(export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib).");
            }
            return new Runtime(api, rt);
        }

        // ── Runtime API ───────────────────────────────────────────────────────

        /**
         * Returns the ABI version reported by the loaded library.
         * Compare with {@link SdxLlm#SDX_LLM_ABI_VERSION} at startup.
         */
        public int abiVersion() {
            checkNotClosed();
            return api.sdxLlmAbiVersion(handle);
        }

        /**
         * Loads a GGUF/SDZ/SameDiff model and builds the generation pipeline.
         * Analogous to {@code new OrtSession(env, modelPath, options)}.
         *
         * <p><b>Note:</b> first-load DSP plan compilation takes 30–90 s on CPU.
         * Subsequent calls reuse the cached plan.
         *
         * @param modelPath     Path to the model file (e.g. {@code .gguf}).
         * @param tokenizerPath Path to {@code tokenizer.json} or its directory; pass
         *                      {@code null} to probe the model's directory.
         * @param optionsJson   Optional generation defaults JSON, e.g.
         *                      {@code {"maxNewTokens":128,"sampling":{"preset":"greedy"}}}.
         *                      Pass {@code null} for library defaults.
         * @return A new {@link Model}; close it when done.
         * @throws IllegalStateException if loading fails.
         */
        public Model loadModel(String modelPath, String tokenizerPath, String optionsJson) {
            checkNotClosed();
            Pointer model = api.sdxLlmLoadModel(handle, modelPath, tokenizerPath, optionsJson);
            if (model == null) {
                throw new IllegalStateException(
                    "sdxLlmLoadModel failed: " + lastError() +
                    "\n  model_path=" + modelPath);
            }
            return new Model(this, model);
        }

        /**
         * Returns the last error string from the native runtime, or an empty string.
         */
        public String lastError() {
            if (closed || handle == null) return "";
            byte[] buf = new byte[2048];
            int len = api.sdxLlmGetLastError(handle, buf, buf.length);
            if (len <= 0) return "";
            return new String(buf, 0, Math.min(len, buf.length - 1), StandardCharsets.UTF_8);
        }

        /**
         * VLM document extraction (SmolDocling) — stateless, loads and releases the
         * model per call. Use for one-off extractions; for high-throughput pipelines
         * prefer the stateful future VLM API.
         *
         * @param modelPath     Path to the SmolDocling model directory.
         * @param tokenizerPath Path to {@code tokenizer.json} or {@code null}.
         * @param inputPath     Path to an image (PNG/JPEG) or PDF file.
         * @param optionsJson   e.g. {@code {"maxNewTokens":512,"format":"doctags"}}.
         * @return Extracted text in the requested format.
         * @throws IllegalStateException if extraction fails.
         */
        public String vlmExtract(String modelPath, String tokenizerPath,
                                  String inputPath, String optionsJson) {
            checkNotClosed();
            PointerByReference outText = new PointerByReference();
            int status = api.sdxVlmExtract(handle, modelPath, tokenizerPath,
                                            inputPath, optionsJson, outText);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxVlmExtract failed (status=" + status + "): " + lastError());
            }
            return readAndFree(outText.getValue());
        }

        /**
         * Whisper speech-to-text — stateless, loads and releases the model per call.
         *
         * @param modelPath   Path to a Whisper ONNX model directory.
         * @param audioPath   Path to a WAV file (16 kHz mono recommended).
         * @param optionsJson e.g. {@code {"language":"en","maxNewTokens":448}}.
         * @return Transcription text (UTF-8).
         * @throws IllegalStateException if transcription fails.
         */
        public String audioTranscribe(String modelPath, String audioPath, String optionsJson) {
            checkNotClosed();
            PointerByReference outText = new PointerByReference();
            int status = api.sdxAudioTranscribe(handle, modelPath, audioPath,
                                                 optionsJson, outText);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxAudioTranscribe failed (status=" + status + "): " + lastError());
            }
            return readAndFree(outText.getValue());
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                api.sdxLlmDestroyRuntime(handle);
            }
        }

        // ── Internal helpers ──────────────────────────────────────────────────

        private void checkNotClosed() {
            if (closed) {
                throw new IllegalStateException("SdxLlm.Runtime has been closed");
            }
        }

        String readAndFree(Pointer ptr) {
            if (ptr == null) return "";
            try {
                String value = ptr.getString(0, StandardCharsets.UTF_8.name());
                return value == null ? "" : value;
            } finally {
                api.sdxLlmFree(handle, ptr);
            }
        }
    }

    // ── Model ─────────────────────────────────────────────────────────────────

    /**
     * A loaded LLM/VLM model in the SDX runtime.
     *
     * <p>Obtain via {@link Runtime#loadModel}. The model is thread-bound to the
     * same OS thread as its parent {@link Runtime}.
     *
     * <p>Always close the model (via try-with-resources or explicitly) before
     * closing the parent runtime.
     */
    public static final class Model implements AutoCloseable {

        private final Runtime runtime;
        private Pointer modelHandle;
        private volatile boolean closed = false;

        Model(Runtime runtime, Pointer modelHandle) {
            this.runtime     = runtime;
            this.modelHandle = modelHandle;
        }

        // ── Generation ────────────────────────────────────────────────────────

        /**
         * Generates text for {@code prompt} and returns the generated continuation.
         *
         * <p>The pipeline's compiled plan and KV state are reused across calls on
         * the same model (stateful decode). On the first call the SameDiff graph is
         * compiled (DSP warmup); subsequent calls are much faster.
         *
         * @param prompt      The input prompt (UTF-8).
         * @param optionsJson Optional per-call overrides, e.g.
         *                    {@code {"maxNewTokens":64,"sampling":{"temperature":0.8}}}.
         *                    Pass {@code null} to use load-time defaults.
         * @return The generated text (the continuation, <b>not</b> the prompt).
         * @throws IllegalStateException if generation fails.
         */
        public String generate(String prompt, String optionsJson) {
            checkNotClosed();
            PointerByReference outText = new PointerByReference();
            int status = runtime.api.sdxLlmGenerate(
                    runtime.handle, modelHandle, prompt, optionsJson, outText);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxLlmGenerate failed (status=" + status + "): " + runtime.lastError());
            }
            return runtime.readAndFree(outText.getValue());
        }

        /**
         * Convenience overload — uses load-time generation defaults.
         */
        public String generate(String prompt) {
            return generate(prompt, null);
        }

        // ── Stats + info ──────────────────────────────────────────────────────

        /**
         * Returns a JSON string with stats for the most recent {@link #generate}
         * call: token counts, timings, tok/s, finish reason.
         *
         * @throws IllegalStateException if the ABI call fails.
         */
        public String lastResultJson() {
            checkNotClosed();
            PointerByReference outJson = new PointerByReference();
            int status = runtime.api.sdxLlmLastResultJson(runtime.handle, modelHandle, outJson);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxLlmLastResultJson failed (status=" + status + "): " + runtime.lastError());
            }
            return runtime.readAndFree(outJson.getValue());
        }

        /**
         * Returns a JSON summary of this model and its tokenizer: inputs, outputs,
         * vocab size, and whether a chat template is present.
         *
         * @throws IllegalStateException if the ABI call fails.
         */
        public String infoJson() {
            checkNotClosed();
            PointerByReference outJson = new PointerByReference();
            int status = runtime.api.sdxLlmInfoJson(runtime.handle, modelHandle, outJson);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxLlmInfoJson failed (status=" + status + "): " + runtime.lastError());
            }
            return runtime.readAndFree(outJson.getValue());
        }

        // ── Tokenization ──────────────────────────────────────────────────────

        /**
         * Encodes {@code text} to an array of token IDs.
         *
         * @param text             Input text (UTF-8).
         * @param addSpecialTokens {@code true} to prepend/append BOS/EOS tokens.
         * @return Array of token IDs.
         * @throws IllegalStateException if tokenization fails.
         */
        public int[] tokenize(String text, boolean addSpecialTokens) {
            checkNotClosed();
            PointerByReference outIds   = new PointerByReference();
            IntByReference     outCount = new IntByReference();
            int status = runtime.api.sdxLlmTokenize(
                runtime.handle, modelHandle, text,
                addSpecialTokens ? 1 : 0,
                outIds, outCount);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxLlmTokenize failed (status=" + status + "): " + runtime.lastError());
            }
            int     count  = outCount.getValue();
            Pointer idsPtr = outIds.getValue();
            int[]   ids    = idsPtr.getIntArray(0, count);
            runtime.api.sdxLlmFree(runtime.handle, idsPtr);
            return ids;
        }

        /**
         * Decodes an array of token IDs back to text.
         *
         * @param ids               Token ID array.
         * @param skipSpecialTokens {@code true} to omit BOS/EOS/pad from output.
         * @return Decoded text (UTF-8).
         * @throws IllegalStateException if detokenization fails.
         */
        public String detokenize(int[] ids, boolean skipSpecialTokens) {
            checkNotClosed();
            PointerByReference outText = new PointerByReference();
            int status = runtime.api.sdxLlmDetokenize(
                runtime.handle, modelHandle,
                ids, ids.length,
                skipSpecialTokens ? 1 : 0,
                outText);
            if (status != SDX_LLM_STATUS_OK) {
                throw new IllegalStateException(
                    "sdxLlmDetokenize failed (status=" + status + "): " + runtime.lastError());
            }
            return runtime.readAndFree(outText.getValue());
        }

        // ── Resource management ───────────────────────────────────────────────

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                if (modelHandle != null) {
                    runtime.api.sdxLlmUnloadModel(runtime.handle, modelHandle);
                    modelHandle = null;
                }
            }
        }

        // ── Internal helpers ──────────────────────────────────────────────────

        private void checkNotClosed() {
            if (closed) {
                throw new IllegalStateException("SdxLlm.Model has been closed");
            }
        }
    }
}
