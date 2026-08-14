/* ******************************************************************************
 *
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
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.function.CFunctionPointer;
import org.graalvm.nativeimage.c.function.InvokeCFunctionPointer;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CCharPointerPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.WordPointer;
import org.graalvm.word.WordFactory;

import java.nio.charset.StandardCharsets;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * C ABI of {@code libsdx_llm} — the AOT-compiled LLM surface of the SDX SDK.
 * The stable contract is {@code include/sdx_llm_c.h}; every entrypoint here must
 * stay in sync with that header. Conventions mirror {@code dsp_runtime_c.h}:
 * {@code sdxLlm} prefix, integer status codes, out-parameters, last-error query.
 *
 * <p>Threading (v1): a runtime handle is an isolate thread — create, use and destroy
 * it from the same OS thread. Multi-thread attachment is a v2 extension.</p>
 */
public final class SdxLlmCApi {

    /** Keep in sync with SDX_LLM_ABI_VERSION in sdx_llm_c.h. */
    static final int ABI_VERSION = 2;

    /* Status codes — mirrors the sdx_status_t subset used by this ABI. */
    static final int OK = 0;
    static final int INVALID_ARGUMENT = 1;
    static final int MODEL_LOAD_FAILED = 3;
    static final int EXECUTION_FAILED = 4;
    static final int IO_ERROR = 6;

    private static volatile String lastError = "";

    private SdxLlmCApi() {
    }

    /* ── Runtime lifecycle ──────────────────────────────────────────────── */

    @CEntryPoint(name = "sdxLlmCreateRuntime", builtin = CEntryPoint.Builtin.CREATE_ISOLATE)
    public static native IsolateThread sdxLlmCreateRuntime();

    @CEntryPoint(name = "sdxLlmDestroyRuntime", builtin = CEntryPoint.Builtin.TEAR_DOWN_ISOLATE)
    public static native int sdxLlmDestroyRuntime(IsolateThread runtime);

    @CEntryPoint(name = "sdxLlmAbiVersion")
    public static int sdxLlmAbiVersion(IsolateThread runtime) {
        return ABI_VERSION;
    }

    /* ── Canonical model preparation/resolution ─────────────────────────── */

    @CEntryPoint(name = "sdxLlmPrepareGguf")
    public static int sdxLlmPrepareGguf(IsolateThread runtime, CCharPointer sourceGguf,
                                        CCharPointer tokenizerPath, CCharPointer targetProfile,
                                        CCharPointer cacheDirectory, CCharPointer optionsJson,
                                        CCharPointerPointer outJson) {
        try {
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String source = toJavaString(sourceGguf);
            String target = toJavaString(targetProfile);
            String cache = toJavaString(cacheDirectory);
            if (source == null || target == null || cache == null || outJson.isNull()) {
                lastError = "source_gguf, target_profile, cache_directory and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(SdxGgufModelPreparer.prepare(source,
                    toJavaString(tokenizerPath), target, cache, toJavaString(optionsJson))));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return statusFor(t);
        }
    }

    @CEntryPoint(name = "sdxLlmResolveModelBundle")
    public static int sdxLlmResolveModelBundle(IsolateThread runtime, CCharPointer sourceSdz,
                                               CCharPointer targetProfile,
                                               CCharPointer cacheDirectory,
                                               CCharPointerPointer outJson) {
        try {
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String source = toJavaString(sourceSdz);
            String target = toJavaString(targetProfile);
            String cache = toJavaString(cacheDirectory);
            if (source == null || target == null || cache == null || outJson.isNull()) {
                lastError = "source_sdz, target_profile, cache_directory and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(SdxGgufModelPreparer.resolve(source, target, cache)));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return statusFor(t);
        }
    }

    /* ── Model lifecycle ────────────────────────────────────────────────── */

    /**
     * Load a model (+ tokenizer) and build the generation pipeline.
     * Returns a null handle on failure; query sdxLlmGetLastError.
     */
    @CEntryPoint(name = "sdxLlmLoadModel")
    public static ObjectHandle sdxLlmLoadModel(IsolateThread runtime, CCharPointer modelPath,
                                               CCharPointer tokenizerPath, CCharPointer optionsJson) {
        try {
            // Side-loaded natives (SDK lib/ next to the library) must be resolvable
            // before ND4J / tokenizer classes initialize (ADR 0109). Idempotent.
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String model = toJavaString(modelPath);
            if (model == null) {
                lastError = "modelPath must not be null";
                return WordFactory.zero();
            }
            SdxLlmCore core = SdxLlmCore.load(model, toJavaString(tokenizerPath), toJavaString(optionsJson));
            return ObjectHandles.getGlobal().create(core);
        } catch (Throwable t) {
            lastError = describe(t);
            return WordFactory.zero();
        }
    }

    /** Open a resolver-produced accelerator bundle through the shared SDX runtime. */
    @CEntryPoint(name = "sdxLlmLoadCompiledModel")
    public static ObjectHandle sdxLlmLoadCompiledModel(IsolateThread runtime,
                                                       CCharPointer bundlePath,
                                                       CCharPointer tokenizerPath,
                                                       CCharPointer targetProfile,
                                                       CCharPointer optionsJson) {
        try {
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String bundle = toJavaString(bundlePath);
            String target = toJavaString(targetProfile);
            if (bundle == null || target == null) {
                lastError = "bundle_path and target_profile must not be null";
                return WordFactory.zero();
            }
            SdxCompiledLlmCore core = SdxCompiledLlmCore.load(bundle,
                    toJavaString(tokenizerPath), target, toJavaString(optionsJson));
            return ObjectHandles.getGlobal().create(core);
        } catch (Throwable t) {
            lastError = describe(t);
            return WordFactory.zero();
        }
    }

    @CEntryPoint(name = "sdxLlmUnloadModel")
    public static int sdxLlmUnloadModel(IsolateThread runtime, ObjectHandle model) {
        try {
            SdxLlmModel core = resolve(model);
            if (core == null) {
                return INVALID_ARGUMENT;
            }
            core.close();
            ObjectHandles.getGlobal().destroy(model);
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /* ── Generation ─────────────────────────────────────────────────────── */

    /**
     * Blocking generation. {@code outText} receives a malloc'd UTF-8 string owned by
     * the caller — release it with sdxLlmFree. Per-call options JSON may override
     * maxNewTokens/sampling; pass NULL for the load-time defaults.
     */
    @CEntryPoint(name = "sdxLlmGenerate")
    public static int sdxLlmGenerate(IsolateThread runtime, ObjectHandle model, CCharPointer prompt,
                                     CCharPointer optionsJson, CCharPointerPointer outText) {
        try {
            SdxLlmModel core = resolve(model);
            String promptText = toJavaString(prompt);
            if (core == null || promptText == null || outText.isNull()) {
                lastError = "model, prompt and out_text must not be null";
                return INVALID_ARGUMENT;
            }
            outText.write(toCString(core.generateText(promptText, toJavaString(optionsJson))));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** C callback receiving one complete UTF-8 text chunk. */
    public interface ChunkCallback extends CFunctionPointer {
        @InvokeCFunctionPointer
        void invoke(CCharPointer chunk);
    }

    /** C callback returning non-zero when cooperative cancellation is requested. */
    public interface CancelCallback extends CFunctionPointer {
        @InvokeCFunctionPointer
        int invoke();
    }

    @CEntryPoint(name = "sdxLlmGenerateStreaming")
    public static int sdxLlmGenerateStreaming(IsolateThread runtime, ObjectHandle model,
                                              CCharPointer prompt, CCharPointer optionsJson,
                                              ChunkCallback onChunk,
                                              CancelCallback shouldCancel,
                                              CCharPointerPointer outText) {
        try {
            SdxLlmModel core = resolve(model);
            String promptText = toJavaString(prompt);
            if (core == null || promptText == null || outText.isNull()) {
                lastError = "model, prompt and out_text must not be null";
                return INVALID_ARGUMENT;
            }
            Consumer<String> consumer = onChunk.isNull()
                    ? null : new NativeChunkConsumer(onChunk.rawValue());
            BooleanSupplier cancel = shouldCancel.isNull()
                    ? null : new NativeCancelSupplier(shouldCancel.rawValue());
            outText.write(toCString(core.generateStreaming(promptText,
                    toJavaString(optionsJson), consumer, cancel)));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /**
     * Graal native words cannot be captured by Java lambdas. Store only the raw
     * address in an ordinary Java object and reconstruct the function pointer at
     * the invocation boundary.
     */
    private static final class NativeChunkConsumer implements Consumer<String> {
        private final long callbackAddress;

        private NativeChunkConsumer(long callbackAddress) {
            this.callbackAddress = callbackAddress;
        }

        @Override
        public void accept(String chunk) {
            CCharPointer nativeChunk = toCString(chunk);
            try {
                ChunkCallback callback = WordFactory.pointer(callbackAddress);
                callback.invoke(nativeChunk);
            } finally {
                UnmanagedMemory.free(nativeChunk);
            }
        }
    }

    private static final class NativeCancelSupplier implements BooleanSupplier {
        private final long callbackAddress;

        private NativeCancelSupplier(long callbackAddress) {
            this.callbackAddress = callbackAddress;
        }

        @Override
        public boolean getAsBoolean() {
            CancelCallback callback = WordFactory.pointer(callbackAddress);
            return callback.invoke() != 0;
        }
    }

    /**
     * Generate a structured chat result. The imported tokenizer/model owns
     * rendering, reasoning-block normalization, and tool-call decoding.
     */
    @CEntryPoint(name = "sdxLlmGenerateChat")
    public static int sdxLlmGenerateChat(IsolateThread runtime, ObjectHandle model,
                                         CCharPointer requestJson,
                                         CCharPointer optionsJson,
                                         CCharPointerPointer outJson) {
        try {
            SdxLlmModel core = resolve(model);
            String request = toJavaString(requestJson);
            if (core == null || request == null || outJson.isNull()) {
                lastError = "model, request_json and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(core.generateChat(
                    request, toJavaString(optionsJson))));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** Decode streamed assistant text with the imported model's chat protocol. */
    @CEntryPoint(name = "sdxLlmParseChatResult")
    public static int sdxLlmParseChatResult(IsolateThread runtime, ObjectHandle model,
                                            CCharPointer requestJson,
                                            CCharPointer rawText,
                                            CCharPointerPointer outJson) {
        try {
            SdxLlmModel core = resolve(model);
            String request = toJavaString(requestJson);
            String raw = toJavaString(rawText);
            if (core == null || request == null || raw == null || outJson.isNull()) {
                lastError = "model, request_json, raw_text and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(core.parseChatResult(request, raw)));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** Render the tokenizer/model-owned template from a message array or full context object. */
    @CEntryPoint(name = "sdxLlmRenderChatPrompt")
    public static int sdxLlmRenderChatPrompt(IsolateThread runtime, ObjectHandle model,
                                             CCharPointer messagesOrContextJson,
                                             int addGenerationPrompt,
                                             CCharPointerPointer outPrompt) {
        try {
            SdxLlmModel core = resolve(model);
            String context = toJavaString(messagesOrContextJson);
            if (core == null || context == null || outPrompt.isNull()) {
                lastError = "model, chat context and out_prompt must not be null";
                return INVALID_ARGUMENT;
            }
            outPrompt.write(toCString(core.renderChatPrompt(context, addGenerationPrompt != 0)));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** Stats JSON for the most recent sdxLlmGenerate on this model handle. */
    @CEntryPoint(name = "sdxLlmLastResultJson")
    public static int sdxLlmLastResultJson(IsolateThread runtime, ObjectHandle model, CCharPointerPointer outJson) {
        try {
            SdxLlmModel core = resolve(model);
            if (core == null || outJson.isNull()) {
                lastError = "model and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(core.lastResultJson()));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** Model/tokenizer summary JSON (same document as `sdx-llm info`). */
    @CEntryPoint(name = "sdxLlmInfoJson")
    public static int sdxLlmInfoJson(IsolateThread runtime, ObjectHandle model, CCharPointerPointer outJson) {
        try {
            SdxLlmModel core = resolve(model);
            if (core == null || outJson.isNull()) {
                lastError = "model and out_json must not be null";
                return INVALID_ARGUMENT;
            }
            outJson.write(toCString(core.infoJson()));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /* ── Tokenization ───────────────────────────────────────────────────── */

    /**
     * Encode UTF-8 text. {@code outIds} receives a malloc'd int32 array (caller frees
     * with sdxLlmFree), {@code outCount} its length.
     */
    @CEntryPoint(name = "sdxLlmTokenize")
    public static int sdxLlmTokenize(IsolateThread runtime, ObjectHandle model, CCharPointer text,
                                     int addSpecialTokens, WordPointer outIds, CIntPointer outCount) {
        try {
            SdxLlmModel core = resolve(model);
            String input = toJavaString(text);
            if (core == null || input == null || outIds.isNull() || outCount.isNull()) {
                lastError = "model, text, out_ids and out_count must not be null";
                return INVALID_ARGUMENT;
            }
            int[] ids = core.tokenize(input, addSpecialTokens != 0);
            CIntPointer arr = UnmanagedMemory.malloc(WordFactory.unsigned(4L * Math.max(1, ids.length)));
            for (int i = 0; i < ids.length; i++) {
                arr.write(i, ids[i]);
            }
            outIds.write(arr);
            outCount.write(ids.length);
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /** Decode int32 token ids back to text (malloc'd UTF-8, caller frees with sdxLlmFree). */
    @CEntryPoint(name = "sdxLlmDetokenize")
    public static int sdxLlmDetokenize(IsolateThread runtime, ObjectHandle model, CIntPointer ids,
                                       int count, int skipSpecialTokens, CCharPointerPointer outText) {
        try {
            SdxLlmModel core = resolve(model);
            if (core == null || ids.isNull() || count < 0 || outText.isNull()) {
                lastError = "model, ids and out_text must not be null (count >= 0)";
                return INVALID_ARGUMENT;
            }
            int[] javaIds = new int[count];
            for (int i = 0; i < count; i++) {
                javaIds[i] = ids.read(i);
            }
            outText.write(toCString(core.detokenize(javaIds, skipSpecialTokens != 0)));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /* ── VLM + audio ────────────────────────────────────────────────────── */

    /**
     * VLM document extraction (SmolDocling). Stateless: loads, runs, releases.
     * {@code outText} receives a malloc'd UTF-8 string — free with sdxLlmFree.
     */
    @CEntryPoint(name = "sdxVlmExtract")
    public static int sdxVlmExtract(IsolateThread runtime, CCharPointer modelPath, CCharPointer tokenizerPath,
                                    CCharPointer inputPath, CCharPointer optionsJson, CCharPointerPointer outText) {
        try {
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String model = toJavaString(modelPath);
            String input = toJavaString(inputPath);
            if (model == null || input == null || outText.isNull()) {
                lastError = "model_path, input_path and out_text must not be null";
                return INVALID_ARGUMENT;
            }
            outText.write(toCString(SdxVlm.extract(model, toJavaString(tokenizerPath), input,
                    toJavaString(optionsJson))));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /**
     * Whisper speech-to-text. Stateless: loads, runs, releases.
     * {@code outText} receives a malloc'd UTF-8 string — free with sdxLlmFree.
     */
    @CEntryPoint(name = "sdxAudioTranscribe")
    public static int sdxAudioTranscribe(IsolateThread runtime, CCharPointer modelPath, CCharPointer audioPath,
                                         CCharPointer optionsJson, CCharPointerPointer outText) {
        try {
            SdxNativeLibs.bootstrapForEmbeddedNativeImage();
            String model = toJavaString(modelPath);
            String audio = toJavaString(audioPath);
            if (model == null || audio == null || outText.isNull()) {
                lastError = "model_path, audio_path and out_text must not be null";
                return INVALID_ARGUMENT;
            }
            outText.write(toCString(SdxAudio.transcribe(model, audio, toJavaString(optionsJson))));
            return OK;
        } catch (Throwable t) {
            lastError = describe(t);
            return EXECUTION_FAILED;
        }
    }

    /* ── Memory + errors ────────────────────────────────────────────────── */

    /** Free any pointer returned through an out-parameter of this ABI. */
    @CEntryPoint(name = "sdxLlmFree")
    public static void sdxLlmFree(IsolateThread runtime, CCharPointer pointer) {
        if (pointer.isNonNull()) {
            UnmanagedMemory.free(pointer);
        }
    }

    /**
     * Copy the last error message (UTF-8, NUL-terminated) into {@code buffer}.
     * Returns the full message length in bytes (excluding NUL), or 0 if none.
     */
    @CEntryPoint(name = "sdxLlmGetLastError")
    public static int sdxLlmGetLastError(IsolateThread runtime, CCharPointer buffer, int capacity) {
        String message = lastError;
        byte[] utf8 = message.getBytes(StandardCharsets.UTF_8);
        if (buffer.isNonNull() && capacity > 0) {
            int n = Math.min(utf8.length, capacity - 1);
            for (int i = 0; i < n; i++) {
                buffer.write(i, utf8[i]);
            }
            buffer.write(n, (byte) 0);
        }
        return utf8.length;
    }

    /* ── Helpers ────────────────────────────────────────────────────────── */

    private static SdxLlmModel resolve(ObjectHandle handle) {
        if (handle.equal(WordFactory.zero())) {
            return null;
        }
        Object o = ObjectHandles.getGlobal().get(handle);
        return o instanceof SdxLlmModel ? (SdxLlmModel) o : null;
    }

    private static String toJavaString(CCharPointer pointer) {
        return pointer.isNull() ? null : CTypeConversion.toJavaString(pointer);
    }

    /** Malloc'd, NUL-terminated UTF-8 copy the caller releases with sdxLlmFree. */
    private static CCharPointer toCString(String value) {
        byte[] utf8 = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        CCharPointer p = UnmanagedMemory.malloc(WordFactory.unsigned(utf8.length + 1L));
        for (int i = 0; i < utf8.length; i++) {
            p.write(i, utf8[i]);
        }
        p.write(utf8.length, (byte) 0);
        return p;
    }

    private static String describe(Throwable t) {
        StringBuilder description = new StringBuilder();
        Throwable current = t;
        for (int depth = 0; current != null && depth < 8; depth++) {
            if (depth > 0) description.append("; caused by ");
            description.append(current.getClass().getSimpleName());
            String message = current.getMessage();
            if (message != null && !message.isBlank()) description.append(": ").append(message);
            Throwable cause = current.getCause();
            if (cause == current) break;
            current = cause;
        }
        return description.toString();
    }

    private static int statusFor(Throwable t) {
        Throwable current = t;
        for (int depth = 0; current != null && depth < 8; depth++) {
            if (current instanceof IllegalArgumentException || current instanceof NullPointerException) {
                return INVALID_ARGUMENT;
            }
            if (current instanceof java.io.IOException) return IO_ERROR;
            Throwable cause = current.getCause();
            if (cause == current) break;
            current = cause;
        }
        return EXECUTION_FAILED;
    }
}
