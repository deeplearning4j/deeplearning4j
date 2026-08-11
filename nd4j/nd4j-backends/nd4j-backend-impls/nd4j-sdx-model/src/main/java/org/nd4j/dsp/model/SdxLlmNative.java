/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;

/** JavaCPP transport for the provider-independent {@code libsdx_llm} C ABI. */
@Platform(include = "sdx_llm_c.h", link = "sdx_llm", preload = "sdx_llm", library = "jnisdx_llm")
public final class SdxLlmNative {
    public static final int SDX_LLM_ABI_VERSION = 2;
    static { Loader.load(); }
    private SdxLlmNative() { }

    public static class ChunkCallback extends FunctionPointer {
        protected ChunkCallback() { allocate(); }
        private native void allocate();
        public native void call(@Cast("const char*") BytePointer chunk);
    }
    public static class CancelCallback extends FunctionPointer {
        protected CancelCallback() { allocate(); }
        private native void allocate();
        public native int call();
    }
    public static native @Cast("sdx_llm_runtime_t*") Pointer sdxLlmCreateRuntime();
    public static native int sdxLlmDestroyRuntime(@Cast("sdx_llm_runtime_t*") Pointer runtime);
    public static native int sdxLlmAbiVersion(@Cast("sdx_llm_runtime_t*") Pointer runtime);
    public static native int sdxLlmPrepareGguf(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("const char*") BytePointer sourceGguf, @Cast("const char*") BytePointer tokenizerPath,
        @Cast("const char*") BytePointer targetProfile, @Cast("const char*") BytePointer cacheDirectory,
        @Cast("const char*") BytePointer optionsJson, @Cast("char**") PointerPointer outJson);
    public static native int sdxLlmResolveModelBundle(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("const char*") BytePointer sourceSdz, @Cast("const char*") BytePointer targetProfile,
        @Cast("const char*") BytePointer cacheDirectory, @Cast("char**") PointerPointer outJson);
    public static native @Cast("sdx_llm_model_t*") Pointer sdxLlmLoadCompiledModel(
        @Cast("sdx_llm_runtime_t*") Pointer runtime, @Cast("const char*") BytePointer bundlePath,
        @Cast("const char*") BytePointer tokenizerPath, @Cast("const char*") BytePointer targetProfile,
        @Cast("const char*") BytePointer optionsJson);
    public static native int sdxLlmUnloadModel(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("sdx_llm_model_t*") Pointer model);
    public static native int sdxLlmRenderChatPrompt(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("sdx_llm_model_t*") Pointer model, @Cast("const char*") BytePointer messagesJson,
        int addGenerationPrompt, @Cast("char**") PointerPointer outPrompt);
    public static native int sdxLlmParseChatResult(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("sdx_llm_model_t*") Pointer model, @Cast("const char*") BytePointer requestJson,
        @Cast("const char*") BytePointer rawText, @Cast("char**") PointerPointer outJson);
    public static native int sdxLlmGenerateStreaming(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("sdx_llm_model_t*") Pointer model, @Cast("const char*") BytePointer prompt,
        @Cast("const char*") BytePointer optionsJson, ChunkCallback onChunk,
        CancelCallback shouldCancel, @Cast("char**") PointerPointer outText);
    public static native void sdxLlmFree(@Cast("sdx_llm_runtime_t*") Pointer runtime, Pointer pointer);
    public static native int sdxLlmGetLastError(@Cast("sdx_llm_runtime_t*") Pointer runtime,
        @Cast("char*") BytePointer buffer, int capacity);
}
