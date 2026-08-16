#include <jni.h>

#include "sdx_llm_c.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

template <typename T>
T* from_handle(jlong value) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(static_cast<uint64_t>(value)));
}

jlong to_handle(const void* value) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(value));
}

void throw_exception(JNIEnv* env, const char* class_name, const char* message) {
    jclass exception_class = env->FindClass(class_name);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
        env->DeleteLocalRef(exception_class);
    }
}

class Utf8Bytes {
public:
    Utf8Bytes(JNIEnv* env, jbyteArray value) : env_(env), present_(value != nullptr) {
        if (!present_) {
            return;
        }
        const jsize length = env_->GetArrayLength(value);
        if (env_->ExceptionCheck()) {
            return;
        }
        bytes_.resize(static_cast<size_t>(length) + 1U);
        if (length > 0) {
            env_->GetByteArrayRegion(value, 0, length, reinterpret_cast<jbyte*>(bytes_.data()));
        }
        bytes_[static_cast<size_t>(length)] = '\0';
    }

    const char* get() const {
        return present_ && !bytes_.empty() ? bytes_.data() : nullptr;
    }

    bool valid() const {
        return !env_->ExceptionCheck();
    }

private:
    JNIEnv* env_;
    bool present_;
    std::vector<char> bytes_;
};

bool prepare_output(JNIEnv* env, jlongArray output) {
    if (output == nullptr || env->GetArrayLength(output) < 1) {
        if (!env->ExceptionCheck()) {
            throw_exception(env, "java/lang/IllegalArgumentException",
                            "SDX output handle array must contain one element");
        }
        return false;
    }
    const jlong zero = 0;
    env->SetLongArrayRegion(output, 0, 1, &zero);
    return !env->ExceptionCheck();
}

bool publish_output(JNIEnv* env, jlongArray output, const void* value) {
    const jlong handle = to_handle(value);
    env->SetLongArrayRegion(output, 0, 1, &handle);
    return !env->ExceptionCheck();
}

struct CallbackContext {
    JNIEnv* env;
    jobject chunk_callback;
    jmethodID chunk_method;
    jobject cancel_callback;
    jmethodID cancel_method;
};

thread_local CallbackContext* active_callbacks = nullptr;

class CallbackScope {
public:
    explicit CallbackScope(CallbackContext* current)
        : previous_(active_callbacks) {
        active_callbacks = current;
    }

    ~CallbackScope() {
        active_callbacks = previous_;
    }

private:
    CallbackContext* previous_;
};

void deliver_chunk(const char* utf8_chunk) {
    CallbackContext* context = active_callbacks;
    if (context == nullptr || context->chunk_callback == nullptr ||
        utf8_chunk == nullptr || context->env->ExceptionCheck()) {
        return;
    }

    const size_t length = std::strlen(utf8_chunk);
    if (length > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        throw_exception(context->env, "java/lang/OutOfMemoryError",
                        "SDX streaming chunk exceeds the JNI array limit");
        return;
    }

    jbyteArray bytes = context->env->NewByteArray(static_cast<jsize>(length));
    if (bytes == nullptr) {
        return;
    }
    if (length > 0) {
        context->env->SetByteArrayRegion(
                bytes, 0, static_cast<jsize>(length),
                reinterpret_cast<const jbyte*>(utf8_chunk));
    }
    if (!context->env->ExceptionCheck()) {
        context->env->CallVoidMethod(context->chunk_callback, context->chunk_method, bytes);
    }
    context->env->DeleteLocalRef(bytes);
}

int32_t cancellation_requested() {
    CallbackContext* context = active_callbacks;
    if (context == nullptr) {
        return 0;
    }
    if (context->env->ExceptionCheck()) {
        return 1;
    }
    if (context->cancel_callback == nullptr) {
        return 0;
    }
    const jint result =
            context->env->CallIntMethod(context->cancel_callback, context->cancel_method);
    return context->env->ExceptionCheck() ? 1 : static_cast<int32_t>(result);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeCreateRuntime(
        JNIEnv*, jclass) {
    return to_handle(sdxLlmCreateRuntime());
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeDestroyRuntime(
        JNIEnv*, jclass, jlong runtime) {
    return sdxLlmDestroyRuntime(from_handle<sdx_llm_runtime_t>(runtime));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeAbiVersion(
        JNIEnv*, jclass, jlong runtime) {
    return sdxLlmAbiVersion(from_handle<sdx_llm_runtime_t>(runtime));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativePrepareGguf(
        JNIEnv* env, jclass, jlong runtime, jbyteArray source_gguf,
        jbyteArray tokenizer_path, jbyteArray target_profile,
        jbyteArray cache_directory, jbyteArray options_json, jlongArray out_json) {
    if (!prepare_output(env, out_json)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }
    Utf8Bytes source(env, source_gguf);
    Utf8Bytes tokenizer(env, tokenizer_path);
    Utf8Bytes target(env, target_profile);
    Utf8Bytes cache(env, cache_directory);
    Utf8Bytes options(env, options_json);
    if (!source.valid() || !tokenizer.valid() || !target.valid() ||
        !cache.valid() || !options.valid()) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmPrepareGguf(
            from_handle<sdx_llm_runtime_t>(runtime), source.get(), tokenizer.get(),
            target.get(), cache.get(), options.get(), &output);
    if (!env->ExceptionCheck()) {
        publish_output(env, out_json, output);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeResolveModelBundle(
        JNIEnv* env, jclass, jlong runtime, jbyteArray source_sdz,
        jbyteArray target_profile, jbyteArray cache_directory, jlongArray out_json) {
    if (!prepare_output(env, out_json)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }
    Utf8Bytes source(env, source_sdz);
    Utf8Bytes target(env, target_profile);
    Utf8Bytes cache(env, cache_directory);
    if (!source.valid() || !target.valid() || !cache.valid()) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmResolveModelBundle(
            from_handle<sdx_llm_runtime_t>(runtime), source.get(), target.get(),
            cache.get(), &output);
    if (!env->ExceptionCheck()) {
        publish_output(env, out_json, output);
    }
    return status;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeLoadCompiledModel(
        JNIEnv* env, jclass, jlong runtime, jbyteArray bundle_path,
        jbyteArray tokenizer_path, jbyteArray target_profile, jbyteArray options_json) {
    Utf8Bytes bundle(env, bundle_path);
    Utf8Bytes tokenizer(env, tokenizer_path);
    Utf8Bytes target(env, target_profile);
    Utf8Bytes options(env, options_json);
    if (!bundle.valid() || !tokenizer.valid() || !target.valid() || !options.valid()) {
        return 0;
    }
    return to_handle(sdxLlmLoadCompiledModel(
            from_handle<sdx_llm_runtime_t>(runtime), bundle.get(), tokenizer.get(),
            target.get(), options.get()));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeUnloadModel(
        JNIEnv*, jclass, jlong runtime, jlong model) {
    return sdxLlmUnloadModel(
            from_handle<sdx_llm_runtime_t>(runtime),
            from_handle<sdx_llm_model_t>(model));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeRenderChatPrompt(
        JNIEnv* env, jclass, jlong runtime, jlong model, jbyteArray messages_json,
        jint add_generation_prompt, jlongArray out_prompt) {
    if (!prepare_output(env, out_prompt)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }
    Utf8Bytes messages(env, messages_json);
    if (!messages.valid()) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmRenderChatPrompt(
            from_handle<sdx_llm_runtime_t>(runtime),
            from_handle<sdx_llm_model_t>(model), messages.get(),
            static_cast<int32_t>(add_generation_prompt), &output);
    if (!env->ExceptionCheck()) {
        publish_output(env, out_prompt, output);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeParseChatResult(
        JNIEnv* env, jclass, jlong runtime, jlong model, jbyteArray request_json,
        jbyteArray raw_text, jlongArray out_json) {
    if (!prepare_output(env, out_json)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }
    Utf8Bytes request(env, request_json);
    Utf8Bytes raw(env, raw_text);
    if (!request.valid() || !raw.valid()) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmParseChatResult(
            from_handle<sdx_llm_runtime_t>(runtime),
            from_handle<sdx_llm_model_t>(model), request.get(), raw.get(), &output);
    if (!env->ExceptionCheck()) {
        publish_output(env, out_json, output);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeLastResultJson(
        JNIEnv* env, jclass, jlong runtime, jlong model, jlongArray out_json) {
    if (!prepare_output(env, out_json)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmLastResultJson(
            from_handle<sdx_llm_runtime_t>(runtime),
            from_handle<sdx_llm_model_t>(model), &output);
    if (!env->ExceptionCheck()) {
        publish_output(env, out_json, output);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeGenerateStreaming(
        JNIEnv* env, jclass, jlong runtime, jlong model, jbyteArray prompt,
        jbyteArray options_json, jobject on_chunk, jobject should_cancel,
        jlongArray out_text) {
    if (!prepare_output(env, out_text)) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }
    Utf8Bytes prompt_value(env, prompt);
    Utf8Bytes options(env, options_json);
    if (!prompt_value.valid() || !options.valid()) {
        return SDX_LLM_STATUS_INVALID_ARGUMENT;
    }

    jmethodID chunk_method = nullptr;
    if (on_chunk != nullptr) {
        jclass chunk_class = env->GetObjectClass(on_chunk);
        if (chunk_class == nullptr) {
            return SDX_LLM_STATUS_INVALID_ARGUMENT;
        }
        chunk_method = env->GetMethodID(chunk_class, "onChunk", "([B)V");
        env->DeleteLocalRef(chunk_class);
        if (chunk_method == nullptr) {
            return SDX_LLM_STATUS_INVALID_ARGUMENT;
        }
    }

    jmethodID cancel_method = nullptr;
    if (should_cancel != nullptr) {
        jclass cancel_class = env->GetObjectClass(should_cancel);
        if (cancel_class == nullptr) {
            return SDX_LLM_STATUS_INVALID_ARGUMENT;
        }
        cancel_method = env->GetMethodID(cancel_class, "shouldCancel", "()I");
        env->DeleteLocalRef(cancel_class);
        if (cancel_method == nullptr) {
            return SDX_LLM_STATUS_INVALID_ARGUMENT;
        }
    }

    CallbackContext callbacks{
            env, on_chunk, chunk_method, should_cancel, cancel_method};
    CallbackScope callback_scope(&callbacks);
    char* output = nullptr;
    const sdx_llm_status_t status = sdxLlmGenerateStreaming(
            from_handle<sdx_llm_runtime_t>(runtime),
            from_handle<sdx_llm_model_t>(model), prompt_value.get(), options.get(),
            on_chunk == nullptr ? nullptr : deliver_chunk,
            should_cancel == nullptr ? nullptr : cancellation_requested, &output);

    if (env->ExceptionCheck()) {
        if (output != nullptr) {
            sdxLlmFree(from_handle<sdx_llm_runtime_t>(runtime), output);
        }
        return status;
    }
    publish_output(env, out_text, output);
    return status;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeReadUtf8(
        JNIEnv* env, jclass, jlong pointer) {
    const char* value = from_handle<char>(pointer);
    if (value == nullptr) {
        return nullptr;
    }
    const size_t length = std::strlen(value);
    if (length > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        throw_exception(env, "java/lang/OutOfMemoryError",
                        "SDX UTF-8 result exceeds the JNI array limit");
        return nullptr;
    }
    jbyteArray result = env->NewByteArray(static_cast<jsize>(length));
    if (result != nullptr && length > 0) {
        env->SetByteArrayRegion(
                result, 0, static_cast<jsize>(length),
                reinterpret_cast<const jbyte*>(value));
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeFree(
        JNIEnv*, jclass, jlong runtime, jlong pointer) {
    sdxLlmFree(from_handle<sdx_llm_runtime_t>(runtime), from_handle<void>(pointer));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_kompile_chat_local_android_model_SdxAndroidLlmNative_nativeGetLastError(
        JNIEnv* env, jclass, jlong runtime, jbyteArray buffer, jint capacity) {
    if (buffer == nullptr || capacity < 0 || capacity > env->GetArrayLength(buffer)) {
        if (!env->ExceptionCheck()) {
            throw_exception(env, "java/lang/IllegalArgumentException",
                            "SDX last-error capacity exceeds its Java byte array");
        }
        return -1;
    }

    std::vector<char> native_buffer(static_cast<size_t>(capacity));
    char* destination = capacity == 0 ? nullptr : native_buffer.data();
    const int result = sdxLlmGetLastError(
            from_handle<sdx_llm_runtime_t>(runtime), destination,
            static_cast<int32_t>(capacity));
    if (capacity > 0 && !env->ExceptionCheck()) {
        env->SetByteArrayRegion(
                buffer, 0, capacity,
                reinterpret_cast<const jbyte*>(native_buffer.data()));
    }
    return result;
}
