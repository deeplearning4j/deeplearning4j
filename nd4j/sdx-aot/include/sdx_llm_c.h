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

/*
 * sdx_llm_c.h — C ABI of libsdx_llm, the AOT-compiled (GraalVM native-image)
 * LLM surface of the SDX Runtime SDK. Companion to dsp_runtime_c.h: that header
 * executes SameDiff graphs; this one adds the Java-stack capabilities on top —
 * GGUF import, HuggingFace tokenization, and the full autoregressive generation
 * pipeline (KV cache management, sampling, chat templates) — with no JVM.
 *
 * Implemented by @CEntryPoint methods in
 * nd4j/sdx-aot/src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxLlmCApi.java.
 * Keep the two in sync. See ADR 0109.
 *
 * Threading (v1): a runtime handle is bound to the OS thread that created it.
 * Create, use and destroy each runtime from one thread. Use one runtime per
 * thread for concurrent generation (models are not shared across runtimes).
 */

#ifndef SDX_LLM_C_H
#define SDX_LLM_C_H

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(__GNUC__)
#define SDX_LLM_API __attribute__((dllexport))
#else
#define SDX_LLM_API __declspec(dllexport)
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define SDX_LLM_API __attribute__((visibility("default")))
#else
#define SDX_LLM_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  SDX_LLM_ABI_VERSION = 1
};

/* Opaque handles. sdx_llm_runtime_t is a GraalVM isolate thread; sdx_llm_model_t
 * is an object handle valid only within the runtime that created it. */
typedef struct sdx_llm_runtime sdx_llm_runtime_t;
typedef struct sdx_llm_model sdx_llm_model_t;

/* Status codes — aligned with the sdx_status_t values of dsp_runtime_c.h. */
typedef enum {
  SDX_LLM_STATUS_OK = 0,
  SDX_LLM_STATUS_INVALID_ARGUMENT = 1,
  SDX_LLM_STATUS_MODEL_LOAD_FAILED = 3,
  SDX_LLM_STATUS_EXECUTION_FAILED = 4,
  SDX_LLM_STATUS_IO_ERROR = 6
} sdx_llm_status_t;

/* ── Runtime lifecycle ────────────────────────────────────────────────── */

/* Create an isolated runtime (heap + threads). Returns NULL on failure. */
SDX_LLM_API sdx_llm_runtime_t* sdxLlmCreateRuntime(void);

/* Tear the runtime down. All model handles created in it become invalid.
 * Returns 0 on success. */
SDX_LLM_API int sdxLlmDestroyRuntime(sdx_llm_runtime_t* runtime);

/* ABI version compiled into the library (compare with SDX_LLM_ABI_VERSION). */
SDX_LLM_API int sdxLlmAbiVersion(sdx_llm_runtime_t* runtime);

/* ── Model lifecycle ──────────────────────────────────────────────────── */

/*
 * Load a model and build the generation pipeline.
 *   model_path     .gguf/.ggml (imported in-process) or .sdz/.sdnb/.fb
 *   tokenizer_path optional tokenizer.json file or directory; NULL tries the
 *                  model's directory
 *   options_json   optional, e.g. {"maxNewTokens":128,"graphOptimizer":true,
 *                  "sampling":{"preset":"greedy"}} or explicit sampling fields
 *                  {"temperature":0.8,"topK":40,"topP":0.9,
 *                   "repetitionPenalty":1.1,"seed":42}
 * Returns NULL on failure — query sdxLlmGetLastError.
 */
SDX_LLM_API sdx_llm_model_t* sdxLlmLoadModel(
    sdx_llm_runtime_t* runtime,
    const char* model_path,
    const char* tokenizer_path,
    const char* options_json);

SDX_LLM_API sdx_llm_status_t sdxLlmUnloadModel(sdx_llm_runtime_t* runtime, sdx_llm_model_t* model);

/* ── Generation ───────────────────────────────────────────────────────── */

/*
 * Blocking text generation. *out_text receives a malloc'd UTF-8 string owned
 * by the caller — release with sdxLlmFree. options_json may override
 * {"maxNewTokens":N,"sampling":{...}} per call (NULL = load-time defaults).
 * The pipeline's compiled plan and KV state are reused across calls.
 */
SDX_LLM_API sdx_llm_status_t sdxLlmGenerate(
    sdx_llm_runtime_t* runtime,
    sdx_llm_model_t* model,
    const char* prompt,
    const char* options_json,
    char** out_text);

/* Stats JSON of the most recent sdxLlmGenerate on this model: token counts,
 * timings, tok/s, finish reason. Caller frees *out_json with sdxLlmFree. */
SDX_LLM_API sdx_llm_status_t sdxLlmLastResultJson(
    sdx_llm_runtime_t* runtime,
    sdx_llm_model_t* model,
    char** out_json);

/* Model/tokenizer summary JSON (inputs, outputs, vocab, chat template flag). */
SDX_LLM_API sdx_llm_status_t sdxLlmInfoJson(
    sdx_llm_runtime_t* runtime,
    sdx_llm_model_t* model,
    char** out_json);

/* ── Tokenization ─────────────────────────────────────────────────────── */

/* Encode UTF-8 text. *out_ids receives a malloc'd int32 array (caller frees
 * with sdxLlmFree), *out_count its length. */
SDX_LLM_API sdx_llm_status_t sdxLlmTokenize(
    sdx_llm_runtime_t* runtime,
    sdx_llm_model_t* model,
    const char* text,
    int32_t add_special_tokens,
    int32_t** out_ids,
    int32_t* out_count);

/* Decode int32 token ids to text. Caller frees *out_text with sdxLlmFree. */
SDX_LLM_API sdx_llm_status_t sdxLlmDetokenize(
    sdx_llm_runtime_t* runtime,
    sdx_llm_model_t* model,
    const int32_t* ids,
    int32_t count,
    int32_t skip_special_tokens,
    char** out_text);

/* ── VLM + audio ──────────────────────────────────────────────────────── */

/* VLM document extraction (SmolDocling): image/PDF -> doctags/markdown/text.
 * options_json: {"maxNewTokens":N,"prompt":"...","format":"doctags|markdown|text"}.
 * Stateless — loads and releases the model per call. Caller frees *out_text
 * with sdxLlmFree. */
SDX_LLM_API sdx_llm_status_t sdxVlmExtract(
    sdx_llm_runtime_t* runtime,
    const char* model_path,
    const char* tokenizer_path,
    const char* input_path,
    const char* options_json,
    char** out_text);

/* Whisper speech-to-text. options_json: {"language":"en","maxNewTokens":N}.
 * Stateless — loads and releases the model per call. Caller frees *out_text
 * with sdxLlmFree. */
SDX_LLM_API sdx_llm_status_t sdxAudioTranscribe(
    sdx_llm_runtime_t* runtime,
    const char* model_path,
    const char* audio_path,
    const char* options_json,
    char** out_text);

/* ── Memory + errors ──────────────────────────────────────────────────── */

/* Free any pointer returned through an out-parameter of this ABI. */
SDX_LLM_API void sdxLlmFree(sdx_llm_runtime_t* runtime, void* pointer);

/* Copy the last error message (UTF-8, NUL-terminated) into buffer. Returns the
 * full message length in bytes (excluding NUL), 0 if no error is recorded. */
SDX_LLM_API int sdxLlmGetLastError(sdx_llm_runtime_t* runtime, char* buffer, int32_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* SDX_LLM_C_H */
