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

#ifndef LIBND4J_DSP_RUNTIME_C_H
#define LIBND4J_DSP_RUNTIME_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(__GNUC__)
#define SDX_API __attribute__((dllexport))
#else
#define SDX_API __declspec(dllexport)
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define SDX_API __attribute__((visibility("default")))
#else
#define SDX_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sdx_runtime sdx_runtime_t;
typedef struct sdx_model sdx_model_t;
typedef struct sdx_context sdx_context_t;
typedef struct sdx_generation_session sdx_generation_session_t;

enum {
  SDX_RUNTIME_ABI_VERSION = 1
};

typedef enum {
  SDX_STATUS_OK = 0,
  SDX_STATUS_INVALID_ARGUMENT = 1,
  SDX_STATUS_INCOMPATIBLE_ABI = 2,
  SDX_STATUS_MODEL_LOAD_FAILED = 3,
  SDX_STATUS_EXECUTION_FAILED = 4,
  SDX_STATUS_BACKEND_UNAVAILABLE = 5,
  SDX_STATUS_IO_ERROR = 6,
  SDX_STATUS_UNSUPPORTED = 7
} sdx_status_t;

typedef enum {
  SDX_GENERATION_FINISH_NONE = 0,
  SDX_GENERATION_FINISH_MAX_TOKENS = 1,
  SDX_GENERATION_FINISH_EOS = 2,
  SDX_GENERATION_FINISH_CANCELLED = 3,
  SDX_GENERATION_FINISH_CONTEXT_LIMIT = 4
} sdx_generation_finish_reason_t;

typedef enum {
  SDX_BACKEND_AUTO = 0,
  SDX_BACKEND_SLOT_BY_SLOT = 1,
  SDX_BACKEND_CUDA_GRAPHS = 2,
  SDX_BACKEND_NVRTC = 3,
  SDX_BACKEND_PTX = 4,
  SDX_BACKEND_TRITON = 5,
  SDX_BACKEND_MLX = 6,
  SDX_BACKEND_ARM_HYBRID = 7,
  SDX_BACKEND_NNAPI = 8,
  SDX_BACKEND_HIP_GRAPHS = 9,
  SDX_BACKEND_LEVEL_ZERO = 10,
  SDX_BACKEND_VULKAN = 11,
  SDX_BACKEND_METAL = 12,
  SDX_BACKEND_TPU = 13,
  SDX_BACKEND_HEXAGON = 14,
  SDX_BACKEND_OPENVINO = 15
} sdx_backend_t;

typedef enum {
  SDX_DEVICE_HOST = 0,
  SDX_DEVICE_CUDA = 1,
  SDX_DEVICE_AMD = 2
} sdx_device_type_t;

typedef enum {
  SDX_GPU_TARGET_AUTO = 0,
  SDX_GPU_TARGET_CUDA = 1,
  SDX_GPU_TARGET_AMD = 2,
  SDX_GPU_TARGET_VULKAN = 3,
  SDX_GPU_TARGET_METAL = 4
} sdx_gpu_target_t;

typedef struct {
  uint32_t struct_size;
} sdx_runtime_options_t;

typedef struct {
  uint32_t struct_size;
  int32_t backend;
  int32_t strict_backend;
  /* 0 forbids all backend code generation. Hardware backends must then carry
   * their validated bundle assets (for example compiledArtifacts.vulkanSpirv
   * or compiledArtifacts.hexagonKernels); an artifact miss is a hard failure. */
  int32_t allow_runtime_jit;
  int32_t gpu_target;
  /* App-owned writable directory for persistent device/compiler artifacts.
   * sdxLoadBundle copies this path before returning; callers retain ownership. */
  const char* device_compilation_cache_directory;
} sdx_model_options_t;

typedef struct {
  uint32_t struct_size;
  /** When non-zero, constants and variables loaded from the model bundle are
   * bound by the runtime and omitted from the public context input list.
   * This is the inference/mobile mode: sdxGetNumInputs() then reports only
   * placeholders or other values not present in the bundle. */
  int32_t bind_model_parameters;
} sdx_context_options_t;

typedef struct {
  uint32_t struct_size;
  int32_t backend;
  int32_t strict_signature;
  int32_t gpu_target;
} sdx_run_options_t;

/**
 * Options for the reusable token-generation session. The session reads its
 * explicit I/O names, KV layout, token IDs, and fixed mobile shape envelope
 * from the bundle's text-generation metadata. Runtime discovery heuristics are
 * deliberately not part of this API.
 */
typedef struct {
  uint32_t struct_size;
  /** One physical plan/buffer capacity. Zero uses limits.maxPrefillLength. */
  int32_t fixed_context_capacity;
} sdx_generation_session_options_t;

/**
 * Scalar generation policy consumed by the shared TokenSampleConfig primitive.
 * Zero-initialize, set struct_size, then override fields as required. A
 * temperature <= 0 selects greedy decoding. top_p <= 0 or >= 1 disables
 * nucleus filtering; repetition_penalty <= 0 is normalized to 1.
 */
typedef struct {
  uint32_t struct_size;
  int32_t max_new_tokens;
  int32_t min_new_tokens;
  double temperature;
  int32_t top_k;
  double top_p;
  double min_p;
  double repetition_penalty;
  double frequency_penalty;
  double presence_penalty;
  double typical_p;
  double xtc_probability;
  double xtc_threshold;
  int64_t seed;
} sdx_generation_options_t;

/** Called synchronously after a token has been committed to session state. */
typedef void (*sdx_token_callback_t)(int64_t token_id, void* user_data);

/** Return non-zero to request cancellation at the next coherent token boundary. */
typedef int32_t (*sdx_cancel_callback_t)(void* user_data);

typedef struct {
  uint32_t struct_size;
  sdx_token_callback_t on_token;
  sdx_cancel_callback_t should_cancel;
  void* user_data;
} sdx_generation_callbacks_t;

typedef struct {
  uint32_t struct_size;
  int32_t finish_reason;
  int32_t prompt_token_count;
  int32_t generated_token_count;
  int32_t total_generated_token_count;
  int32_t context_position;
  uint64_t elapsed_time_ns;
  uint64_t prefill_time_ns;
  uint64_t decode_time_ns;
  double decode_tokens_per_second;
  /** Non-zero when backend evidence was read from the executed decode context. */
  int32_t backend_report_available;
  int32_t requested_backend;
  int32_t applied_backend;
  int32_t backend_status_code;
  /** 1 = host/capture fallback observed, 0 = requested path, -1 = unknown. */
  int32_t used_fallback;
  int32_t requested_gpu_target;
  int32_t applied_gpu_target;
  int32_t plan_phase;
  int32_t execution_count;
} sdx_generation_report_t;

typedef struct {
  void* data;
  const int64_t* shape;
  int32_t rank;
  int32_t dtype;
  size_t bytes;
  int32_t device_type;
  /* For CUDA/AMD tensors, device_id must be >= 0. Mixed device_ids are
   * supported — the runtime elects the majority device and migrates
   * off-device inputs automatically. Constant replicas are cached. */
  int32_t device_id;
} sdx_tensor_view_t;

typedef struct {
  uint32_t struct_size;
  int32_t requested_backend;
  /** Backend read back from the plan after execution (reflects clamping and
   *  plan-side mode changes; sdx_backend_t values). */
  int32_t applied_backend;
  int32_t status_code;
  /** 1 = fallback observed (a segment failed graph capture, or the plan is
   *  REPLAY_BLOCKED); 0 = requested/optimal path in force; -1 = unknown
   *  (no execution yet). */
  int32_t used_fallback;
  uint64_t execution_time_ns;
  int32_t requested_gpu_target;
  int32_t applied_gpu_target;
  /** Plan phase after execution: 0=SLOT_BY_SLOT (warmup), 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED */
  int32_t plan_phase;
  /** Number of executions completed on this context */
  int32_t execution_count;
} sdx_execution_report_t;

SDX_API int sdxGetRuntimeAbiVersion(void);

SDX_API sdx_status_t sdxCreateRuntime(const sdx_runtime_options_t* options, sdx_runtime_t** out_runtime);
SDX_API void sdxDestroyRuntime(sdx_runtime_t* runtime);

/**
 * Configure, record, clear, and persist diagnostics owned by the selected SDX
 * backend library. These functions intentionally live on the backend-neutral
 * SDX transport: an Android accelerator process may also contain an embedded
 * CPU backend, but only the selected provider owns the active compiled plan
 * and its diagnostic ring.
 *
 * category_mask/category use the DspDiagCategory bit assignments and level is
 * 0=summary, 1=detailed, or 2=full. An empty json_path disables file output.
 */
SDX_API sdx_status_t sdxConfigureDiagnostics(
    sdx_runtime_t* runtime,
    uint32_t category_mask,
    int32_t level,
    const char* json_path);
SDX_API sdx_status_t sdxRecordDiagnosticEvent(
    sdx_runtime_t* runtime,
    uint32_t category,
    const char* message);
SDX_API void sdxClearDiagnostics(void);
SDX_API void sdxFlushDiagnostics(void);

SDX_API sdx_status_t sdxLoadBundle(
    sdx_runtime_t* runtime,
    const char* bundle_path,
    const sdx_model_options_t* options,
    sdx_model_t** out_model);
SDX_API void sdxUnloadModel(sdx_model_t* model);

/**
 * Resolved offline text-generation asset paths from the loaded bundle. The
 * returned pointers are owned by the model and stay valid until
 * sdxUnloadModel(). NULL means the manifest did not declare that asset.
 */
SDX_API const char* sdxGetTokenizerPath(const sdx_model_t* model);
SDX_API const char* sdxGetTextGenerationConfigPath(const sdx_model_t* model);

/**
 * Create a reusable, metadata-driven generation session over a loaded model.
 * The model must outlive the session. Supported mobile profiles are
 * causal-lm-in-graph-kv-v1 and causal-lm-in-graph-state-v2: batch 1, fixed
 * prefill/decode envelopes, BSHD KV, plan-owned in-graph KV writes, and
 * metadata-declared recurrent state feedback for v2. Unsupported profiles fail
 * explicitly.
 */
SDX_API sdx_status_t sdxCreateGenerationSession(
    sdx_model_t* model,
    const sdx_generation_session_options_t* options,
    sdx_generation_session_t** out_session);
/** Effective fixed physical capacity after model-envelope clamping. */
SDX_API int32_t sdxGetGenerationContextCapacity(
    const sdx_generation_session_t* session);
SDX_API void sdxDestroyGenerationSession(sdx_generation_session_t* session);

/**
 * Clear logical prompt/KV/recurrent state while retaining the session's sole
 * precompiled context, fixed buffers, and plan.
 */
SDX_API sdx_status_t sdxResetGenerationSession(
    sdx_generation_session_t* session);

/**
 * Cooperatively cancel an in-flight generation from another thread. The native
 * decode loop observes cancellation between complete token commits, leaving the
 * session safe to continue.
 */
SDX_API void sdxCancelGeneration(sdx_generation_session_t* session);

/**
 * Reset logical state, ingest a rolling prompt window through the precompiled
 * fixed plan, and generate to EOS/cancellation/context capacity. Tokens are returned in
 * out_token_ids and optionally streamed through callbacks on the calling
 * thread. out_capacity must be at least max_new_tokens when out_token_ids is
 * non-NULL; out_count is always required.
 */
SDX_API sdx_status_t sdxGenerationGenerate(
    sdx_generation_session_t* session,
    const int64_t* prompt_token_ids,
    int32_t num_prompt_tokens,
    const sdx_generation_options_t* options,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* out_token_ids,
    int32_t out_capacity,
    int32_t* out_count,
    sdx_generation_report_t* out_report);

/**
 * Continue from the last coherent token/KV boundary without re-prefill. The
 * same output, callback, cancellation, and reporting contract as
 * sdxGenerationGenerate applies.
 */
SDX_API sdx_status_t sdxGenerationContinue(
    sdx_generation_session_t* session,
    const sdx_generation_options_t* options,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* out_token_ids,
    int32_t out_capacity,
    int32_t* out_count,
    sdx_generation_report_t* out_report);

/**
 * Create a context using the legacy all-external-input contract. The model
 * must outlive every context created from it.
 */
SDX_API sdx_status_t sdxCreateContext(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    sdx_context_t** out_context);

/**
 * Create a context with explicit input-binding behavior. Setting
 * bind_model_parameters=1 binds constants and variables owned by the loaded
 * bundle internally, leaving only runtime placeholders as public inputs. This
 * is the recommended inference API for mobile and offline applications.
 *
 * This is additive to ABI v1: sdxCreateContext() retains its original behavior.
 * The model must outlive every context created from it.
 */
SDX_API sdx_status_t sdxCreateContextWithOptions(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    const sdx_context_options_t* options,
    sdx_context_t** out_context);
SDX_API void sdxDestroyContext(sdx_context_t* context);

SDX_API sdx_status_t sdxRun(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_tensor_view_t* outputs,
    int32_t num_outputs,
    const sdx_run_options_t* options);

/**
 * Execute without caller-allocated output buffers. This is the recommended
 * path for dynamic-shape inference such as autoregressive logits. After a
 * successful call, use sdxGetOutputTensor() to obtain each output.
 */
SDX_API sdx_status_t sdxRunAllocating(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_run_options_t* options);

SDX_API const char* sdxGetLastError(const sdx_runtime_t* runtime);
SDX_API sdx_status_t sdxGetExecutionReport(
    const sdx_context_t* context,
    sdx_execution_report_t* out_report);

/**
 * Mark an external input index as a VARIABLE (changes value between runs but
 * keeps the same shape). This enables D2D staging buffers for that input.
 * Call after sdxCreateContext(), before the first sdxRun().
 */
SDX_API sdx_status_t sdxMarkInputVariable(sdx_context_t* context, int32_t input_index);

/**
 * Mark an external input index as a PLACEHOLDER (changes both value and
 * potentially shape between runs). Placeholders always get H2D sync.
 * Call after sdxCreateContext(), before the first sdxRun().
 */
SDX_API sdx_status_t sdxMarkInputPlaceholder(sdx_context_t* context, int32_t input_index);

/**
 * Freeze shapes on the plan. After freezing, all shapes are assumed constant,
 * enabling CUDA graph capture and the argTableStable fast path.
 * Typically called after a warmup phase (a few sdxRun() calls to stabilize shapes).
 */
SDX_API sdx_status_t sdxFreezeShapes(sdx_context_t* context);

/**
 * Query the current plan phase.
 * Returns: 0=SLOT_BY_SLOT (warmup), 1=SHAPES_FROZEN, 2=REPLAYING,
 * 3=REPLAY_BLOCKED, -1 on error.
 */
SDX_API int32_t sdxGetPlanPhase(const sdx_context_t* context);

/**
 * Return a JSON array describing the functional-replay segments owned by this
 * context's compiled plan. Each entry includes its inclusive start/end op
 * indices, shape key, capture/replay status, execution counts, and op counts.
 *
 * The returned pointer is thread-local and remains valid until the next call
 * to this function on the same thread. Returns NULL for an invalid context or
 * a context without a compiled plan.
 */
SDX_API const char* sdxGetPlanSegmentsSummaryJson(
    const sdx_context_t* context);

/**
 * Get the number of executions completed on this context.
 */
SDX_API int32_t sdxGetExecutionCount(const sdx_context_t* context);

/**
 * Number of public inputs this context expects per sdxRun() call. For a
 * legacy context this covers constants, variables, and placeholders. For a
 * parameter-bound context it excludes values owned by the loaded bundle and
 * normally contains only placeholders. Returns -1 on error.
 */
SDX_API int32_t sdxGetNumInputs(const sdx_context_t* context);

/**
 * Number of output tensors the plan produces per sdxRun() call.
 * Returns -1 on error.
 */
SDX_API int32_t sdxGetNumOutputs(const sdx_context_t* context);

/**
 * Name of the public input at the given context index. Use together with
 * sdxGetNumInputs() to discover the required input binding order. In a
 * parameter-bound context, bundle-owned weights are not present. The returned
 * pointer stays valid for the context lifetime.
 * Returns NULL for invalid arguments or out-of-range indices.
 */
SDX_API const char* sdxGetInputName(const sdx_context_t* context, int32_t input_index);

/**
 * Requested output name at the given context index. The returned pointer stays
 * valid for the context lifetime. Returns NULL when the output was not created
 * from an explicit requested name, or for an invalid index.
 */
SDX_API const char* sdxGetOutputName(const sdx_context_t* context, int32_t output_index);

/**
 * Borrow a host-readable, C-contiguous output produced by the most recent
 * successful sdxRunAllocating() or sdxRun(). The runtime fills out_tensor;
 * its data and shape pointers remain valid until the next run on this context
 * or until the context is destroyed. Callers must not free either pointer.
 */
SDX_API sdx_status_t sdxGetOutputTensor(
    sdx_context_t* context,
    int32_t output_index,
    sdx_tensor_view_t* out_tensor);

#ifdef __cplusplus
}
#endif

#endif  // LIBND4J_DSP_RUNTIME_C_H
