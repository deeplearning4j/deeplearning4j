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
  SDX_BACKEND_AUTO = 0,
  SDX_BACKEND_SLOT_BY_SLOT = 1,
  SDX_BACKEND_CUDA_GRAPHS = 2,
  SDX_BACKEND_NVRTC = 3,
  SDX_BACKEND_PTX = 4,
  SDX_BACKEND_TRITON = 5,
  SDX_BACKEND_MLX = 6,
  SDX_BACKEND_ARM_HYBRID = 7,
  SDX_BACKEND_NNAPI = 8
} sdx_backend_t;

typedef enum {
  SDX_DEVICE_HOST = 0,
  SDX_DEVICE_CUDA = 1,
  SDX_DEVICE_AMD = 2
} sdx_device_type_t;

typedef enum {
  SDX_GPU_TARGET_AUTO = 0,
  SDX_GPU_TARGET_CUDA = 1,
  SDX_GPU_TARGET_AMD = 2
} sdx_gpu_target_t;

typedef struct {
  uint32_t struct_size;
} sdx_runtime_options_t;

typedef struct {
  uint32_t struct_size;
  int32_t backend;
  int32_t strict_backend;
  int32_t allow_runtime_jit;
  int32_t gpu_target;
} sdx_model_options_t;

typedef struct {
  uint32_t struct_size;
  int32_t backend;
  int32_t strict_signature;
  int32_t gpu_target;
} sdx_run_options_t;

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

SDX_API sdx_status_t sdxLoadBundle(
    sdx_runtime_t* runtime,
    const char* bundle_path,
    const sdx_model_options_t* options,
    sdx_model_t** out_model);
SDX_API void sdxUnloadModel(sdx_model_t* model);

SDX_API sdx_status_t sdxCreateContext(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    sdx_context_t** out_context);
SDX_API void sdxDestroyContext(sdx_context_t* context);

SDX_API sdx_status_t sdxRun(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_tensor_view_t* outputs,
    int32_t num_outputs,
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
 * Get the number of executions completed on this context.
 */
SDX_API int32_t sdxGetExecutionCount(const sdx_context_t* context);

/**
 * Number of external inputs the plan expects per sdxRun() call. External
 * inputs cover the model's constants, variables, AND placeholders — callers
 * must bind a tensor for every one of them, positionally in plan order.
 * Returns -1 on error.
 */
SDX_API int32_t sdxGetNumInputs(const sdx_context_t* context);

/**
 * Number of output tensors the plan produces per sdxRun() call.
 * Returns -1 on error.
 */
SDX_API int32_t sdxGetNumOutputs(const sdx_context_t* context);

/**
 * Name of the external input at the given plan index. Use together with
 * sdxGetNumInputs() to discover the required input binding order for a
 * loaded model. The returned pointer stays valid for the context lifetime.
 * Returns NULL for invalid arguments or out-of-range indices.
 */
SDX_API const char* sdxGetInputName(const sdx_context_t* context, int32_t input_index);

#ifdef __cplusplus
}
#endif

#endif  // LIBND4J_DSP_RUNTIME_C_H
