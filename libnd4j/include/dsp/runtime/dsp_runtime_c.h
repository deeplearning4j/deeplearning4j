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
  /* For CUDA/AMD tensors, device_id must be >= 0 and consistent across all GPU tensors in a run. */
  int32_t device_id;
} sdx_tensor_view_t;

typedef struct {
  uint32_t struct_size;
  int32_t requested_backend;
  int32_t applied_backend;
  int32_t status_code;
  int32_t used_fallback;
  uint64_t execution_time_ns;
  int32_t requested_gpu_target;
  int32_t applied_gpu_target;
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

#ifdef __cplusplus
}
#endif

#endif  // LIBND4J_DSP_RUNTIME_C_H
