/* ******************************************************************************
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

// cuBLAS workspace management for NativeDynamicShapePlan.
// Separated into a .cu file because cublas_v2.h includes cuda_fp16.h which
// defines __half with constructors/members that conflict with our float16.h
// when compiled by g++ (non-nvcc). This file is compiled by nvcc where both
// __half definitions are compatible.

#include <graph/NativeDynamicShapePlan.h>
#include <cublas_v2.h>
#include <helpers/cublasHelper.h>

// Thread-local cuBLAS workspace for MmulHelper to re-apply after cublasSetStream.
// cublasSetStream resets the user-provided workspace (per cuBLAS docs), so
// MmulHelper::reapplyCublasWorkspace() reads these to restore it.
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;

namespace sd {
namespace graph {

void NativeDynamicShapePlan::ensureCublasWorkspace(size_t minBytes) {
  if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ >= minBytes) {
    return;  // Already have a large enough workspace
  }

  // Free old workspace if it exists
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }

  // Allocate workspace on the current device
  auto err = cudaMalloc(&cublasWorkspaceBuffer_, minBytes);
  if (err != cudaSuccess) {
    sd_printf("NativeDynamicShapePlan: failed to allocate cuBLAS workspace (%zu bytes): %s\n",
              minBytes, cudaGetErrorString(err));
    cudaGetLastError();  // Clear sticky error
    return;
  }
  cublasWorkspaceSize_ = minBytes;
  sd_printf("NativeDynamicShapePlan: allocated cuBLAS workspace: %zu MB\n",
            minBytes / (1024 * 1024));
}

void NativeDynamicShapePlan::setCublasWorkspaceForCapture(void* stream) {
  if (cublasWorkspaceBuffer_ == nullptr) return;

  // Get the cuBLAS handle for the current device.
  // CublasHelper::handle() returns void* which is cublasHandle_t* (pointer to the handle).
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  // Set the cuBLAS handle to use the capture stream so GEMM ops are recorded
  // into the graph on the correct stream
  cublasSetStream_v2(*handlePtr, stream != nullptr
      ? *static_cast<cudaStream_t*>(stream) : nullptr);

  // Set explicit workspace so cuBLAS doesn't do internal cudaMalloc during capture.
  auto status = cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    sd_printf("NativeDynamicShapePlan: cublasSetWorkspace failed: %d\n", static_cast<int>(status));
  }

  // Store in thread-locals so MmulHelper can re-apply after cublasSetStream
  // (cublasSetStream resets the user-provided workspace per cuBLAS docs).
  tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
  tl_cublasWorkspaceSize = cublasWorkspaceSize_;
}

void NativeDynamicShapePlan::restoreCublasWorkspaceAfterCapture(void* stream) {
  // Reset cuBLAS to use its own internal workspace allocation again.
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  cublasSetWorkspace(*handlePtr, nullptr, 0);

  // Clear thread-locals so MmulHelper stops re-applying workspace
  tl_cublasWorkspacePtr = nullptr;
  tl_cublasWorkspaceSize = 0;
}

}  // namespace graph
}  // namespace sd
