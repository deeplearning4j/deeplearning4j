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
#include <graph/DspDiagnostics.h>
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
    DSP_DIAG(MEMORY, "failed to allocate cuBLAS workspace (%zu bytes): %s",
             minBytes, cudaGetErrorString(err));
    cudaGetLastError();  // Clear sticky error
    return;
  }
  cublasWorkspaceSize_ = minBytes;
  DSP_DIAG(MEMORY, "allocated cuBLAS workspace: %zu MB",
           minBytes / (1024 * 1024));
}

void NativeDynamicShapePlan::setCublasWorkspaceForCapture(void* stream) {
  // Get the cuBLAS handle for the current device.
  // CublasHelper::handle() returns void* which is cublasHandle_t* (pointer to the handle).
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  // Set the cuBLAS handle to use the capture stream so GEMM ops are recorded
  // into the graph on the correct stream.
  cublasSetStream_v2(*handlePtr, stream != nullptr
      ? *static_cast<cudaStream_t*>(stream) : nullptr);

  // NOTE: We intentionally do NOT set an explicit cuBLAS workspace here.
  // A 256MB workspace causes cuBLAS to select different GEMM algorithms (e.g.,
  // split-K) than it uses without workspace, producing different floating-point
  // rounding that compounds across ~3840 ops and corrupts VLM output.
  // Instead, we rely on CUDA 12's relaxed capture mode to handle cuBLAS's
  // internal workspace allocations. cuBLAS may create MemAlloc/MemFree graph
  // nodes during capture, which CUDA 12.x supports natively.
  // If this causes SIGSEGV on replay, set ND4J_CUBLAS_CAPTURE_WORKSPACE=1
  // to re-enable explicit workspace (at the cost of accuracy).
  static bool useCublasWorkspace = []() {
    const char* v = std::getenv("ND4J_CUBLAS_CAPTURE_WORKSPACE");
    return v != nullptr && std::string(v) == "1";
  }();

  if (useCublasWorkspace) {
    ensureCublasWorkspace(256 * 1024 * 1024);
    cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
    tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
    tl_cublasWorkspaceSize = cublasWorkspaceSize_;
  } else {
    // Clear any previously set workspace so cuBLAS uses its own internal allocator
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
  }
}

void NativeDynamicShapePlan::setCublasWorkspaceForWarmup() {
  // Set the SAME explicit workspace during warmup as during capture.
  // This ensures cuBLAS selects identical GEMM algorithms in both paths,
  // preventing floating-point divergence from different algorithm choices.
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  ensureCublasWorkspace(256 * 1024 * 1024);
  cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);

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
