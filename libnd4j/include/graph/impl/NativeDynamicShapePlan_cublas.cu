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
#include <system/Environment.h>
#include <cublas_v2.h>
#include <helpers/cublasHelper.h>

namespace sd {

// Thread-local cuBLAS workspace for MmulHelper to re-apply after cublasSetStream.
// cublasSetStream resets the user-provided workspace (per cuBLAS docs), so
// MmulHelper::reapplyCublasWorkspace() reads these to restore it.
// All defined in DataBuffer.cu inside namespace sd.
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;
extern SD_TLS_EXPORT thread_local bool   tl_cublasLtDisabled;
extern SD_TLS_EXPORT thread_local bool   tl_graphExecutionActive;

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
    cudaGetLastError();
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
  if (handlePtr == nullptr) {
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: cuBLAS handle is NULL — cannot configure");
    return;
  }

  cudaStream_t resolvedStream = stream != nullptr ? *static_cast<cudaStream_t*>(stream) : nullptr;
  DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: setting cuBLAS stream=%p (from void*=%p) "
           "tl_graphExecutionActive=%d tl_cublasLtDisabled=%d",
           (void*)resolvedStream, stream, (int)tl_graphExecutionActive, (int)tl_cublasLtDisabled);

  // Set the cuBLAS handle to use the capture stream so GEMM ops are recorded
  // into the graph on the correct stream.
  cublasSetStream_v2(*handlePtr, resolvedStream);

  // ── Deterministic capture (CUDA_GRAPHS mode) ──────────────────────────────
  // When tl_cublasLtDisabled is set (CUDA_GRAPHS and SLOT_BY_SLOT modes), we
  // must NOT provide a workspace to cuBLAS. With workspace available, cuBLAS may
  // select workspace-using algorithms (split-K variants) whose internal reduction
  // order differs between graph capture and graph replay. This produces tiny FP
  // differences that compound through GDN recurrent state until token divergence.
  //
  // Without workspace, cuBLAS is forced to select algorithms that DON'T use
  // workspace scratch — matching exactly what SLOT_BY_SLOT does during live
  // execution. For M=1 decode GEMV (the entire decode hot path), cuBLAS does NOT
  // need workspace for the basic non-split-K algorithm.
  //
  // We explicitly call cublasSetWorkspace(handle, nullptr, 0) to clear any
  // previously-set workspace on the handle, ensuring the captured kernels are
  // identical to what live execution would produce.
  if (tl_cublasLtDisabled) {
    cublasSetWorkspace(*handlePtr, nullptr, 0);
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: workspace CLEARED (deterministic mode, "
             "tl_cublasLtDisabled=true) — forces non-split-K algorithms matching SLOT_BY_SLOT");
    return;
  }

  // Explicit workspace prevents cuBLAS from creating per-GEMM MemAlloc/MemFree
  // graph nodes during capture, which cause OOM on graph launch.
  // This path is only used by TRITON composite mode (which manages its own
  // capture segments around matmul gaps).
  bool useCublasWorkspace = sd::Environment::getInstance().cublasCaptureWorkspace();

  if (useCublasWorkspace) {
    ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
    cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
    tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
    tl_cublasWorkspaceSize = cublasWorkspaceSize_;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: workspace SET buffer=%p size=%zuMB",
             cublasWorkspaceBuffer_, cublasWorkspaceSize_ / (1024*1024));
  } else {
    // Clear any previously set workspace so cuBLAS uses its own internal allocator
    cublasSetWorkspace(*handlePtr, nullptr, 0);
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: workspace DISABLED (cublasCaptureWorkspace=false)");
  }
}

void NativeDynamicShapePlan::setCublasWorkspaceForWarmup() {
  // In deterministic mode (tl_cublasLtDisabled = CUDA_GRAPHS or SLOT_BY_SLOT),
  // warmup must see the same cuBLAS state as capture: no workspace. This ensures
  // cuBLAS selects the same non-split-K algorithm during warmup, capture, and
  // replay, producing bit-identical results.
  if (tl_cublasLtDisabled) {
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetWorkspace(*handlePtr, nullptr, 0);
    }
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    return;
  }

  // Non-deterministic mode (TRITON): same workspace as capture so cuBLAS
  // selects identical algorithms during warmup and capture.
  if (!sd::Environment::getInstance().cublasCaptureWorkspace()) return;

  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
  cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);

  tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
  tl_cublasWorkspaceSize = cublasWorkspaceSize_;
}

void NativeDynamicShapePlan::restoreCublasWorkspaceAfterCapture(void* stream) {
  // Reset cuBLAS to use its own internal workspace allocation again.
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) {
    DSP_DIAG(EXECUTE, "restoreCublasWorkspaceAfterCapture: cuBLAS handle is NULL — cannot restore");
    return;
  }

  DSP_DIAG(EXECUTE, "restoreCublasWorkspaceAfterCapture: clearing cuBLAS workspace "
           "(was ptr=%p size=%zu) tl_graphExecutionActive=%d",
           (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize,
           (int)tl_graphExecutionActive);

  cublasSetWorkspace(*handlePtr, nullptr, 0);

  // Clear thread-locals so MmulHelper stops re-applying workspace
  tl_cublasWorkspacePtr = nullptr;
  tl_cublasWorkspaceSize = 0;
}

}  // namespace graph
}  // namespace sd
