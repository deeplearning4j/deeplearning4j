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
#include <graph/ModeContract.h>
#include <graph/DspDiagnostics.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/Environment.h>
#include <cublas_v2.h>
#include <helpers/cublasHelper.h>

namespace sd {

namespace graph {

void NativeDynamicShapePlan::ensureCublasWorkspace(size_t minBytes) {
  if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ >= minBytes) {
    return;  // Already have a large enough workspace
  }

  int deviceId = 0;
  cudaGetDevice(&deviceId);
  auto& pool = memory::CudaMemoryPool::getInstance();

  // Free old workspace if it exists
  if (cublasWorkspaceBuffer_ != nullptr) {
    pool.free(cublasWorkspaceBuffer_, deviceId);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
    cublasWorkspaceDevice_ = -1;
  }

  // Allocate workspace on the current device.
  // allocateDirect: this buffer is set on the cuBLAS handle before CUDA graph capture
  // and baked as a workspace pointer into captured GEMM nodes — it must survive across
  // capture/replay cycles without going through the async pool.
  cublasWorkspaceBuffer_ = pool.allocateDirect(minBytes, deviceId);
  if (cublasWorkspaceBuffer_ == nullptr) {
    DSP_DIAG(MEMORY, "failed to allocate cuBLAS workspace (%zu bytes) on device %d",
             minBytes, deviceId);
    return;
  }
  cublasWorkspaceSize_ = minBytes;
  cublasWorkspaceDevice_ = deviceId;  // record alloc device for safe teardown free
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

  // CUDA graph capture bakes cuBLAS GEMM algorithm choices into the graph. Honor
  // the execution-mode contract here: AUTO/TRITON/CUDA_GRAPHS require the same
  // deterministic cuBLAS policy during warmup, capture, and live replay. Captured
  // external-workspace gaps otherwise use tensor-core/default math while live gaps
  // use PEDANTIC, which is enough to change VLM decode tokens.
  const bool deterministic = ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas;
  cublasSetMathMode(*handlePtr, deterministic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH);
  tl_cublasLtDisabled = true;

  // ── Workspace configuration ───────────────────────────────────────────────
  // Provide explicit workspace to prevent per-GEMM MemAlloc/MemFree graph nodes.
  bool useCublasWorkspace = sd::Environment::getInstance().cublasCaptureWorkspace();

  if (useCublasWorkspace) {
    ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
    cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
    tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
    tl_cublasWorkspaceSize = cublasWorkspaceSize_;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: %s + workspace SET buffer=%p size=%zuMB "
             "tl_cublasLtDisabled=1",
             deterministic ? "PEDANTIC_MATH" : "DEFAULT_MATH",
             cublasWorkspaceBuffer_, cublasWorkspaceSize_ / (1024*1024));
  } else {
    cublasSetWorkspace(*handlePtr, nullptr, 0);
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: %s + workspace DISABLED "
             "tl_cublasLtDisabled=1",
             deterministic ? "PEDANTIC_MATH" : "DEFAULT_MATH");
  }
}

void NativeDynamicShapePlan::setCublasWorkspaceForWarmup() {
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) return;

  // Warmup must use the SAME cuBLAS settings as capture so that cuBLAS selects
  // identical GEMM algorithms. Algorithm selection depends on math mode and
  // workspace availability — any mismatch means the warmup pre-allocates output
  // buffers with one algorithm's layout but capture records a different algorithm,
  // causing shape/result divergence on replay.
  //
  const bool deterministic = ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas;
  cublasSetMathMode(*handlePtr, deterministic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH);
  tl_cublasLtDisabled = true;

  // Workspace: match capture's workspace configuration.
  bool useCublasWorkspace = sd::Environment::getInstance().cublasCaptureWorkspace();
  if (useCublasWorkspace) {
    ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
    cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
    tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
    tl_cublasWorkspaceSize = cublasWorkspaceSize_;
  } else {
    cublasSetWorkspace(*handlePtr, nullptr, 0);
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
  }
}

void NativeDynamicShapePlan::restoreCublasWorkspaceAfterCapture(void* stream) {
  // Reset cuBLAS to use its own internal workspace allocation again.
  auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
  if (handlePtr == nullptr) {
    DSP_DIAG(EXECUTE, "restoreCublasWorkspaceAfterCapture: cuBLAS handle is NULL — cannot restore");
    return;
  }

  DSP_DIAG(EXECUTE, "restoreCublasWorkspaceAfterCapture: clearing cuBLAS workspace "
           "(was ptr=%p size=%zu) tl_graphExecutionActive=%d tl_cublasLtDisabled=%d",
           (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize,
           (int)tl_graphExecutionActive, (int)tl_cublasLtDisabled);

  cublasSetWorkspace(*handlePtr, nullptr, 0);

  // cuBLAS math mode for live gap ops after capture ends:
  // For modes with requiresDeterministicCublas=true (AUTO), platformBeginExecution
  // or platformSetDeterministicCublas already sets PEDANTIC for live execution.
  // Captured CUDA graphs are immune to runtime math mode — they replay the exact
  // algorithms selected at capture time.
  //
  // platformEndExecution restores the cuBLAS math mode to DEFAULT at plan end.
  if (!ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    tl_cublasLtDisabled = false;
  }

  // Clear thread-locals so MmulHelper stops re-applying workspace
  tl_cublasWorkspacePtr = nullptr;
  tl_cublasWorkspaceSize = 0;
}

}  // namespace graph
}  // namespace sd
