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

  // ── Deterministic cuBLAS for ALL capture modes ────────────────────────────
  // CUDA graph capture bakes cuBLAS GEMM algorithm choices into the graph.
  // Without CUBLAS_PEDANTIC_MATH, cuBLAS may select nondeterministic algorithms
  // (split-K with nondeterministic reduction order, TF32 tensor ops with varying
  // threadblock scheduling). Two independent captures of the same operations can
  // select different algorithms, producing numerically different results that
  // compound through recurrent/autoregressive state until token divergence.
  //
  // CUBLAS_DEFAULT_MATH allows cuBLAS to use tensor cores and optimal algorithms
  // during CUDA graph capture. At commit 0e221dd2c8 (86 tok/s), no math mode was
  // ever set — cuBLAS used its default which includes tensor cores.
  //
  // Previously CUBLAS_PEDANTIC_MATH was used here for bitwise determinism, but this
  // baked SLOW algorithms into captured CUDA graphs (causing 7.24 tok/s when gap
  // matmuls were captured). Decode-phase matmuls are M=1 GEMVs that are memory-
  // bandwidth-bound — algorithm selection barely matters for them, and the overhead
  // of slow algorithms in prefill-sized captures is catastrophic.
  //
  // Workspace is still provided (when cublasCaptureWorkspace=true) to prevent
  // cuBLAS from inserting MemAlloc/MemFree graph nodes during capture (OOM).
  //
  // cublasLt remains disabled during capture: cublasLt has its own internal
  // allocation patterns that can break CUDA graph capture.
  cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
  tl_cublasLtDisabled = true;

  // ── Workspace configuration ───────────────────────────────────────────────
  // Provide explicit workspace to prevent per-GEMM MemAlloc/MemFree graph nodes.
  bool useCublasWorkspace = sd::Environment::getInstance().cublasCaptureWorkspace();

  if (useCublasWorkspace) {
    ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
    cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
    tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
    tl_cublasWorkspaceSize = cublasWorkspaceSize_;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: DEFAULT_MATH + workspace SET buffer=%p size=%zuMB "
             "tl_cublasLtDisabled=1",
             cublasWorkspaceBuffer_, cublasWorkspaceSize_ / (1024*1024));
  } else {
    cublasSetWorkspace(*handlePtr, nullptr, 0);
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    DSP_DIAG(EXECUTE, "setCublasWorkspaceForCapture: DEFAULT_MATH + workspace DISABLED "
             "tl_cublasLtDisabled=1");
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
  // Matches capture's CUBLAS_DEFAULT_MATH (allows tensor cores + optimal algorithms).
  cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
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
  // algorithms selected at capture time (now CUBLAS_DEFAULT_MATH with tensor cores).
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
