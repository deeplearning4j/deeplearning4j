/******************************************************************************
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

//
// Capture-safe helper functions for ExtraArguments.
//
// ExtraArguments.cpp is compiled by GCC but needs access to NVCC-defined
// thread_local variables (tl_graphExecutionActive, tl_captureWorkspace, etc.).
// GCC and NVCC have incompatible TLS models on Linux, so direct extern access
// from .cpp files causes linker errors. These thin wrappers live in a .cu file
// (compiled by NVCC) and provide the same functionality via function calls.
//

#include <array/ExtraArguments_cuda.h>
#include <cuda_runtime.h>
#include <array/DataBuffer.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/DebugHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/op_boilerplate.h>

namespace sd {

// Forward declarations — definitions follow after extra_args_detail.
bool extraArgsCaptureActive();
void extraArgsCaptureH2D(void* dst, const void* src, size_t bytes);

namespace extra_args_detail {

void* extraArgsAllocDevice(size_t bytes) {
  int deviceId = sd::AffinityManager::currentDeviceId();
  auto ptr = sd::memory::CudaMemoryPool::getInstance().allocate(bytes, deviceId);
  if (!ptr) THROW_EXCEPTION("ExtraArguments: CudaMemoryPool::allocate failed");
  return ptr;
}

void extraArgsFreeDevice(void* ptr) {
  if (ptr == nullptr) return;
  int deviceId = sd::AffinityManager::currentDeviceId();
  sd::memory::CudaMemoryPool::getInstance().free(ptr, deviceId);
}

void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes) {
  // extraArgsCaptureActive and extraArgsCaptureH2D are defined in namespace sd
  // (below, in the same TU), not in extra_args_detail.
  if (tl_graphExecutionActive) {
    // Inside per-group CUDA graph capture: synchronous cudaMemcpy on the legacy
    // stream poisons the capture stream with error 901. Use the async capture path.
    // Do NOT synchronize — the copy is recorded into the graph, not executed now.
    ::sd::extraArgsCaptureH2D(dst, src, bytes);
  } else {
    // Outside per-group capture: use a capture-safe stream (composite scope stream
    // when set, otherwise LaunchContext default stream), then synchronize so callers
    // can safely read dst.  Avoids error 906 from legacy stream 0 AND avoids
    // cudaStreamPerThread which can cause ordering issues near capture boundaries.
    ::sd::extraArgsCaptureH2D(dst, src, bytes);
    // Sync on the SAME stream the copy used. captureSafeStream resolves it (here, outside
    // per-group capture: the composite outer stream if set, else the LaunchContext default) —
    // the authority, not an inline re-derivation. The tl_graphExecutionActive gating above
    // keeps this off the per-group capture path; between groups this is the ordering barrier.
    auto* sp = LaunchContext::defaultContext()->getCudaStream();
    cudaStream_t fallback = (sp != nullptr) ? *sp : cudaStreamPerThread;
    cudaStreamSynchronize(DebugHelper::captureSafeStream(fallback));
  }
}

}  // namespace extra_args_detail


bool extraArgsCaptureActive() {
  // True during per-group CUDA graph capture.
  // Note: we intentionally do NOT include tl_compositeCaptureStream here —
  // that scope is for outer composite-capture routing of async ops only, not
  // for suppressing host-side synchronization (which is needed between groups).
  return tl_graphExecutionActive;
}

void* extraArgsCaptureDevAlloc(size_t bytes) {
  if (!tl_graphExecutionActive || tl_captureWorkspace == nullptr) return nullptr;
  size_t aligned = (bytes + 255) & ~255ULL;
  if (tl_captureWorkspaceOffset + aligned > tl_captureWorkspaceSize) return nullptr;
  void* ptr = static_cast<char*>(tl_captureWorkspace) + tl_captureWorkspaceOffset;
  tl_captureWorkspaceOffset += aligned;
  return ptr;
}

void* extraArgsCaptureHostAlloc(size_t bytes) {
  if (!tl_graphExecutionActive || tl_captureHostWorkspace == nullptr) return nullptr;
  size_t aligned = (bytes + 255) & ~255ULL;
  if (tl_captureHostWorkspaceOffset + aligned > tl_captureHostWorkspaceSize) return nullptr;
  void* ptr = static_cast<char*>(tl_captureHostWorkspace) + tl_captureHostWorkspaceOffset;
  tl_captureHostWorkspaceOffset += aligned;
  return ptr;
}

void extraArgsCaptureH2D(void* dst, const void* src, size_t bytes) {
  // Record onto the active capture stream (per-group / composite / ground-truth), else the
  // LaunchContext default — never cudaStreamPerThread here. Authority: DebugHelper::
  // captureSafeStream. Routing through it also avoids a STALE tl_graphCaptureStream between
  // merged groups (per-group flag cleared but the pointer may linger).
  auto* sp = LaunchContext::defaultContext()->getCudaStream();
  cudaStream_t fallback = (sp != nullptr) ? *sp : nullptr;
  cudaStream_t stream = DebugHelper::captureSafeStream(fallback);
  if (stream != nullptr) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
  }
}

}  // namespace sd
