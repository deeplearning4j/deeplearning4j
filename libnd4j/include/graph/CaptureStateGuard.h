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

#ifndef LIBND4J_CAPTURE_STATE_GUARD_H
#define LIBND4J_CAPTURE_STATE_GUARD_H

#include <system/common.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#include <array/DataBuffer.h>

namespace sd {
namespace graph {

/**
 * RAII guard for the thread-local CUDA graph capture state.
 *
 * On construction, saves all capture-related TLS variables and installs
 * the new capture state. On destruction, restores the previous state.
 * If commit() was NOT called, captured host pointers are freed (capture
 * failed). If commit() WAS called, host pointer ownership has been
 * transferred to the replay handle by the caller.
 *
 * This eliminates the 6+ duplicated cleanup code blocks across
 * _cudagraph.cu and _gpubackend.cpp. Exception-safe: if any code
 * between construction and destruction throws, TLS is always restored.
 *
 * Usage:
 *   {
 *     CaptureStateGuard guard(captureStream, wsPtr, wsSize, hostWs, hostWsSize);
 *     // ... capture ops ...
 *     guard.commit();  // on success: host ptrs NOT freed in destructor
 *   } // TLS restored automatically; host ptrs freed if !committed
 */
class CaptureStateGuard {
 public:
  CaptureStateGuard(cudaStream_t captureStream,
                    void* captureWs, size_t captureWsSize,
                    void* hostWs, size_t hostWsSize)
      : prevCaptureStream_(reinterpret_cast<cudaStream_t>(tl_graphCaptureStream)),
        prevGraphActive_(tl_graphExecutionActive),
        committed_(false) {
    tl_graphExecutionActive = true;
    tl_graphCaptureStream = captureStream;
    tl_captureWorkspace = captureWs;
    tl_captureWorkspaceSize = captureWsSize;
    tl_captureWorkspaceOffset = 0;
    tl_captureHostWorkspace = hostWs;
    tl_captureHostWorkspaceSize = hostWsSize;
    tl_captureHostWorkspaceOffset = 0;
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
  }

  ~CaptureStateGuard() {
    tl_graphExecutionActive = prevGraphActive_;
    tl_graphCaptureStream = prevCaptureStream_;
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    tl_captureHostWorkspace = nullptr;
    tl_captureHostWorkspaceSize = 0;
    tl_captureHostWorkspaceOffset = 0;
    if (!committed_) {
      for (auto* ptr : tl_capturedHostPtrs) {
        if (ptr != nullptr) cudaFreeHost(ptr);
      }
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
  }

  /**
   * Mark capture as successful. Caller has transferred host pointer
   * ownership to the replay handle — destructor will NOT free them.
   */
  void commit() { committed_ = true; }

  /**
   * Track a host pointer for lifetime management. If capture fails
   * (no commit()), this pointer will be freed in the destructor.
   */
  void trackHostPtr(void* ptr) {
    if (ptr != nullptr) tl_capturedHostPtrs.push_back(ptr);
  }

  /** Get workspace bytes consumed so far (for diagnostics). */
  size_t workspaceUsed() const { return tl_captureWorkspaceOffset; }

  /** Get the capture stream that was saved. */
  cudaStream_t previousCaptureStream() const { return prevCaptureStream_; }

  // Non-copyable, non-movable
  CaptureStateGuard(const CaptureStateGuard&) = delete;
  CaptureStateGuard& operator=(const CaptureStateGuard&) = delete;
  CaptureStateGuard(CaptureStateGuard&&) = delete;
  CaptureStateGuard& operator=(CaptureStateGuard&&) = delete;

 private:
  cudaStream_t prevCaptureStream_;
  bool prevGraphActive_;
  bool committed_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_CAPTURE_STATE_GUARD_H
