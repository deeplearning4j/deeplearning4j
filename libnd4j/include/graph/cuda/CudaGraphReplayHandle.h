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

#ifndef LIBND4J_CUDA_GRAPH_REPLAY_HANDLE_H
#define LIBND4J_CUDA_GRAPH_REPLAY_HANDLE_H

#include <system/common.h>

#ifdef SD_CUDA

#include <graph/GraphReplayHandle.h>
#include <execution/cuda/CudaGraphScheduler.h>

#include <memory>

namespace sd {
namespace graph {

/**
 * CUDA implementation of GraphReplayHandle.
 *
 * Wraps an existing CudaGraphHandle via composition, delegating
 * capture/instantiate/replay operations to the CUDA graph API.
 * This is a thin adapter — all CUDA-specific logic remains in CudaGraphHandle.
 *
 * Provides getNativeHandle() for CUDA-specific diagnostics (capture audit,
 * chrome trace export, debug dump) that have no cross-platform equivalent.
 */
class SD_LIB_EXPORT CudaGraphReplayHandle : public GraphReplayHandle {
 public:
  explicit CudaGraphReplayHandle(int deviceId = 0);
  ~CudaGraphReplayHandle() override;

  // Non-copyable
  CudaGraphReplayHandle(const CudaGraphReplayHandle&) = delete;
  CudaGraphReplayHandle& operator=(const CudaGraphReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "CUDA"; }

  // ── Workspace management (pool-aware) ────────────────────────────────

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // ── CUDA-specific access ──────────────────────────────────────────────

  /**
   * Get the underlying CudaGraphHandle for CUDA-specific diagnostics.
   *
   * Returns a raw (non-owning) pointer — the CudaGraphHandle's lifetime is
   * governed by this CudaGraphReplayHandle's shared_ptr member.  Callers
   * must NOT cache or extend the returned pointer past the
   * CudaGraphReplayHandle's lifetime.
   *
   * Previously returned shared_ptr by value — the refcount-increment
   * instruction (lock addl) could be scheduled after the owning unique_ptr
   * was reset, touching a freed control block (SIGSEGV with R13=0x9,
   * RSI=0xdeadbeefcafebabe).  Raw pointer eliminates this hazard entirely.
   */
  sd::cuda::CudaGraphHandle* getNativeHandle() const { return handle_.get(); }

  /**
   * Get the number of nodes in the captured CUDA graph.
   * Delegates to CudaGraphHandle::getNumNodes().
   */
  size_t getNumNodes() const;

  /**
   * Get the number of nodes during an active capture.
   * @param captureStream The stream being captured
   */
  size_t getNumNodesDuringCapture(void* captureStream) const;

  /** Get device ID this handle was created for. */
  int getDeviceId() const { return deviceId_; }

  /**
   * Returns true if the last finalize() (instantiate) failure was OOM.
   * Delegates to CudaGraphHandle::wasLastInstantiateOom().
   */
  bool wasLastInstantiateOom() const {
    return handle_ && handle_->wasLastInstantiateOom();
  }

  /**
   * Returns the CUDA error code from the last instantiate() call.
   * 0 = success or not yet attempted.
   * Delegates to CudaGraphHandle::getLastInstantiateError().
   */
  int getLastInstantiateError() const {
    return handle_ ? handle_->getLastInstantiateError() : 0;
  }

 private:
  std::shared_ptr<sd::cuda::CudaGraphHandle> handle_;
  int deviceId_;
  int replayCount_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_CUDA_GRAPH_REPLAY_HANDLE_H
