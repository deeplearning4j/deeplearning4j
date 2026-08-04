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

#ifndef LIBND4J_HIP_GRAPH_REPLAY_HANDLE_H
#define LIBND4J_HIP_GRAPH_REPLAY_HANDLE_H

#include <system/common.h>

// Requires real HIP headers and is owned exclusively by native SD_HIP builds.
// CUDA/ZLUDA replay stays on the CUDA ABI to avoid mixing CUDA and ROCm types.
#if defined(SD_HIP)

#include <graph/GraphReplayHandle.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <memory>
#include <mutex>

namespace sd {
namespace graph {

/**
 * HIP (AMD ROCm) implementation of GraphReplayHandle.
 *
 * Wraps hipGraph_t and hipGraphExec_t directly, delegating
 * capture/instantiate/replay operations to the HIP graph API.
 * The HIP graph API mirrors CUDA graphs nearly 1:1.
 *
 * Unlike the CUDA version (which delegates to CudaGraphHandle),
 * this class manages the native HIP graph objects directly for
 * simplicity — no intermediate handle class needed.
 */
class SD_LIB_EXPORT HipGraphReplayHandle : public GraphReplayHandle {
 public:
  explicit HipGraphReplayHandle(int deviceId = 0);
  ~HipGraphReplayHandle() override;

  // Non-copyable
  HipGraphReplayHandle(const HipGraphReplayHandle&) = delete;
  HipGraphReplayHandle& operator=(const HipGraphReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "HIP (ROCm)"; }

  // ── Workspace management (raw hipMalloc/hipFree) ────────────────────

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // ── HIP-specific access ──────────────────────────────────────────────

  /**
   * Get the number of nodes in the captured HIP graph.
   * Only valid after endCapture() succeeds.
   */
  size_t getNumNodes() const;

  /**
   * Get the number of nodes during an active capture.
   * @param captureStream The stream being captured
   */
  size_t getNumNodesDuringCapture(void* captureStream) const;

  /** Get device ID this handle was created for. */
  int getDeviceId() const { return deviceId_; }

  /** Get native HIP graph (use with caution). */
  hipGraph_t getGraph() const { return graph_; }

  /** Get native HIP graph exec (use with caution). */
  hipGraphExec_t getGraphExec() const { return graphExec_; }

 private:
  void cleanup();

  hipGraph_t graph_ = nullptr;
  hipGraphExec_t graphExec_ = nullptr;
  ReplayState state_ = ReplayState::EMPTY;
  int deviceId_ = 0;

  // Statistics tracking
  int replayCount_ = 0;
  double captureTimeMs_ = 0.0;
  double lastReplayTimeMs_ = 0.0;
  size_t numNodes_ = 0;

  mutable std::mutex mutex_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
#endif  // LIBND4J_HIP_GRAPH_REPLAY_HANDLE_H
