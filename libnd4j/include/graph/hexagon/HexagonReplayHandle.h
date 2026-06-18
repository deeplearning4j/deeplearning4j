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

#ifndef LIBND4J_HEXAGON_REPLAY_HANDLE_H
#define LIBND4J_HEXAGON_REPLAY_HANDLE_H

#include <system/common.h>

#ifdef HAVE_HEXAGON_MLIR

#include <graph/GraphReplayHandle.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace sd {
namespace graph {

/**
 * Hexagon NPU implementation of GraphReplayHandle.
 *
 * Uses NPU command list recording for replay, analogous to the LevelZero
 * mutable command list pattern:
 *   1. beginCapture(): create an NPU command list, begin recording dispatch commands
 *   2. Record kernel dispatches and DMA operations during capture
 *   3. endCapture(): seal the command list
 *   4. finalize(): validate the command list and transition to READY
 *   5. replay(): submit the command list to the NPU and wait for completion
 *
 * The workspace uses TCM (Tightly Coupled Memory) for staging buffers
 * between DDR and HVX vector operations. DMA operations manage transfers
 * between DDR and TCM.
 */
class SD_LIB_EXPORT HexagonReplayHandle : public GraphReplayHandle {
 public:
  explicit HexagonReplayHandle(int deviceId = 0);
  ~HexagonReplayHandle() override;

  // Non-copyable
  HexagonReplayHandle(const HexagonReplayHandle&) = delete;
  HexagonReplayHandle& operator=(const HexagonReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  /**
   * Begin capturing operations into an NPU command list.
   * Creates the command list and transitions to CAPTURING state.
   * @param stream Ignored for Hexagon NPU (command lists are explicit)
   * @return true if the command list was created successfully
   */
  bool beginCapture(void* stream) override;

  /**
   * End capturing by sealing the command list.
   * After sealing, no more commands can be appended.
   * @param stream Ignored for Hexagon NPU
   * @return true if the command list was sealed successfully
   */
  bool endCapture(void* stream) override;

  /**
   * Finalize the captured command list for replay.
   * Validates the recorded commands and transitions from CAPTURED to READY.
   * @return true if the handle is in a valid state for replay
   */
  bool finalize() override;

  /**
   * Replay the captured command list on the NPU.
   * Submits the sealed command list and waits for completion.
   * @param stream Ignored for Hexagon NPU (uses internal NPU scheduling)
   * @return true if execution completed successfully
   */
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "Hexagon NPU"; }

  // ── Workspace management ──────────────────────────────────────────────

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // ── Hexagon-specific: DMA and kernel recording ────────────────────────

  /**
   * Record a DMA transfer from host DDR to NPU TCM during capture.
   * @param tcmDst TCM destination pointer
   * @param hostSrc Host source pointer
   * @param bytes Number of bytes to transfer
   * @return true if the DMA command was recorded
   */
  bool recordDmaHostToTcm(void* tcmDst, const void* hostSrc, size_t bytes);

  /**
   * Record a DMA transfer from NPU TCM to host DDR during capture.
   * @param hostDst Host destination pointer
   * @param tcmSrc TCM source pointer
   * @param bytes Number of bytes to transfer
   * @return true if the DMA command was recorded
   */
  bool recordDmaTcmToHost(void* hostDst, const void* tcmSrc, size_t bytes);

  /**
   * Record a kernel dispatch during capture.
   * @param kernelHandle Compiled kernel handle from HexagonRuntimeManager
   * @param args Kernel argument pointers
   * @param numArgs Number of arguments
   * @return true if the dispatch command was recorded
   */
  bool recordKernelDispatch(void* kernelHandle, void** args, int numArgs);

  /** Get the number of recorded commands (DMA + kernel dispatches). */
  int getNumCommands() const { return numCommands_; }

  /** Get the NPU device ID. */
  int getDeviceId() const { return deviceId_; }

  /** Get the TCM allocation list (for diagnostics). */
  const std::vector<void*>& getTcmAllocations() const { return tcmAllocations_; }

  /** Get the DDR allocation list (for diagnostics). */
  const std::vector<void*>& getDdrAllocations() const { return ddrAllocations_; }

 private:
  ReplayState state_;
  int deviceId_;

  // NPU handles (lazily initialized in beginCapture)
  void* npuContext_ = nullptr;     // Opaque NPU device context
  void* commandList_ = nullptr;    // Opaque NPU command list

  int replayCount_ = 0;
  int numCommands_ = 0;

  // Track TCM allocations for cleanup
  std::vector<void*> tcmAllocations_;

  // Track DDR allocations for cleanup
  std::vector<void*> ddrAllocations_;

  // Recorded command descriptors for replay
  struct RecordedCommand {
    enum Type { DMA_HOST_TO_TCM, DMA_TCM_TO_HOST, KERNEL_DISPATCH };
    Type type;
    void* kernelHandle = nullptr;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    std::vector<void*> args;
  };
  std::vector<RecordedCommand> recordedCommands_;

  /**
   * Initialize the NPU device via HexagonRuntimeManager.
   * Called lazily on first beginCapture().
   * @return true if the NPU device was initialized successfully
   */
  bool initDevice();

  /**
   * Destroy all NPU handles and free allocations in reverse creation order.
   * Safe to call multiple times (handles are nulled after destruction).
   */
  void cleanup();
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
#endif  // LIBND4J_HEXAGON_REPLAY_HANDLE_H
