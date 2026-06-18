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

#ifndef LIBND4J_LEVELZERO_REPLAY_HANDLE_H
#define LIBND4J_LEVELZERO_REPLAY_HANDLE_H

#include <system/common.h>

#ifdef HAVE_LEVELZERO

#include <graph/GraphReplayHandle.h>

#include <level_zero/ze_api.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * Intel Level Zero implementation of GraphReplayHandle.
 *
 * Uses Level Zero Mutable Command Lists for replay:
 *   1. Create a command list with ze_mutable_command_list_exp_desc_t
 *   2. Record kernel launches with zeCommandListAppendLaunchKernel()
 *   3. Tag each command with zeCommandListGetNextCommandIdExp() for later mutation
 *   4. Close with zeCommandListClose() to seal the command list
 *   5. Execute with zeCommandQueueExecuteCommandLists()
 *   6. Between replays, update arguments via zeCommandListUpdateMutableCommandsExp()
 *
 * The mutable command list pattern is analogous to CUDA graph replay:
 *   - Capture phase  -> appending commands to the mutable command list
 *   - Finalize phase -> zeCommandListClose() (seals the command list)
 *   - Replay phase   -> zeCommandQueueExecuteCommandLists() (re-execute)
 *   - Mutation       -> zeCommandListUpdateMutableCommandsExp() + zeCommandListClose()
 */
class SD_LIB_EXPORT LevelZeroReplayHandle : public GraphReplayHandle {
 public:
  explicit LevelZeroReplayHandle(int deviceId = 0);
  ~LevelZeroReplayHandle() override;

  // Non-copyable
  LevelZeroReplayHandle(const LevelZeroReplayHandle&) = delete;
  LevelZeroReplayHandle& operator=(const LevelZeroReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  /**
   * Begin capturing operations into a mutable command list.
   * Creates the command list with ze_mutable_command_list_exp_desc_t flag.
   * @param stream Ignored for Level Zero (command lists are explicit)
   * @return true if the mutable command list was created successfully
   */
  bool beginCapture(void* stream) override;

  /**
   * End capturing by closing the command list.
   * After close, the command list is sealed and ready for execution.
   * @param stream Ignored for Level Zero
   * @return true if zeCommandListClose() succeeded
   */
  bool endCapture(void* stream) override;

  /**
   * Finalize the captured command list for replay.
   * For Level Zero, zeCommandListClose() already finalizes the list,
   * so this transitions state from CAPTURED to READY.
   * @return true if the handle is in a valid state for replay
   */
  bool finalize() override;

  /**
   * Replay the captured command list.
   * Submits to the command queue and waits for completion via event.
   * @param stream Ignored for Level Zero (uses internal command queue)
   * @return true if execution completed successfully
   */
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "Level Zero"; }

  // ── Workspace management ──────────────────────────────────────────────

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // ── Level Zero-specific: kernel recording and mutation ────────────────

  /**
   * Record a kernel launch during capture phase.
   * Appends the kernel to the mutable command list and retrieves a
   * command ID that can be used for later mutation.
   *
   * @param kernel       Level Zero kernel handle
   * @param launchArgs   Thread group dimensions for the launch
   * @param outCommandId Output: mutable command ID for this launch
   * @return true if the kernel was appended and command ID retrieved
   */
  bool recordKernelLaunch(ze_kernel_handle_t kernel,
                          const ze_group_count_t* launchArgs,
                          uint64_t* outCommandId);

  /**
   * Update kernel arguments for a previously recorded command.
   * Uses zeCommandListUpdateMutableCommandsExp() to mutate the sealed
   * command list, then re-closes it for the next execution.
   *
   * @param commandId  Command ID from recordKernelLaunch()
   * @param kernel     Updated kernel handle (may have new argument bindings)
   * @param launchArgs Updated thread group dimensions (nullptr = keep existing)
   * @return true if mutation and re-close succeeded
   */
  bool updateKernelArgs(uint64_t commandId, ze_kernel_handle_t kernel,
                        const ze_group_count_t* launchArgs);

  /** Get the number of recorded kernel commands. */
  int getNumCommands() const { return numCommands_; }

  /** Get the Level Zero device handle. */
  ze_device_handle_t getDevice() const { return device_; }

  /** Get the Level Zero context handle. */
  ze_context_handle_t getContext() const { return context_; }

 private:
  ReplayState state_;
  int deviceId_;

  // Level Zero handles (lazily initialized in beginCapture)
  ze_driver_handle_t driver_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_command_list_handle_t cmdList_ = nullptr;
  ze_command_queue_handle_t cmdQueue_ = nullptr;
  ze_event_pool_handle_t eventPool_ = nullptr;
  ze_event_handle_t completionEvent_ = nullptr;

  int replayCount_ = 0;
  int numCommands_ = 0;

  // Track device memory allocations for cleanup
  std::vector<void*> deviceAllocations_;

  // Track host memory allocations for cleanup
  std::vector<void*> hostAllocations_;

  /**
   * Initialize Level Zero driver, device, context, command queue, and event.
   * Called lazily on first beginCapture().
   * @return true if all Level Zero objects were created successfully
   */
  bool initDevice();

  /**
   * Destroy all Level Zero handles in reverse creation order.
   * Safe to call multiple times (handles are nulled after destruction).
   */
  void cleanup();
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_LEVELZERO
#endif  // LIBND4J_LEVELZERO_REPLAY_HANDLE_H
