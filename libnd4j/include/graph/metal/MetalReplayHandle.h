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

// Author: Adam Gibson

#ifndef LIBND4J_METAL_REPLAY_HANDLE_H
#define LIBND4J_METAL_REPLAY_HANDLE_H

#include <system/common.h>

#ifdef SD_METAL

#include <graph/GraphReplayHandle.h>

#include <cstdint>
#include <memory>

namespace sd {
namespace graph {

/**
 * Apple Metal implementation of GraphReplayHandle.
 *
 * Uses Metal Indirect Command Buffers (ICBs) for command replay.
 * ICBs allow pre-encoding compute dispatches that can be replayed
 * multiple times without re-encoding, analogous to CUDA graph replay.
 *
 * Lifecycle:
 *   1. beginCapture()  -- Reset ICB, prepare for encoding
 *   2. recordComputeCommand() x N  -- Encode dispatches into ICB
 *   3. endCapture()    -- Mark ICB encoding complete
 *   4. finalize()      -- Transition to READY state (ICB is already usable)
 *   5. replay() x M    -- Execute ICB via compute encoder
 *
 * Objective-C Metal types are stored as opaque void* pointers in this
 * C++ header. The .mm implementation bridges to typed id<MTL...> handles
 * via __bridge casts under ARC.
 */
class SD_LIB_EXPORT MetalReplayHandle : public GraphReplayHandle {
 public:
  explicit MetalReplayHandle(int deviceId = 0);
  ~MetalReplayHandle() override;

  // Non-copyable, non-movable
  MetalReplayHandle(const MetalReplayHandle&) = delete;
  MetalReplayHandle& operator=(const MetalReplayHandle&) = delete;
  MetalReplayHandle(MetalReplayHandle&&) = delete;
  MetalReplayHandle& operator=(MetalReplayHandle&&) = delete;

  // -- GraphReplayHandle interface ------------------------------------------

  /**
   * Begin capturing commands into the ICB.
   * Resets the ICB and prepares for new command encoding.
   * @param stream Unused for Metal (pass nullptr). Reserved for API symmetry.
   * @return true if ICB was reset and capture started successfully
   */
  bool beginCapture(void* stream) override;

  /**
   * End capturing. Marks the ICB encoding as complete.
   * @param stream Unused for Metal (pass nullptr).
   * @return true if capture ended successfully
   */
  bool endCapture(void* stream) override;

  /**
   * Finalize the captured ICB for replay.
   * For Metal ICBs this is essentially a state transition -- the ICB
   * is usable immediately after encoding completes.
   * @return true if state transitioned to READY
   */
  bool finalize() override;

  /**
   * Replay the captured ICB.
   * Creates a command buffer, encodes ICB execution via a compute encoder,
   * commits, and waits for completion.
   * @param stream Unused for Metal (pass nullptr).
   * @return true if replay executed successfully
   */
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "Metal ICB"; }

  // -- Workspace management -------------------------------------------------

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // -- Metal-specific: record compute dispatches during capture --------------

  /**
   * Record a compute dispatch command into the ICB at the current index.
   * Must be called between beginCapture() and endCapture().
   *
   * @param pipelineState  id<MTLComputePipelineState> as void*
   * @param buffer         id<MTLBuffer> as void* (bound at bufferIndex)
   * @param bufferIndex    Metal buffer binding index
   * @param threadsPerGroup  Threads per threadgroup (1D dispatch)
   * @param threadgroupsPerGrid  Number of threadgroups (1D dispatch)
   * @return true if the command was recorded successfully
   */
  bool recordComputeCommand(void* pipelineState,
                            void* buffer,
                            uint32_t bufferIndex,
                            uint32_t threadsPerGroup,
                            uint32_t threadgroupsPerGrid);

  /** Maximum number of commands the ICB can hold. */
  int getMaxCommands() const;

  /**
   * Set the maximum command capacity.
   * Must be called before beginCapture() to take effect; triggers ICB
   * recreation if Metal is already initialized.
   */
  void setMaxCommands(int max);

  /** Number of commands recorded in the current/last capture. */
  int getNumCommands() const;

  /** Device ID this handle was created for. */
  int getDeviceId() const;

 private:
  ReplayState state_ = ReplayState::EMPTY;
  int deviceId_;
  int maxCommands_ = 256;

  // Opaque Metal handles (bridge to Objective-C in .mm file)
  void* device_ = nullptr;          // id<MTLDevice>
  void* commandQueue_ = nullptr;    // id<MTLCommandQueue>
  void* icb_ = nullptr;             // id<MTLIndirectCommandBuffer>

  // Workspace buffer (Metal GPU-private storage)
  void* workspaceBuffer_ = nullptr; // id<MTLBuffer>

  int replayCount_ = 0;
  int numCommands_ = 0;

  /**
   * Initialize Metal device, command queue, and ICB.
   * @return true if all Metal objects were created successfully
   */
  bool initMetal();

  /**
   * Recreate the ICB with the current maxCommands_ capacity.
   * Called by setMaxCommands() and initMetal().
   * @return true if ICB was created successfully
   */
  bool createICB();

  /**
   * Release all Metal objects and reset state.
   */
  void cleanup();
};

}  // namespace graph
}  // namespace sd

#endif  // SD_METAL
#endif  // LIBND4J_METAL_REPLAY_HANDLE_H
