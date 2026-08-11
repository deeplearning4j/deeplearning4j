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

#ifndef LIBND4J_VULKAN_SEGMENT_RECORDER_H
#define LIBND4J_VULKAN_SEGMENT_RECORDER_H

#include <graph/GraphBackend.h>
#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>

#include <cstddef>
#include <string>
#include <vector>

namespace sd {
class DataBuffer;
class NDArray;
namespace graph {

struct NativeSlot;
struct VulkanDeviceCaps;
class RandomGenerator;
class VulkanExecutionStream;
class VulkanReplayHandle;
class VulkanPipelineCache;

/**
 * VulkanSegmentRecorder — bridges nd4j slot ops to Vulkan compute dispatches.
 *
 * Owned by VulkanReplayHandle (via unique_ptr). Created during capture when
 * ALL ops in a segment are Vulkan-recordable. Records each op as a
 * vkCmdDispatch into the handle's command buffer. Descriptor sets bind the
 * VkBuffers already owned by each operand's DataBuffer::special() allocation;
 * the recorder never creates host mirrors or copies every result back to host.
 *
 * Lifecycle:
 *   1. opIsRecordable(slot, inputs, numIn, outputs, numOut) — static query
 *   2. recordOp(...)                                        — per op
 *   3. prepareReplayInputs(stream)                           — sync inputs/state
 *   4. [handle->replay()]                                    — Vulkan executes
 *   5. markReplayOutputs()                                   — device side is authoritative
 *   6. destructor                                            — frees descriptor pools
 *
 * replay() is fence-complete before it returns. Host materialization therefore
 * remains the normal DataBuffer::syncToPrimary responsibility and is performed
 * only when a host consumer actually requests the data.
 *
 * Recordability is defined by the descriptor-hash handler registry. Each active
 * handler owns validation, MLIR emission, and dispatch geometry for its op.
 */
class SD_LIB_EXPORT VulkanSegmentRecorder {
 public:
  /**
   * Construct a recorder attached to the given replay handle.
   * @param handle  The VulkanReplayHandle that owns this recorder.
   *                Must outlive this recorder.
   */
  explicit VulkanSegmentRecorder(
      VulkanReplayHandle* handle,
      GraphCompilationPolicy compilationPolicy = {});
  ~VulkanSegmentRecorder();

  // Non-copyable, non-movable
  VulkanSegmentRecorder(const VulkanSegmentRecorder&) = delete;
  VulkanSegmentRecorder& operator=(const VulkanSegmentRecorder&) = delete;

  // ── Recordability gate ─────────────────────────────────────────────────────

  /**
   * Return true if the given op is Vulkan-recordable via the MLIR lowering
   * patterns in VulkanOpLowerings.  Conservative: any doubt → false.
   *
   * Supported forms are the exact contracts accepted by the descriptor-hash
   * handler registry, including their frozen arguments, shapes, and dtypes.
   *
   * @param slot      Captured op identity and immutable arguments.
   * @param inputs    Array of NDArray* input pointers.
   * @param numIn     Number of inputs.
   * @param outputs   Array of NDArray* output pointers.
   * @param numOut    Number of outputs.
   * @return true if all conditions for Vulkan dispatch are met.
   */
  static bool opIsRecordable(const NativeSlot& slot,
                             NDArray** inputs, int numIn,
                             NDArray** outputs, int numOut,
                             const VulkanDeviceCaps& caps);

  // ── Recording ──────────────────────────────────────────────────────────────

  /**
   * Record a single op dispatch into the handle's command buffer.
   *
   * Steps:
   *   1. Ensure each operand has a Vulkan DataBuffer::special() allocation and
   *      resolve its pool-owned VkBuffer.
   *   2. Emit the textual MLIR module for the op.
   *   3. Compile via pipelineCache->getOrCompile(...).
   *   4. Allocate a VkDescriptorSet, write direct device-buffer bindings, and
   *      record the dispatch.
   *   5. Record a compute barrier after the dispatch.
   *
   * @param slot    Captured op identity and immutable arguments.
   * @param inputs  Input NDArray* array.
   * @param numIn   Number of inputs.
   * @param outputs Output NDArray* array.
   * @param numOut  Number of outputs.
   * @param randomState Caller-owned RNG state for emitters carrying the generic
   *                    RANDOM_STATE contract; ignored by all other emitters.
   * @return true on success, false on any Vulkan or MLIR failure.
   *         On false the caller must destroy this recorder and abort Vulkan
   *         recording for the whole segment.
   */
  bool recordOp(const NativeSlot& slot,
                NDArray** inputs, int numIn,
                NDArray** outputs, int numOut,
                RandomGenerator* randomState);

  // ── Device actuality ──────────────────────────────────────────────────────

  /**
   * Synchronize read-before-write operands and upload caller-owned runtime
   * state on the exact replay stream. Returns false if any frozen binding no
   * longer resolves to the VkBuffer captured in the descriptor set.
   */
  bool prepareReplayInputs(VulkanExecutionStream* stream);

  /**
   * Mark every segment-written DataBuffer device allocation authoritative after
   * the replay fence completes. No eager device-to-host copy is performed.
   */
  void markReplayOutputs();

 private:
  // ── Per-operand binding record ─────────────────────────────────────────────
  struct OperandBinding {
    DataBuffer*  dataBuffer        = nullptr;
    void*        specialToken      = nullptr;
    VkBuffer     buffer            = VK_NULL_HANDLE;
    VkDeviceSize bytes             = 0;
    bool         readBeforeWrite   = false;
    bool         writtenBySegment  = false;
  };

  struct RandomStateBinding {
    RandomGenerator* hostState = nullptr;
    OperandBinding operand;
    LongType steps = 0;
    bool replayPending = false;
  };

  // ── Helpers ────────────────────────────────────────────────────────────────

  /**
   * Resolve the pool-owned Vulkan buffer backing the NDArray DataBuffer.
   * Input values are synchronized once before descriptor capture; output-only
   * values allocate device storage without materializing host contents.
   * @return true when the allocation belongs to this replay handle's device.
   */
  bool allocateBinding(NDArray* arr, bool readBeforeWrite,
                       OperandBinding& binding);

  /**
   * Emit the textual MLIR module for a captured slot and its operand NDArrays.
   * @return Non-empty string on success, empty on unsupported metadata or shape.
   */
  std::string emitMlirModule(const NativeSlot& slot,
                              NDArray** inputs, int numIn,
                              NDArray** outputs, int numOut) const;

  /**
   * Allocate a per-dispatch descriptor pool and set using descSetLayout,
   * bind all operand buffers, and record dispatch + barrier into the handle's
   * command buffer.
   */
  bool recordDispatch(VkPipeline pipeline,
                      VkPipelineLayout pipelineLayout,
                      VkDescriptorSetLayout descSetLayout,
                      const std::vector<uint32_t>& descriptorBindings,
                      const std::vector<OperandBinding*>& operands,
                      uint32_t groupCountX,
                      uint32_t groupCountY,
                      uint32_t groupCountZ);

  // ── Members ────────────────────────────────────────────────────────────────

  VulkanReplayHandle* handle_;                // owning handle (not owned by us)
  GraphCompilationPolicy compilationPolicy_;  // explicit artifact policy

  // Unique DataBuffer bindings accumulated across recordOp() calls. A graph
  // value retains one VkBuffer across producer and consumer dispatches.
  std::vector<OperandBinding> bindings_;

  // Recorder-owned storage buffers paired with caller-owned host generators.
  // The buffers remain stable for the command buffer lifetime, matching CUDA's
  // captured random-state pointer contract.
  std::vector<RandomStateBinding> randomStateBindings_;

  // One exactly-sized descriptor pool per recorded dispatch. Capture is finite,
  // and retaining the pools keeps all descriptor sets alive through replay.
  std::vector<VkDescriptorPool> descriptorPools_;
};

}  // namespace graph
}  // namespace sd

#endif  // defined(HAVE_VULKAN) && HAVE_VULKAN
#endif  // LIBND4J_VULKAN_SEGMENT_RECORDER_H

