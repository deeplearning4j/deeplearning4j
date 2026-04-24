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

#ifndef LIBND4J_FUNCTIONAL_REPLAY_HANDLE_H
#define LIBND4J_FUNCTIONAL_REPLAY_HANDLE_H

#include <graph/GraphReplayHandle.h>
#include <ops/declarable/DeclarableOp.h>

#include <vector>

namespace sd {
namespace graph {

/**
 * CPU implementation of GraphReplayHandle.
 *
 * Records the op execution order and resolved op pointers during the "capture"
 * (warmup) pass. On subsequent "replay" passes:
 *   - Skips shape inference (uses cached shapes from capture)
 *   - Skips op dispatch lookup (direct function pointer call)
 *   - Still executes the actual compute (no hardware command stream)
 *
 * This eliminates ~100-200us/step of shape inference + dispatch overhead
 * for large models (3840+ ops), which is the CPU equivalent of CUDA graph
 * replay's kernel launch overhead elimination.
 *
 * Note: This is a lightweight caching mechanism, not a true hardware
 * command replay. The actual computation still runs on each replay.
 */
class SD_LIB_EXPORT FunctionalReplayHandle : public GraphReplayHandle {
 public:
  FunctionalReplayHandle();
  ~FunctionalReplayHandle() override;

  // Non-copyable
  FunctionalReplayHandle(const FunctionalReplayHandle&) = delete;
  FunctionalReplayHandle& operator=(const FunctionalReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "CPU"; }

  // ── CPU-specific: op recording ────────────────────────────────────────

  /**
   * Record an op execution during the capture pass.
   * Called by the segment executor for each op in the segment.
   *
   * @param op  Resolved op pointer (not owned)
   * @param slotIndex  The slot index of this op in the plan
   */
  void recordOp(sd::ops::DeclarableOp* op, int slotIndex);

  /** Get the number of recorded ops. */
  int getRecordedOpCount() const { return static_cast<int>(recordedOps_.size()); }

  /** Get the recorded slot indices that actually need execution.
   *  These are the non-trivial slots (not frozen constants, not fused chain
   *  tails, not identity ops). Used by executeSegmentSlotBySlot to skip
   *  iteration over trivially skippable slots. */
  const std::vector<int>& getExecutableSlotIndices() const { return executableSlotIndices_; }

  /** Whether the executable slot index list has been built. */
  bool hasExecutableSlotIndices() const { return !executableSlotIndices_.empty(); }

 private:
  /** Recorded op entry from the capture pass. */
  struct RecordedOp {
    sd::ops::DeclarableOp* op;  // Not owned
    int slotIndex;
  };

  ReplayState state_;
  std::vector<RecordedOp> recordedOps_;
  /** Slot indices that were actually executed (not skipped) during capture.
   *  Built during finalize() from the recordedOps_ list. */
  std::vector<int> executableSlotIndices_;
  int replayCount_;
  double lastReplayTimeMs_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FUNCTIONAL_REPLAY_HANDLE_H
