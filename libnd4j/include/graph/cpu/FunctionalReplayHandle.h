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

#include <cstdint>
#include <vector>

namespace sd {
namespace graph {

/** Logical actions emitted by a functional replay recorder. */
enum class FunctionalReplayCommandType : uint8_t {
  EXECUTE_SLOT = 0,
  FORWARD_IDENTITY = 1,
  EXECUTE_BATCHED_GEMM = 2
};

/**
 * One immutable command in a finalized functional replay program.
 *
 * The op pointer is borrowed from the owning NativeDynamicShapePlan and is
 * checked again at replay time. A mismatch means the plan topology was rebuilt
 * and the recorded program is stale.
 */
struct FunctionalReplayCommand {
  FunctionalReplayCommandType type = FunctionalReplayCommandType::EXECUTE_SLOT;
  sd::ops::DeclarableOp* op = nullptr;  // Not owned
  int slotIndex = -1;
  int argument = -1;
};

/** Canonical operand roles tracked by EMULATED_REPLAY functional programs. */
enum class FunctionalReplayPointerRole : uint8_t {
  EXTERNAL_INPUT = 0,
  CROSS_SEGMENT_INPUT = 1,
  SEGMENT_OUTPUT = 2
};

/**
 * One late-bound operand identity observed by EMULATED_REPLAY.
 *
 * Functional programs never bake these addresses into commands. Every field is
 * still tracked so pointer churn, view-offset changes, wrapper replacement, and
 * invalid buffers are visible without incorrectly disabling late-bound replay.
 */
struct FunctionalReplayPointerBinding {
  FunctionalReplayPointerRole role = FunctionalReplayPointerRole::EXTERNAL_INPUT;
  int index = -1;
  int sourceType = -1;
  bool requiredAtEntry = false;

  const void* array = nullptr;
  const void* dataBuffer = nullptr;
  const void* primaryBuffer = nullptr;
  const void* specialBuffer = nullptr;
  const void* shapeInfo = nullptr;
  LongType offset = 0;
  LongType length = 0;
  int dataType = -1;
  bool empty = false;
  bool live = false;

  bool sameTopology(const FunctionalReplayPointerBinding& other) const;
  bool sameIdentity(const FunctionalReplayPointerBinding& other) const;
};

/** Per-execution pointer drift observed by the functional replay tracker. */
struct FunctionalReplayPointerChanges {
  int bindingCount = 0;
  int changedBindings = 0;
  int arrayChanges = 0;
  int dataBufferChanges = 0;
  int primaryBufferChanges = 0;
  int specialBufferChanges = 0;
  int shapeInfoChanges = 0;
  int offsetChanges = 0;
  int metadataChanges = 0;
  int invalidBindings = 0;
};

/**
 * Transactional pointer snapshots for functional replay.
 *
 * The tracker is deliberately independent from GraphReplayHandle's baked
 * external-address snapshot. It is invoked only by executeSegmentEmulatedReplay:
 * all valid identity changes are accepted and committed because commands resolve
 * current operands on every call. Topology drift and invalid required bindings
 * are rejected.
 */
class SD_LIB_EXPORT FunctionalReplayPointerTracker {
 public:
  Status publish(const std::vector<FunctionalReplayPointerBinding>& bindings);
  Status validateEntry(
      const std::vector<FunctionalReplayPointerBinding>& bindings) const;
  Status commit(const std::vector<FunctionalReplayPointerBinding>& bindings);
  void clear();

  bool hasSnapshot() const { return snapshotPublished_; }
  const std::vector<FunctionalReplayPointerBinding>& capturedBindings() const {
    return capturedBindings_;
  }
  const std::vector<FunctionalReplayPointerBinding>& currentBindings() const {
    return currentBindings_;
  }
  const FunctionalReplayPointerChanges& lastChanges() const {
    return lastChanges_;
  }
  long long comparisonCount() const { return comparisonCount_; }
  long long totalChangedBindings() const { return totalChangedBindings_; }

 private:
  static Status validateCanonical(
      const std::vector<FunctionalReplayPointerBinding>& bindings,
      bool requireAllLive);
  Status validateTopology(
      const std::vector<FunctionalReplayPointerBinding>& bindings) const;
  static FunctionalReplayPointerChanges compare(
      const std::vector<FunctionalReplayPointerBinding>& previous,
      const std::vector<FunctionalReplayPointerBinding>& current);

  std::vector<FunctionalReplayPointerBinding> capturedBindings_;
  std::vector<FunctionalReplayPointerBinding> currentBindings_;
  FunctionalReplayPointerChanges lastChanges_;
  bool snapshotPublished_ = false;
  long long comparisonCount_ = 0;
  long long totalChangedBindings_ = 0;
};

/**
 * Late-bound execution contract used by the functional replayer.
 *
 * Functional replay records logical plan commands rather than platform stream
 * commands, so replay needs the current inputs, outputs, and stream from its
 * caller. userData owns that invocation state; execute runs one command.
 * Both must remain valid for the synchronous duration of replayWithContext().
 */
struct FunctionalReplayExecutionContext {
  void* userData = nullptr;
  Status (*execute)(void* userData, const FunctionalReplayCommand& command) = nullptr;
};

/** Immutable command sequence published after a successful capture. */
class SD_LIB_EXPORT FunctionalReplayProgram {
 public:
  int size() const { return static_cast<int>(commands_.size()); }
  bool empty() const { return commands_.empty(); }
  bool isFinalized() const { return finalized_; }
  const std::vector<FunctionalReplayCommand>& commands() const { return commands_; }

 private:
  friend class FunctionalReplayRecorder;
  friend class FunctionalReplayHandle;

  std::vector<FunctionalReplayCommand> commands_;
  bool finalized_ = false;
};

/**
 * Transactional recorder for functional replay programs.
 *
 * Commands are accumulated privately and are copied into a captured program
 * only when endCapture succeeds. A failed recording therefore cannot publish a
 * partial program.
 */
class SD_LIB_EXPORT FunctionalReplayRecorder {
 public:
  bool beginCapture();
  bool record(const FunctionalReplayCommand& command);
  bool endCapture(FunctionalReplayProgram* capturedProgram);
  void abortCapture();

  bool isCapturing() const { return capturing_; }
  bool hasError() const { return error_; }
  int pendingCommandCount() const { return static_cast<int>(pendingCommands_.size()); }

 private:
  std::vector<FunctionalReplayCommand> pendingCommands_;
  bool capturing_ = false;
  bool error_ = false;
  int lastSlotIndex_ = -1;
};

/** Executes a finalized functional replay program in recorded order. */
class SD_LIB_EXPORT FunctionalReplayer {
 public:
  Status replay(const FunctionalReplayProgram& program,
                const FunctionalReplayExecutionContext& context,
                double* elapsedMilliseconds) const;
};

/**
 * Software implementation of GraphReplayHandle.
 *
 * Capture builds an immutable logical command program. Replay executes that
 * program through a late-bound FunctionalReplayExecutionContext, allowing the
 * current input arrays and platform stream to be supplied on every invocation.
 * Statistics are committed only after the entire program succeeds.
 */
class SD_LIB_EXPORT FunctionalReplayHandle : public GraphReplayHandle {
 public:
  FunctionalReplayHandle();
  ~FunctionalReplayHandle() override;

  FunctionalReplayHandle(const FunctionalReplayHandle&) = delete;
  FunctionalReplayHandle& operator=(const FunctionalReplayHandle&) = delete;

  // GraphReplayHandle interface. The generic replay(stream) entry point
  // cannot supply the late-bound plan invocation required by functional replay,
  // so it safely returns false; callers must use replayWithContext().
  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "Functional"; }

  // Typed replay entry point used by plan executors.
  Status replayWithContext(const FunctionalReplayExecutionContext& context);

  // Complete operand tracking used exclusively by the EMULATED_REPLAY caller.
  // Merely constructing/using a FunctionalReplayHandle does not activate it.
  Status publishPointerSnapshot(
      const std::vector<FunctionalReplayPointerBinding>& bindings);
  Status validatePointerSnapshotForReplay(
      const std::vector<FunctionalReplayPointerBinding>& bindings) const;
  Status commitPointerSnapshot(
      const std::vector<FunctionalReplayPointerBinding>& bindings);
  bool hasPointerSnapshot() const { return pointerTracker_.hasSnapshot(); }
  const FunctionalReplayPointerChanges& getLastPointerChanges() const {
    return pointerTracker_.lastChanges();
  }
  long long getPointerComparisonCount() const {
    return pointerTracker_.comparisonCount();
  }
  long long getTotalChangedPointerBindings() const {
    return pointerTracker_.totalChangedBindings();
  }

  // Recording helpers. All return false if capture is not active or if the
  // command would make the program invalid.
  bool recordCommand(FunctionalReplayCommandType type,
                     sd::ops::DeclarableOp* op,
                     int slotIndex,
                     int argument = -1);
  bool recordOp(sd::ops::DeclarableOp* op, int slotIndex);
  bool recordIdentity(sd::ops::DeclarableOp* op, int slotIndex);
  bool recordBatchedGemm(sd::ops::DeclarableOp* op, int slotIndex, int groupIndex);

  // Cancel an in-progress/failed recapture. If a prior finalized program
  // existed, it becomes ready again.
  void abortCapture();

  int getRecordedOpCount() const;
  bool hasReplayProgram() const { return state_ == ReplayState::READY && program_.isFinalized(); }
  const FunctionalReplayProgram& getProgram() const { return program_; }

  // Compatibility accessors for older callers. This list contains only
  // EXECUTE_SLOT commands and is not a complete semantic program. New code
  // must invoke replayWithContext().
  const std::vector<int>& getExecutableSlotIndices() const { return executableSlotIndices_; }
  bool hasExecutableSlotIndices() const { return !executableSlotIndices_.empty(); }

 private:
  ReplayState state_;
  FunctionalReplayRecorder recorder_;
  FunctionalReplayProgram capturedProgram_;
  FunctionalReplayProgram program_;
  FunctionalReplayer replayer_;
  FunctionalReplayPointerTracker pointerTracker_;
  std::vector<int> executableSlotIndices_;
  bool hadReadyProgramBeforeCapture_;
  long long captureStartNanos_;
  int replayCount_;
  double pendingCaptureTimeMs_;
  double captureTimeMs_;
  double lastReplayTimeMs_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FUNCTIONAL_REPLAY_HANDLE_H
