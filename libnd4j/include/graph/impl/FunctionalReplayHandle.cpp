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

#include <graph/cpu/FunctionalReplayHandle.h>

#include <chrono>

namespace sd {
namespace graph {
namespace {

long long functionalReplayNowNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

bool FunctionalReplayPointerBinding::sameTopology(
    const FunctionalReplayPointerBinding& other) const {
  return role == other.role && index == other.index &&
         sourceType == other.sourceType &&
         requiredAtEntry == other.requiredAtEntry;
}

bool FunctionalReplayPointerBinding::sameIdentity(
    const FunctionalReplayPointerBinding& other) const {
  return array == other.array &&
         dataBuffer == other.dataBuffer &&
         primaryBuffer == other.primaryBuffer &&
         specialBuffer == other.specialBuffer &&
         shapeInfo == other.shapeInfo &&
         offset == other.offset &&
         length == other.length &&
         dataType == other.dataType &&
         empty == other.empty &&
         live == other.live;
}

Status FunctionalReplayPointerTracker::validateCanonical(
    const std::vector<FunctionalReplayPointerBinding>& bindings,
    bool requireAllLive) {
  for (size_t i = 0; i < bindings.size(); i++) {
    const auto& binding = bindings[i];
    bool validRole =
        binding.role == FunctionalReplayPointerRole::EXTERNAL_INPUT ||
        binding.role == FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT ||
        binding.role == FunctionalReplayPointerRole::SEGMENT_OUTPUT;
    bool entryRole =
        binding.role == FunctionalReplayPointerRole::EXTERNAL_INPUT ||
        binding.role == FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT;
    if (!validRole || binding.index < 0 ||
        binding.requiredAtEntry != entryRole) {
      return Status::BAD_GRAPH;
    }

    bool hasLiveIdentity =
        binding.array != nullptr && binding.shapeInfo != nullptr &&
        (binding.empty ||
         (binding.dataBuffer != nullptr &&
          (binding.primaryBuffer != nullptr ||
           binding.specialBuffer != nullptr)));
    if (binding.live && !hasLiveIdentity) return Status::BAD_INPUT;
    if ((requireAllLive || binding.requiredAtEntry) && !binding.live) {
      return Status::BAD_INPUT;
    }
    if (i == 0) continue;

    const auto& previous = bindings[i - 1];
    int previousRole = static_cast<int>(previous.role);
    int currentRole = static_cast<int>(binding.role);
    if (currentRole < previousRole ||
        (currentRole == previousRole && binding.index <= previous.index)) {
      return Status::BAD_GRAPH;
    }
  }
  return Status::OK;
}

Status FunctionalReplayPointerTracker::validateTopology(
    const std::vector<FunctionalReplayPointerBinding>& bindings) const {
  if (!snapshotPublished_) return Status::BAD_INPUT;
  if (bindings.size() != capturedBindings_.size()) return Status::BAD_GRAPH;
  for (size_t i = 0; i < bindings.size(); i++) {
    if (!bindings[i].sameTopology(capturedBindings_[i])) {
      return Status::BAD_GRAPH;
    }
  }
  return Status::OK;
}

FunctionalReplayPointerChanges FunctionalReplayPointerTracker::compare(
    const std::vector<FunctionalReplayPointerBinding>& previous,
    const std::vector<FunctionalReplayPointerBinding>& current) {
  FunctionalReplayPointerChanges changes;
  changes.bindingCount = static_cast<int>(current.size());
  for (size_t i = 0; i < current.size(); i++) {
    const auto& before = previous[i];
    const auto& after = current[i];
    if (!after.live) changes.invalidBindings++;
    if (before.sameIdentity(after)) continue;

    changes.changedBindings++;
    if (before.array != after.array) changes.arrayChanges++;
    if (before.dataBuffer != after.dataBuffer) changes.dataBufferChanges++;
    if (before.primaryBuffer != after.primaryBuffer) {
      changes.primaryBufferChanges++;
    }
    if (before.specialBuffer != after.specialBuffer) {
      changes.specialBufferChanges++;
    }
    if (before.shapeInfo != after.shapeInfo) changes.shapeInfoChanges++;
    if (before.offset != after.offset) changes.offsetChanges++;
    if (before.length != after.length ||
        before.dataType != after.dataType ||
        before.empty != after.empty ||
        before.live != after.live) {
      changes.metadataChanges++;
    }
  }
  return changes;
}

Status FunctionalReplayPointerTracker::publish(
    const std::vector<FunctionalReplayPointerBinding>& bindings) {
  Status status = validateCanonical(bindings, true);
  if (status != Status::OK) return status;

  capturedBindings_ = bindings;
  currentBindings_ = bindings;
  lastChanges_ = FunctionalReplayPointerChanges();
  lastChanges_.bindingCount = static_cast<int>(bindings.size());
  snapshotPublished_ = true;
  comparisonCount_ = 0;
  totalChangedBindings_ = 0;
  return Status::OK;
}

Status FunctionalReplayPointerTracker::validateEntry(
    const std::vector<FunctionalReplayPointerBinding>& bindings) const {
  Status status = validateCanonical(bindings, false);
  if (status != Status::OK) return status;
  return validateTopology(bindings);
}

Status FunctionalReplayPointerTracker::commit(
    const std::vector<FunctionalReplayPointerBinding>& bindings) {
  Status status = validateCanonical(bindings, true);
  if (status != Status::OK) return status;
  status = validateTopology(bindings);
  if (status != Status::OK) return status;

  lastChanges_ = compare(currentBindings_, bindings);
  currentBindings_ = bindings;
  comparisonCount_++;
  totalChangedBindings_ += lastChanges_.changedBindings;
  return Status::OK;
}

void FunctionalReplayPointerTracker::clear() {
  capturedBindings_.clear();
  currentBindings_.clear();
  lastChanges_ = FunctionalReplayPointerChanges();
  snapshotPublished_ = false;
  comparisonCount_ = 0;
  totalChangedBindings_ = 0;
}

bool FunctionalReplayRecorder::beginCapture() {
  if (capturing_) return false;

  pendingCommands_.clear();
  capturing_ = true;
  error_ = false;
  lastSlotIndex_ = -1;
  return true;
}

bool FunctionalReplayRecorder::record(const FunctionalReplayCommand& command) {
  if (!capturing_ || error_) return false;

  bool validType = command.type == FunctionalReplayCommandType::EXECUTE_SLOT ||
                   command.type == FunctionalReplayCommandType::FORWARD_IDENTITY ||
                   command.type == FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM;
  bool validArgument = command.type != FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM ||
                       command.argument >= 0;
  if (!validType || !validArgument || command.op == nullptr ||
      command.slotIndex < 0 || command.slotIndex <= lastSlotIndex_) {
    error_ = true;
    return false;
  }

  pendingCommands_.push_back(command);
  lastSlotIndex_ = command.slotIndex;
  return true;
}

bool FunctionalReplayRecorder::endCapture(FunctionalReplayProgram* capturedProgram) {
  if (!capturing_ || error_ || capturedProgram == nullptr) {
    capturing_ = false;
    return false;
  }

  capturedProgram->commands_ = pendingCommands_;
  capturedProgram->finalized_ = false;
  capturing_ = false;
  return true;
}

void FunctionalReplayRecorder::abortCapture() {
  pendingCommands_.clear();
  capturing_ = false;
  error_ = false;
  lastSlotIndex_ = -1;
}

Status FunctionalReplayer::replay(
    const FunctionalReplayProgram& program,
    const FunctionalReplayExecutionContext& context,
    double* elapsedMilliseconds) const {
  if (elapsedMilliseconds != nullptr) *elapsedMilliseconds = 0.0;
  if (!program.isFinalized()) return Status::BAD_INPUT;
  if (!program.empty() && context.execute == nullptr) return Status::BAD_INPUT;

  auto start = std::chrono::steady_clock::now();
  for (const auto& command : program.commands()) {
    Status status = context.execute(context.userData, command);
    if (status != Status::OK) return status;
  }
  auto end = std::chrono::steady_clock::now();

  if (elapsedMilliseconds != nullptr) {
    *elapsedMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
  }
  return Status::OK;
}

FunctionalReplayHandle::FunctionalReplayHandle()
    : state_(ReplayState::EMPTY),
      hadReadyProgramBeforeCapture_(false),
      captureStartNanos_(0),
      replayCount_(0),
      pendingCaptureTimeMs_(0.0),
      captureTimeMs_(0.0),
      lastReplayTimeMs_(0.0) {}

FunctionalReplayHandle::~FunctionalReplayHandle() = default;

bool FunctionalReplayHandle::beginCapture(void* stream) {
  (void)stream;
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::READY) return false;

  hadReadyProgramBeforeCapture_ =
      state_ == ReplayState::READY && program_.isFinalized();
  capturedProgram_.commands_.clear();
  capturedProgram_.finalized_ = false;
  if (!recorder_.beginCapture()) return false;

  captureStartNanos_ = functionalReplayNowNanos();
  pendingCaptureTimeMs_ = 0.0;
  state_ = ReplayState::CAPTURING;
  return true;
}

bool FunctionalReplayHandle::endCapture(void* stream) {
  (void)stream;
  if (state_ != ReplayState::CAPTURING) return false;

  if (!recorder_.endCapture(&capturedProgram_)) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  pendingCaptureTimeMs_ = static_cast<double>(
                              functionalReplayNowNanos() - captureStartNanos_) /
                          1000000.0;
  state_ = ReplayState::CAPTURED;
  return true;
}

bool FunctionalReplayHandle::finalize() {
  if (state_ != ReplayState::CAPTURED) return false;

  int previousSlot = -1;
  for (const auto& command : capturedProgram_.commands_) {
    bool validType = command.type == FunctionalReplayCommandType::EXECUTE_SLOT ||
                     command.type == FunctionalReplayCommandType::FORWARD_IDENTITY ||
                     command.type == FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM;
    bool validArgument =
        command.type != FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM ||
        command.argument >= 0;
    if (!validType || command.op == nullptr || command.slotIndex < 0 ||
        command.slotIndex <= previousSlot || !validArgument) {
      state_ = ReplayState::ERRORED;
      return false;
    }
    previousSlot = command.slotIndex;
  }

  // Publish atomically from the completed capture. The previous ready program
  // remains untouched until every captured command has passed validation.
  program_.commands_ = capturedProgram_.commands_;
  program_.finalized_ = true;

  executableSlotIndices_.clear();
  executableSlotIndices_.reserve(program_.commands_.size());
  for (const auto& command : program_.commands_) {
    if (command.type == FunctionalReplayCommandType::EXECUTE_SLOT) {
      executableSlotIndices_.push_back(command.slotIndex);
    }
  }

  captureTimeMs_ = pendingCaptureTimeMs_;
  pendingCaptureTimeMs_ = 0.0;
  capturedProgram_.commands_.clear();
  capturedProgram_.finalized_ = false;
  hadReadyProgramBeforeCapture_ = false;
  state_ = ReplayState::READY;
  return true;
}

bool FunctionalReplayHandle::replay(void* stream) {
  // A platform stream does not contain the current NativeDynamicShapePlan
  // invocation required by logical replay. Never reinterpret it as a context.
  (void)stream;
  return false;
}

Status FunctionalReplayHandle::replayWithContext(
    const FunctionalReplayExecutionContext& context) {
  if (state_ != ReplayState::READY) return Status::BAD_INPUT;

  double elapsedMilliseconds = 0.0;
  Status status = replayer_.replay(program_, context, &elapsedMilliseconds);
  if (status == Status::OK) {
    lastReplayTimeMs_ = elapsedMilliseconds;
    replayCount_++;
  }
  return status;
}

Status FunctionalReplayHandle::publishPointerSnapshot(
    const std::vector<FunctionalReplayPointerBinding>& bindings) {
  if (state_ != ReplayState::READY || !program_.isFinalized()) {
    return Status::BAD_INPUT;
  }
  return pointerTracker_.publish(bindings);
}

Status FunctionalReplayHandle::validatePointerSnapshotForReplay(
    const std::vector<FunctionalReplayPointerBinding>& bindings) const {
  if (state_ != ReplayState::READY || !program_.isFinalized()) {
    return Status::BAD_INPUT;
  }
  return pointerTracker_.validateEntry(bindings);
}

Status FunctionalReplayHandle::commitPointerSnapshot(
    const std::vector<FunctionalReplayPointerBinding>& bindings) {
  if (state_ != ReplayState::READY || !program_.isFinalized()) {
    return Status::BAD_INPUT;
  }
  return pointerTracker_.commit(bindings);
}

ReplayState FunctionalReplayHandle::getState() const { return state_; }

ReplayStatistics FunctionalReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = program_.isFinalized() ? program_.size() : 0;
  stats.captureTimeMs = captureTimeMs_;
  stats.replayCount = replayCount_;
  stats.lastReplayTimeMs = lastReplayTimeMs_;
  return stats;
}

bool FunctionalReplayHandle::recordCommand(
    FunctionalReplayCommandType type, sd::ops::DeclarableOp* op,
    int slotIndex, int argument) {
  if (state_ != ReplayState::CAPTURING) return false;

  FunctionalReplayCommand command;
  command.type = type;
  command.op = op;
  command.slotIndex = slotIndex;
  command.argument = argument;
  if (!recorder_.record(command)) {
    state_ = ReplayState::ERRORED;
    return false;
  }
  return true;
}

bool FunctionalReplayHandle::recordOp(
    sd::ops::DeclarableOp* op, int slotIndex) {
  return recordCommand(
      FunctionalReplayCommandType::EXECUTE_SLOT, op, slotIndex);
}

bool FunctionalReplayHandle::recordIdentity(
    sd::ops::DeclarableOp* op, int slotIndex) {
  return recordCommand(
      FunctionalReplayCommandType::FORWARD_IDENTITY, op, slotIndex);
}

bool FunctionalReplayHandle::recordBatchedGemm(
    sd::ops::DeclarableOp* op, int slotIndex, int groupIndex) {
  return recordCommand(
      FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM,
      op, slotIndex, groupIndex);
}

void FunctionalReplayHandle::abortCapture() {
  recorder_.abortCapture();
  capturedProgram_.commands_.clear();
  capturedProgram_.finalized_ = false;
  state_ = hadReadyProgramBeforeCapture_ && program_.isFinalized()
               ? ReplayState::READY
               : ReplayState::EMPTY;
  hadReadyProgramBeforeCapture_ = false;
  captureStartNanos_ = 0;
  pendingCaptureTimeMs_ = 0.0;
}

int FunctionalReplayHandle::getRecordedOpCount() const {
  if (state_ == ReplayState::CAPTURING || state_ == ReplayState::ERRORED) {
    return recorder_.pendingCommandCount();
  }
  if (state_ == ReplayState::CAPTURED) return capturedProgram_.size();
  return program_.isFinalized() ? program_.size() : 0;
}

}  // namespace graph
}  // namespace sd
