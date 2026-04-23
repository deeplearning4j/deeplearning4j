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

#include <graph/SegmentExecutor.h>
#include <graph/PlanTopology.h>
#include <graph/ResourceBinder.h>
#include <graph/SlotArray.h>
#include <graph/DspDiagnostics.h>
#include <graph/NativeDynamicShapePlan.h>

#include <cassert>

namespace sd {
namespace graph {

SegmentExecutor::SegmentExecutor(std::shared_ptr<const PlanTopology> topology,
                                 int segIdx,
                                 ResourceBinder* resources)
    : topology_(std::move(topology)),
      segIdx_(segIdx),
      resources_(resources) {
  assert(topology_ != nullptr);
  assert(segIdx >= 0 && segIdx < topology_->numSegments());

  DSP_DIAG(LIFECYCLE, "SegmentExecutor created: seg=%d backend=%d slots=[%d-%d]",
           segIdx,
           static_cast<int>(selectedBackend()),
           topology_->segmentDef(segIdx).startSlot,
           topology_->segmentDef(segIdx).endSlot);
}

SegmentExecutor::~SegmentExecutor() {
  if (phase_ == Phase::SEALED && resources_ != nullptr) {
    // Release GPU resources owned by this segment
    resources_->releaseSegment(segIdx_);
  }

  DSP_DIAG(LIFECYCLE, "SegmentExecutor destroyed: seg=%d phase=%s execCount=%d",
           segIdx_, phaseName(), executionCount_);
}

SelectedBackend SegmentExecutor::selectedBackend() const {
  return topology_->segmentDef(segIdx_).selectedBackend;
}

// ── Build phase operations ─────────────────────────────────────────────────

int SegmentExecutor::buildWarmup(NDArray** inputs, int numInputs, SlotArray* slots, void* stream) {
  if (phase_ != Phase::BUILDING) return -1;

  const auto& segDef = topology_->segmentDef(segIdx_);

  DSP_DIAG(EXECUTE, "buildWarmup: seg=%d slots=[%d-%d]",
           segIdx_, segDef.startSlot, segDef.endSlot);

  // During BUILDING, we execute slot-by-slot without any frozen fast-path.
  // This eliminates the need for temporary unfreeze (demoteFrozenSlotStates).
  // The actual slot-by-slot execution is delegated to the existing
  // NativeDynamicShapePlan::executeSegmentSlotBySlot() infrastructure.
  // In the final architecture (Step 6), this will be inlined here.

  executionCount_++;
  return 0;
}

int SegmentExecutor::buildCompile(NDArray** inputs, int numInputs, SlotArray* slots, void* stream) {
  if (phase_ != Phase::BUILDING) return -1;

  const auto& segDef = topology_->segmentDef(segIdx_);

  // Non-capturable segments don't need compilation
  if (!segDef.isCapturable) {
    DSP_DIAG(EXECUTE, "buildCompile: seg=%d not capturable, skipping", segIdx_);
    return 0;
  }

  DSP_DIAG(EXECUTE, "buildCompile: seg=%d backend=%d",
           segIdx_, static_cast<int>(segDef.selectedBackend));

  // The actual compilation is delegated to the backend.
  // In the final architecture (Step 6), this calls:
  //   backend->compileSegment(topology, segIdx, slots, inputs, stream)
  // For now, the NativeDynamicShapePlan handles this internally.

  executionCount_++;
  return 0;
}

int SegmentExecutor::buildCapture(NDArray** inputs, int numInputs, SlotArray* slots, void* stream) {
  if (phase_ != Phase::BUILDING) return -1;

  const auto& segDef = topology_->segmentDef(segIdx_);

  if (!segDef.isCapturable) {
    // Non-capturable segments go straight to "sealed as slot-by-slot"
    DSP_DIAG(EXECUTE, "buildCapture: seg=%d not capturable, sealing as slot-by-slot", segIdx_);
    backendName_ = "slot-by-slot";
    seal(0);
    return 0;
  }

  DSP_DIAG(EXECUTE, "buildCapture: seg=%d starting capture", segIdx_);

  // Key insight: during BUILDING phase, we simply execute all ops fully.
  // There is no "frozen fast-path to bypass" because the BUILDING executor
  // never uses it. This eliminates the need for demoteFrozenSlotStates entirely.

  // The actual capture logic is:
  //   1. Begin capture (cudaStreamBeginCapture / backend->beginCapture)
  //   2. Execute all slots in the segment (full execution, no shortcuts)
  //   3. End capture (cudaStreamEndCapture / backend->endCapture)
  //   4. Create replay handle from the captured graph
  //   5. Call seal(shapeKey) to lock the executor

  // In the final architecture, this is inlined. For now, the caller
  // (NativeDynamicShapePlan) performs the actual capture and then calls
  // setReplayHandle() + seal() on this executor.

  executionCount_++;
  return 0;
}

// ── Sealed phase operations ────────────────────────────────────────────────

int SegmentExecutor::replay(NDArray** inputs, int numInputs, SlotArray* slots, void* stream) {
  if (phase_ != Phase::SEALED) return -1;
  if (replayHandle_ == nullptr) return -2;

  // Address-key comparison: replaces argTableStable boolean.
  // Zero-cost when stable (single comparison). Self-healing when not.
  LongType currentKey = computeInputAddrKey(inputs, numInputs);
  if (currentKey != lastInputAddrKey_) {
    // Pointer drift detected — refresh arg table
    if (resources_ != nullptr) {
      resources_->refreshArgTable(segIdx_, inputs, numInputs, slots);
    }
    lastInputAddrKey_ = currentKey;
    stableReplayCount_ = 0;

    DSP_DIAG(EXECUTE, "replay: seg=%d arg_table_refresh reason=addr_drift key=%lld",
             segIdx_, (long long)currentKey);
  } else {
    stableReplayCount_++;
  }

  // Bind stream and launch replay
  auto guard = resources_ ? resources_->bindStream(segIdx_) : StreamGuard();

  // The actual replay: replayHandle_->replay(stream)
  // This launches the captured CUDA graph or executes the functional replay.
  bool replayOk = replayHandle_->replay(stream);
  if (!replayOk) {
    DSP_DIAG(EXECUTE, "replay: seg=%d FAILED", segIdx_);
    return static_cast<int>(Status::KERNEL_FAILURE);
  }

  executionCount_++;
  return 0;
}

// ── Lifecycle transitions ──────────────────────────────────────────────────

void SegmentExecutor::seal(LongType shapeKey) {
  assert(phase_ == Phase::BUILDING && "seal() called on non-BUILDING executor");

  phase_ = Phase::SEALED;
  sealedShapeKey_ = shapeKey;

  DSP_DIAG(LIFECYCLE, "SegmentExecutor SEALED: seg=%d shapeKey=%lld addrKey=%lld backend=%s",
           segIdx_, (long long)sealedShapeKey_, (long long)lastInputAddrKey_,
           backendName_.empty() ? "none" : backendName_.c_str());
}

void SegmentExecutor::markFailed(const std::string& reason) {
  assert(phase_ == Phase::BUILDING && "markFailed() called on non-BUILDING executor");

  phase_ = Phase::FAILED;
  failureReason_ = reason;

  // Release any partially-allocated resources
  if (resources_ != nullptr) {
    resources_->releaseSegment(segIdx_);
  }

  // Release replay handle if partially created
  replayHandle_.reset();

  DSP_DIAG(LIFECYCLE, "SegmentExecutor FAILED: seg=%d reason=%s", segIdx_, reason.c_str());
}

// ── Shape drift detection ──────────────────────────────────────────────────

bool SegmentExecutor::shapesMatch(LongType currentShapeKey) const {
  if (phase_ != Phase::SEALED) return true;  // Only relevant for sealed executors
  if (sealedShapeKey_ == 0) return true;      // No key recorded (slot-by-slot sealed)
  return currentShapeKey == sealedShapeKey_;
}

// ── Address key computation ──────────────────────────────────────────────

LongType SegmentExecutor::computeInputAddrKey(NDArray** inputs, int numInputs) {
  if (inputs == nullptr || numInputs == 0) return 0;

  // FNV-1a hash of input buffer addresses
  uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
  for (int i = 0; i < numInputs; i++) {
    uint64_t addr = 0;
    if (inputs[i] != nullptr) {
#ifdef SD_CUDA
      addr = reinterpret_cast<uint64_t>(inputs[i]->specialBuffer());
#else
      addr = reinterpret_cast<uint64_t>(inputs[i]->buffer());
#endif
    }
    hash ^= addr;
    hash *= 1099511628211ULL;  // FNV prime
  }
  return static_cast<LongType>(hash);
}

// ── Replay handle ────────────────────────────────────────────────────────

void SegmentExecutor::setReplayHandle(std::unique_ptr<GraphReplayHandle> handle) {
  assert(phase_ == Phase::BUILDING && "setReplayHandle() called on non-BUILDING executor");
  replayHandle_ = std::move(handle);
}

// ── Diagnostics ──────────────────────────────────────────────────────────

int SegmentExecutor::getPhaseCode() const {
  switch (phase_) {
    case Phase::BUILDING: return executionCount_ == 0 ? 0 : 1;  // WARMUP or COMPILING
    case Phase::SEALED: return 3;   // REPLAYING
    case Phase::FAILED: return 4;   // SLOT_BY_SLOT
    default: return -1;
  }
}

const char* SegmentExecutor::phaseName() const {
  switch (phase_) {
    case Phase::BUILDING: return "BUILDING";
    case Phase::SEALED: return "SEALED";
    case Phase::FAILED: return "FAILED";
    default: return "UNKNOWN";
  }
}

void SegmentExecutor::emitReport() const {
  const auto& segDef = topology_->segmentDef(segIdx_);

  DSP_DIAG(LIFECYCLE, "  seg[%d: %d-%d] phase=%s backend=%s shapeKey=%lld addrKey=%lld "
                      "execCount=%d stableReplays=%d",
           segIdx_, segDef.startSlot, segDef.endSlot,
           phaseName(),
           backendName_.empty() ? "none" : backendName_.c_str(),
           (long long)sealedShapeKey_,
           (long long)lastInputAddrKey_,
           executionCount_,
           stableReplayCount_);
}

}  // namespace graph
}  // namespace sd
