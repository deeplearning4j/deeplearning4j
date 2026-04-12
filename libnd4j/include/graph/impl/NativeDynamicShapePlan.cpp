/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <system/op_boilerplate.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/DspPhaseUtils.h>
#include <sstream>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>

// Portable buffer accessor for DSP: specialBuffer() on CUDA, buffer() on CPU.
// CPU specialBuffer() throws when _buffer is nullptr (freed arrays) because
// CPU has no separate device buffer. Use buffer() on CPU instead.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif
// Null-safe version
#define DSP_BUF_SAFE(arr) ((arr) != nullptr ? DSP_BUF(arr) : nullptr)
#include <graph/FusionPass.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <graph/GraphBackend.h>
#include <array/DataBuffer.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/helper_hash.h>
#include <ops/OpTraitTable.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/helpers/kv_scatter.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <numeric>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <system/Environment.h>

// Include CPU graph backends conditionally
#include <config.h>
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
#endif
#if HAVE_MLIR
#include <graph/cpu/MlirCpuGraphBackend.h>
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
#include <graph/cpu/ArmHybridGraphBackend.h>
#endif
#endif
#if HAVE_NNAPI
#include <graph/cpu/NnapiGraphBackend.h>
#endif
#if HAVE_MLX
#include <graph/cpu/MlxGraphBackend.h>
#endif
// GPU graph backends are included only in the files that use them
// (_gpubackend.cpp, platform dispatch files). This file is platform-neutral.

namespace sd {
namespace graph {

namespace {
std::string normalizeOpName(const std::string& opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}

bool segmentBlocksPlanPhase(const GraphSegment& seg) {
  return seg.def.isCapturable && !seg.exec.compilationFailed;
}

bool segmentIsCompiledSteadyState(const GraphSegment& seg, int minExecutionCountExclusive) {
  if (seg.exec.currentPhase != ExecutionPhase::COMPILED) return false;
  if (seg.exec.executionCount <= minExecutionCountExclusive) return false;

  switch (seg.def.selectedBackend) {
    case SelectedBackend::CPU_GRAPH:
      return seg.resolvedCpuBackend != nullptr;
    case SelectedBackend::GPU_COMPILER:
      return !seg.exec.compiledByBackend.empty();
    default:
      return false;
  }
}

// Delegate to shared utilities in DspAnalysisUtils.h
uint32_t resolvePlanPhaseTraits(const NativeSlot& slot) {
  return dsp::resolveSlotTraits(slot);
}

int findProducerStepInSegment(const GraphSegment& seg, NativeSlot* slots, int outputSlotIdx) {
  return dsp::findProducerStepInSegment(seg, slots, outputSlotIdx);
}

bool segmentHasInternalValueShapeInputs(const GraphSegment& seg, NativeSlot* slots) {
  return dsp::segmentHasInternalValueShapeInputs(seg, slots);
}

bool segmentHasStablePointersForPlanPhase(const GraphSegment& seg, NativeSlot* slots) {
  if (!segmentBlocksPlanPhase(seg)) return true;
  const bool needsReplayInvariantTracking = segmentHasInternalValueShapeInputs(seg, slots);

  switch (seg.def.selectedBackend) {
    case SelectedBackend::EMULATED_REPLAY:
      return seg.exec.argTableStable;

    case SelectedBackend::CPU_GRAPH:
      return seg.exec.currentPhase == ExecutionPhase::REPLAYING ||
             segmentIsCompiledSteadyState(seg, 1);

    case SelectedBackend::GPU_COMPILER:
      if (seg.exec.currentPhase == ExecutionPhase::REPLAYING) {
        const bool expectsReplayHandle =
            seg.exec.replayHandle != nullptr || seg.exec.compiledByBackend == "Triton GPU";
        if (!expectsReplayHandle) return true;
        return seg.exec.replayHandle && seg.exec.replayHandle->isReady() &&
               (!needsReplayInvariantTracking || seg.exec.argTableStable);
      }
      if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) {
        return !needsReplayInvariantTracking || seg.exec.argTableStable;
      }
      return segmentIsCompiledSteadyState(seg, 1) &&
             (!needsReplayInvariantTracking || seg.exec.argTableStable);

    case SelectedBackend::CUDA_GRAPHS:
    case SelectedBackend::SLOT_BY_SLOT:
      return seg.exec.replayHandle && seg.exec.replayHandle->isReady();
  }

  return false;
}

bool segmentIsFullyReplayingForPlanPhase(const GraphSegment& seg, NativeSlot* slots) {
  if (!segmentBlocksPlanPhase(seg)) return true;
  const bool needsReplayInvariantTracking = segmentHasInternalValueShapeInputs(seg, slots);

  switch (seg.def.selectedBackend) {
    case SelectedBackend::EMULATED_REPLAY:
      return seg.exec.currentPhase == ExecutionPhase::REPLAYING &&
             seg.exec.argTableStable;

    case SelectedBackend::CPU_GRAPH:
      return seg.exec.currentPhase == ExecutionPhase::REPLAYING ||
             segmentIsCompiledSteadyState(seg, 2);

    case SelectedBackend::GPU_COMPILER:
      if (seg.exec.replayHandle) {
        return seg.exec.replayHandle && seg.exec.replayHandle->isReady() &&
               seg.exec.currentPhase == ExecutionPhase::REPLAYING &&
               (!needsReplayInvariantTracking || seg.exec.argTableStable);
      }
      if (seg.exec.currentPhase == ExecutionPhase::REPLAYING &&
          seg.exec.compiledByBackend == "Triton GPU") {
        return false;
      }
      if (seg.exec.currentPhase == ExecutionPhase::REPLAYING) {
        return !needsReplayInvariantTracking || seg.exec.argTableStable;
      }
      return segmentIsCompiledSteadyState(seg, 2) &&
             (!needsReplayInvariantTracking || seg.exec.argTableStable);

    case SelectedBackend::CUDA_GRAPHS:
    case SelectedBackend::SLOT_BY_SLOT:
      return seg.exec.replayHandle && seg.exec.replayHandle->isReady() &&
             seg.exec.currentPhase == ExecutionPhase::REPLAYING;
  }

  return false;
}
/**
 * Returns the number of "structural" iArgs for an op — these are control parameters
 * (masks, mode flags, axis indices) that are always passed via iArgs regardless of
 * whether data parameters come from input tensors or from iArgs.
 * Returns -1 if all iArgs are structural (the default for most ops).
 */
static int getStructuralIArgCount(const std::string& opName) {
    static const std::unordered_map<std::string, int> STRUCTURAL_IARGS = {
        {"strided_slice", 5},   // 5 mask bits (begin/end/shrink/new_axis/ellipsis)
        {"concat", 1},          // axis
        {"split", 1},           // num_splits
        {"split_v", 1},         // axis
        {"one_hot", 2},         // axis, depth
        {"top_k", 1},           // k
    };
    auto it = STRUCTURAL_IARGS.find(opName);
    return (it != STRUCTURAL_IARGS.end()) ? it->second : -1;
}

}  // namespace

// NativeSlot move operations removed: sub-structs manage their own memory.
// NativeSlot is now non-movable (deleted in header).

// ─── NativeDynamicShapePlan ─────────────────────────────────────────────────

NativeDynamicShapePlan::NativeDynamicShapePlan()
    : slots_(nullptr), numSlots_(0), totalOutputSlots_(0), numExternalInputs_(0),
      releaseAtStep_(nullptr), releaseAtStepCounts_(nullptr),
      requestedOutputSlotIndices_(nullptr), numRequestedOutputs_(0),
      outputSlots_(nullptr), slotIsViewProducer_(nullptr),
      contextPool_(nullptr), viewProducerDetectionDone_(false), frozenConstantDetectionDone_(false),
      gpuGraphCaptureEnabled_(false), totalGraphReplays_(0), jitMode_(JitMode::GRAPH_ONLY), graphExecutionMode_(GraphExecutionMode::GEM_AUTO),
      shapesFrozen_(false), executeCount_(0), compilationDone_(false), executionTimingEnabled_(false), traceEnabled_(false),
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false),
      gpuGraphBackend_(nullptr), gpuGraphBackendChecked_(false),
      untrackedOutputCache_(nullptr), untrackedOutputCacheSize_(0),
      kvCacheRetentionEnabled_(false), kvCachePosition_(0), kvCacheMaxLen_(0),
      kvCacheNumMappings_(0), kvCacheMappings_(nullptr),
      maxKvCacheLen_(0),
      hasControlFlow_(false), loopRegions_(nullptr), numLoopRegions_(0),
      slotIsDead_(nullptr), slotIsDeadSize_(0),
      slotOwnership_(nullptr)
      {}

void NativeDynamicShapePlan::writeOutputSlot(int slotIdx, NDArray* value, const char* tag) {
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) {
    char buf[128];
    snprintf(buf, sizeof(buf), "writeOutputSlot: index %d out of range [0, %d)", slotIdx, totalOutputSlots_);
    THROW_EXCEPTION(buf);
  }

  NDArray* old = outputSlots_[slotIdx];

  // DIAGNOSTIC: trace writes to the configured trace slot (ND4J_DSP_TRACE_SLOT)
  if (DSP_DIAG_ENABLED(MEMORY) && planPhase_ >= PlanPhase::SHAPES_FROZEN) {
    int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
    if (ts >= 0 && slotIdx == ts) {
      auto* oldDb = old != nullptr ? old->dataBuffer() : nullptr;
      auto* newDb = value != nullptr ? value->dataBuffer() : nullptr;
      DSP_DIAG(MEMORY, "WOS_%d: tag=%s old=%p new=%p oldDb=%p newDb=%p exec=%d phase=%d",
               slotIdx, tag, (void*)old, (void*)value, (void*)oldDb, (void*)newDb,
               executeCount_, (int)planPhase_);
    }
  }

  if (planPhase_ >= PlanPhase::POINTERS_STABLE && executeCount_ > 2) {
    if (old != nullptr && value != nullptr) {
      auto* oldDb = old->dataBuffer();
      auto* newDb = value->dataBuffer();
      if (oldDb != nullptr && newDb != nullptr && oldDb != newDb) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "LIFECYCLE VIOLATION: buffer replacement at slot %d (tag=%s) "
                 "after POINTERS_STABLE (execCount=%d). oldDb=%p newDb=%p",
                 slotIdx, tag, executeCount_, (void*)oldDb, (void*)newDb);
        DSP_DIAG(FALLBACK, "%s", buf);
        THROW_EXCEPTION(buf);
      }
    }
  }

  if (value != nullptr && value->dataBuffer() != nullptr && value->dataBuffer()->isClosed()) {
    char buf[128];
    snprintf(buf, sizeof(buf), "LIFECYCLE VIOLATION: writing closed DataBuffer at slot %d (tag=%s)", slotIdx, tag);
    THROW_EXCEPTION(buf);
  }

  if (value != nullptr && value != old &&
      planOwnedArrays_.count(value) == 0 &&
      value->dataBuffer() != nullptr &&
      protectedWeightBuffers_.count(value->dataBuffer()) == 0) {
    planOwnedArrays_.insert(value);
  }

  DSP_DIAG(MEMORY, "WRITE_SLOT: slot=%d tag=%s phase=%d execCount=%d",
           slotIdx, tag, static_cast<int>(planPhase_), executeCount_);

  // Free the OLD array when it's being replaced, IF:
  // 1. old != value (actually being replaced, not a no-op write)
  // 2. old is plan-owned (we allocated it, safe to delete)
  // 3. old's DataBuffer is not a protected weight buffer
  // 4. old is not still referenced by another slot
  // Without this, replaced arrays stay in planOwnedArrays_ but are unreachable
  // from outputSlots_[], causing ~240 MB/step GPU memory leak in large models.
  if (old != nullptr && old != value) {
    bool isPlanOwned = planOwnedArrays_.count(old) > 0;
    if (isPlanOwned) {
      auto* oldDb = old->dataBuffer();
      bool isProtected = oldDb != nullptr && protectedWeightBuffers_.count(oldDb) > 0;
      if (!isProtected) {
        // Check that no OTHER slot still references this exact NDArray pointer.
        // View ops can share the same NDArray across slots.
        bool referencedElsewhere = false;
        for (int i = 0; i < totalOutputSlots_; i++) {
          if (i != slotIdx && outputSlots_[i] == old) {
            referencedElsewhere = true;
            break;
          }
        }
        if (!referencedElsewhere) {
          planOwnedArrays_.erase(old);
          delete old;
        }
      }
    } else {
      // NOT plan-owned but being replaced — this is a potential leak.
      long long leakedBytes = old->dataBuffer() ? (long long)old->dataBuffer()->getLenInBytes() : 0;
      DSP_DIAG(MEMORY, "WRITE_SLOT_LEAK: slot=%d tag=%s old=%p NOT plan-owned, bytes=%lld planOwned=%d",
               slotIdx, tag, (void*)old, leakedBytes, (int)planOwnedArrays_.size());
    }
  }

  outputSlots_[slotIdx] = value;
}

void NativeDynamicShapePlan::setGraphExecutionMode(GraphExecutionMode mode) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "setGraphExecutionMode");
  DSP_DIAG(EXECUTE, "setGraphExecutionMode: %d -> %d", static_cast<int>(graphExecutionMode_), static_cast<int>(mode));
  graphExecutionMode_ = mode;
  // Reset cached backends so mode changes take effect immediately.
  gpuGraphBackendChecked_ = false;
  gpuGraphBackend_ = nullptr;
  cpuGraphBackendChecked_ = false;
  cpuGraphBackend_ = nullptr;
  cpuGraphBackendChainBuilt_ = false;
  cpuGraphBackendChain_.clear();
  // Enable GPU graph capture for all modes except SLOT_BY_SLOT.
  // JIT backends (Triton/NVRTC/PTX) use graph capture when they
  // can't handle a segment (unsupported ops, etc).
  if (mode != GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    gpuGraphCaptureEnabled_ = true;
  }
  // Clear GPU backend failed-compilation cache so segments that failed with
  // incomplete shapes (e.g., attention with seqK=0 before KV setup)
  // can retry when called again with correct external input shapes.
  clearGpuBackendFailedCache();
}

NativeDynamicShapePlan::~NativeDynamicShapePlan() {
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: START plan=%p numSlots=%d totalOutputSlots=%d planOwned=%zu",
           this, numSlots_, totalOutputSlots_, planOwnedArrays_.size());

  // ── Phase 1: Free GPU resources FIRST ─────────────────────────────────
  // Platform GPU resources (replay handles, JIT kernels, cuBLAS workspace,
  // batch-zero) may hold direct references into outputSlots_. Clean them
  // BEFORE freeing slot arrays to avoid dangling pointer access during teardown.
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing platform GPU resources");
  platformFreePlanResources();

  // Free symbolic shape range profiles from all segments
  for (auto& seg : segments_) {
    if (seg.exec.symbolicRangeData != nullptr) {
      freeSegmentShapeProfile(static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData));
      seg.exec.symbolicRangeData = nullptr;
    }
  }

  // ── Phase 2: Free slot data ───────────────────────────────────────────
  // Free slots metadata
  if (slots_) {
    delete[] slots_;
  }

  // Free release schedule
  if (releaseAtStep_) {
    for (int i = 0; i < numSlots_; i++) {
      delete[] releaseAtStep_[i];
    }
    delete[] releaseAtStep_;
  }
  delete[] releaseAtStepCounts_;

  // Free requested output mapping
  delete[] requestedOutputSlotIndices_;

  // Dedup set to prevent double-free (identity ops can share pointers across slots)
  std::unordered_set<NDArray*> deleted;

  // Free slot arrays. Only delete arrays that the plan created (in planOwnedArrays_).
  // Arrays from external inputs or model variables are NOT plan-owned and must survive.
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing outputSlots_ (%d slots, %zu plan-owned)",
           totalOutputSlots_, planOwnedArrays_.size());
  if (outputSlots_) {
    int freedOwned = 0, skippedExternal = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) continue;
      if (!deleted.insert(outputSlots_[i]).second) continue;

      if (planOwnedArrays_.count(outputSlots_[i]) > 0) {
        freedOwned++;
        delete outputSlots_[i];
      } else {
        skippedExternal++;
      }
    }
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freed %d plan-owned, skipped %d external from outputSlots_",
             freedOwned, skippedExternal);
    delete[] outputSlots_;
  }
  // outputSlots_ owns the NDArray* array — do NOT delete[] separately

  // Free view producer flags
  delete[] slotIsViewProducer_;

  // Free context pool
  if (contextPool_) {
    for (int i = 0; i < numSlots_; i++) {
      delete contextPool_[i];
    }
    delete[] contextPool_;
  }

  // Free owned legacy ops (created during deserialization for ops
  // not registered in OpRegistrator, like exp, log, abs, etc.)
  for (auto* legacyOp : ownedLegacyOps_) {
    delete legacyOp;
  }

  // Free untracked output cache
  if (untrackedOutputCache_) {
    for (int i = 0; i < untrackedOutputCacheSize_; i++) {
      delete untrackedOutputCache_[i];
    }
    delete[] untrackedOutputCache_;
  }

  // Free KV cache mappings
  delete[] kvCacheMappings_;

  // Free control flow structures
  delete[] loopRegions_;
  delete[] slotIsDead_;

  // Free slot buffer ownership metadata
  delete[] slotOwnership_;

  // ── Phase 3: Release references ───────────────────────────────────────
  // Remove frozen reference counts from weight DataBuffers if the plan
  // is still in frozen state when destroyed. This allows the buffers to
  // be migrated by future plans or general device management.
  if (shapesFrozen_) {
    for (auto* db : protectedWeightBuffers_) {
      if (db != nullptr) {
        db->removeFrozenRef();
      }
    }
  }
  // Clear protected weight buffer set so stale DataBuffer pointers don't
  // linger. These are external (caller-owned) — we never freed them, but
  // holding stale pointers after plan destruction is a hazard.
  protectedWeightBuffers_.clear();

  // Free Phase 3/4 structures
  if (planDef_ != nullptr) {
    planDef_->release();
    planDef_ = nullptr;
  }
  delete execState_;
  execState_ = nullptr;

  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: DONE plan=%p", this);

  // Finalize diagnostics AFTER all cleanup so destructor logging is captured
  DspDiagnostics::getInstance().endPlanExecution();
  DspDiagnostics::getInstance().printPlanReport();
  DspDiagnostics::getInstance().flushJsonReport();
}

// ─── Deserialization from binary plan ─────────────────────────────────────────

static const uint32_t DSP_MAGIC = 0x44535031;  // "DSP1"
static const int32_t DSP_VERSION_MAX = 5;  // Max supported version

/**
 * Helper to read typed values from a byte stream.
 */
class BinaryReader {
 public:
  BinaryReader(const uint8_t* data, LongType size)
      : data_(data), size_(size), pos_(0) {}

  template <typename T>
  T read() {
    if (pos_ + sizeof(T) > static_cast<size_t>(size_)) {
      THROW_EXCEPTION("BinaryReader: read past end of buffer");
    }
    T val;
    std::memcpy(&val, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return val;
  }

  template <typename T>
  void readArray(T* dest, int count) {
    size_t bytes = count * sizeof(T);
    if (pos_ + bytes > static_cast<size_t>(size_)) {
      THROW_EXCEPTION("BinaryReader: readArray past end of buffer");
    }
    std::memcpy(dest, data_ + pos_, bytes);
    pos_ += bytes;
  }

  std::string readString() {
    int32_t len = read<int32_t>();
    if (len < 0 || pos_ + len > static_cast<size_t>(size_)) {
      THROW_EXCEPTION("BinaryReader: invalid string length");
    }
    std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return s;
  }

  size_t remaining() const { return size_ - pos_; }

 private:
  const uint8_t* data_;
  LongType size_;
  size_t pos_;
};

NativeDynamicShapePlan* NativeDynamicShapePlan::fromSerializedPlan(
    const void* data, LongType size) {
  BinaryReader reader(static_cast<const uint8_t*>(data), size);

  // Read header
  uint32_t magic = reader.read<uint32_t>();
  if (magic != DSP_MAGIC) {
    DSP_DIAG(COMPILE, "NativeDynamicShapePlan: invalid magic 0x%08x (expected 0x%08x)", magic, DSP_MAGIC);
    return nullptr;
  }

  int32_t version = reader.read<int32_t>();
  if (version < 1 || version > DSP_VERSION_MAX) {
    DSP_DIAG(COMPILE, "NativeDynamicShapePlan: unsupported version %d (expected 1-%d)", version, DSP_VERSION_MAX);
    return nullptr;
  }

  auto* plan = new NativeDynamicShapePlan();
  plan->numSlots_ = reader.read<int32_t>();
  plan->totalOutputSlots_ = reader.read<int32_t>();
  plan->numExternalInputs_ = reader.read<int32_t>();
  plan->numRequestedOutputs_ = reader.read<int32_t>();

  REQUIRE_TRUE(plan->numSlots_ > 0 && plan->numSlots_ < 100000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numSlots %d", plan->numSlots_);
  REQUIRE_TRUE(plan->totalOutputSlots_ >= plan->numSlots_ && plan->totalOutputSlots_ < 500000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid totalOutputSlots %d (numSlots=%d)",
               plan->totalOutputSlots_, plan->numSlots_);
  REQUIRE_TRUE(plan->numExternalInputs_ >= 0 && plan->numExternalInputs_ < 100000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numExternalInputs %d", plan->numExternalInputs_);
  REQUIRE_TRUE(plan->numRequestedOutputs_ >= 0 && plan->numRequestedOutputs_ <= plan->totalOutputSlots_, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numRequestedOutputs %d (totalOutputSlots=%d)",
               plan->numRequestedOutputs_, plan->totalOutputSlots_);

  // Allocate slots
  plan->slots_ = new NativeSlot[plan->numSlots_];

  // Read per-slot data
  for (int s = 0; s < plan->numSlots_; s++) {
    NativeSlot& slot = plan->slots_[s];
    slot.ident.opHash = reader.read<int64_t>();
    slot.ident.opName = reader.readString();
    slot.wiring.numInputs = reader.read<int32_t>();
    slot.wiring.numOutputs = reader.read<int32_t>();

    REQUIRE_TRUE(slot.wiring.numInputs >= 0 && slot.wiring.numInputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numInputs %d", s, slot.wiring.numInputs);
    REQUIRE_TRUE(slot.wiring.numOutputs >= 0 && slot.wiring.numOutputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numOutputs %d", s, slot.wiring.numOutputs);

    // Input wiring
    slot.wiring.inputSourceIndices = new int[slot.wiring.numInputs];
    reader.readArray(slot.wiring.inputSourceIndices, slot.wiring.numInputs);

    // Validate each inputSourceIndex is in valid range
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      REQUIRE_TRUE(slot.wiring.inputSourceIndices[i] >= -(plan->numExternalInputs_ + 1) &&
                   slot.wiring.inputSourceIndices[i] < plan->totalOutputSlots_, 0,
                   "NativeDynamicShapePlan::fromSerializedPlan: slot %d inputSourceIndices[%d]=%d out of range [%d, %d)",
                   s, i, slot.wiring.inputSourceIndices[i], -(plan->numExternalInputs_ + 1), plan->totalOutputSlots_);
    }

    slot.wiring.inputSourceTypes = new int8_t[slot.wiring.numInputs];
    reader.readArray(slot.wiring.inputSourceTypes, slot.wiring.numInputs);

    // Output wiring
    slot.wiring.outputSlotIndices = new int[slot.wiring.numOutputs];
    reader.readArray(slot.wiring.outputSlotIndices, slot.wiring.numOutputs);

    // iArgs
    slot.args.numIArgs = reader.read<int32_t>();
    if (slot.args.numIArgs > 0) {
      slot.args.iArgs = new LongType[slot.args.numIArgs];
      reader.readArray(slot.args.iArgs, slot.args.numIArgs);
    }

    // tArgs
    slot.args.numTArgs = reader.read<int32_t>();
    if (slot.args.numTArgs > 0) {
      slot.args.tArgs = new double[slot.args.numTArgs];
      reader.readArray(slot.args.tArgs, slot.args.numTArgs);
    }

    // bArgs
    slot.args.numBArgs = reader.read<int32_t>();
    if (slot.args.numBArgs > 0) {
      slot.args.bArgs = new bool[slot.args.numBArgs];
      reader.readArray(slot.args.bArgs, slot.args.numBArgs);
    }

    // dArgs
    slot.args.numDArgs = reader.read<int32_t>();
    if (slot.args.numDArgs > 0) {
      slot.args.dArgs = new DataType[slot.args.numDArgs];
      // dArgs are serialized as int32
      for (int i = 0; i < slot.args.numDArgs; i++) {
        slot.args.dArgs[i] = static_cast<DataType>(reader.read<int32_t>());
      }
    }

    slot.args.numSArgs = 0;
    if (version >= 5) {
      slot.args.numSArgs = reader.read<int32_t>();
      if (slot.args.numSArgs > 0) {
        slot.args.sArgs = new std::string[slot.args.numSArgs];
        for (int i = 0; i < slot.args.numSArgs; i++) {
          slot.args.sArgs[i] = reader.readString();
        }
      }
    }

    // Flags
    slot.flags.needsZeroedOutput = reader.read<uint8_t>() != 0;
    slot.flags.isDataDependent = reader.read<uint8_t>() != 0;
    slot.flags.outputShapeDependsOnInputValues = reader.read<uint8_t>() != 0;
    slot.flags.needsIntLongSync = reader.read<uint8_t>() != 0;
    slot.flags.isCustomOp = reader.read<uint8_t>() != 0;
    slot.targetDeviceId = reader.read<int32_t>();

    // V2: legacy op type and opNum for ops not registered as DeclarableOp
    slot.legacy.legacyOpType = 0;
    slot.legacy.legacyOpNum = -1;
    if (version >= 2) {
      slot.legacy.legacyOpType = reader.read<int32_t>();
      slot.legacy.legacyOpNum = reader.read<int32_t>();
    }

    // V3: control flow metadata
    slot.cf.controlFlowType = CF_NONE;
    slot.cf.loopBackTarget = -1;
    slot.cf.loopRegionIndex = -1;
    if (version >= 3) {
      slot.cf.controlFlowType = static_cast<ControlFlowType>(reader.read<uint8_t>());
      slot.cf.loopBackTarget = reader.read<int32_t>();
      slot.cf.loopRegionIndex = reader.read<int32_t>();
    }

    // Resolve op by name (Java and C++ use different hash functions,
    // so we look up by name string and compute the C++ hash from it)
    slot.ident.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.ident.opName);
    if (!slot.ident.op && slot.legacy.legacyOpType > 0 && slot.legacy.legacyOpNum >= 0) {
      // Create a legacy op wrapper for ops not in the OpRegistrator
      // (e.g., exp, log, abs, neg, sqrt, sin, cos, etc.)
      sd::ops::DeclarableOp* legacyOp = nullptr;
      switch (slot.legacy.legacyOpType) {
        case 1:  // LegacyTransformSameOp
          legacyOp = new sd::ops::LegacyTransformSameOp(slot.legacy.legacyOpNum);
          break;
        case 2:  // LegacyTransformStrictOp
          legacyOp = new sd::ops::LegacyTransformStrictOp(slot.legacy.legacyOpNum);
          break;
        case 3:  // LegacyTransformFloatOp
          legacyOp = new sd::ops::LegacyTransformFloatOp(slot.legacy.legacyOpNum);
          break;
        case 4:  // LegacyTransformBoolOp
          legacyOp = new sd::ops::LegacyTransformBoolOp(slot.legacy.legacyOpNum);
          break;
        case 5:  // LegacyScalarOp
          legacyOp = new sd::ops::LegacyScalarOp(slot.legacy.legacyOpNum);
          break;
        case 6:  // LegacyPairwiseTransformOp
          legacyOp = new sd::ops::LegacyPairwiseTransformOp(slot.legacy.legacyOpNum);
          break;
        case 7:  // LegacyScalarBoolOp
          legacyOp = new sd::ops::LegacyScalarBoolOp(slot.legacy.legacyOpNum);
          break;
        default:
          DSP_DIAG(COMPILE, "unknown legacy op type %d for '%s'",
                    slot.legacy.legacyOpType, slot.ident.opName.c_str());
          break;
      }
      if (legacyOp) {
        plan->ownedLegacyOps_.push_back(legacyOp);
        slot.ident.op = legacyOp;
        sd_debug("NativeDynamicShapePlan: created legacy op type=%d num=%d for '%s'\n",
                 slot.legacy.legacyOpType, slot.legacy.legacyOpNum, slot.ident.opName.c_str());
      }
    }
    if (!slot.ident.op && slot.cf.controlFlowType != CF_NONE) {
      // Control flow ops dont need a DeclarableOp — dispatched by CF engine
      sd_debug("NativeDynamicShapePlan: CF op '%s' (type=%d) — no DeclarableOp needed\n",
               slot.ident.opName.c_str(), static_cast<int>(slot.cf.controlFlowType));
    } else if (!slot.ident.op) {
      DSP_DIAG(COMPILE, "NativeDynamicShapePlan: op not found for name '%s' (serialized hash: %lld, legacyType: %d, legacyNum: %d)",
                slot.ident.opName.c_str(), slot.ident.opHash, slot.legacy.legacyOpType, slot.legacy.legacyOpNum);
      delete plan;
      return nullptr;
    }


    // Use the C++ hash for internal computations (shape key, etc.)
    slot.ident.opHash = sd::ops::HashHelper::getInstance().getLongHash(slot.ident.opName);
    // Structural replay/capture classification must come from op traits, not
    // hardcoded op-name lists.
    uint32_t opTraits = 0;
    if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
      opTraits = slot.ident.op->getOpDescriptor()->getTraits();
    }
    // Fallback: look up traits by op name from the trait table.
    if (opTraits == 0 && !slot.ident.opName.empty()) {
      opTraits = sd::ops::getOpTraitsByName(slot.ident.opName);
    }
    slot.flags.isIdentityOp = (opTraits & sd::ops::OP_TRAIT_IDENTITY) != 0;
    slot.flags.isViewCapableOp = (opTraits & sd::ops::OP_TRAIT_VIEW_PRODUCING) != 0;
    // View-capable ops share input buffer → no zeroing needed
    if (slot.flags.isViewCapableOp) slot.flags.needsZeroedOutput = false;

    // Set structural iArg count from table (consistent with NativePlanCompiler)
    slot.flags.structuralIArgCount = getStructuralIArgCount(normalizeOpName(slot.ident.opName));

    // Initialize fusion fields (will be set by FusionPass::applyFusions later)
    slot.flags.inPlaceFused = false;
    slot.flags.inPlaceFusedInputIdx = -1;
    slot.fusedChain.isFusedChainHead = false;
    slot.fusedChain.fusedChainLength = 0;
    slot.fusedChain.isFusedChainTail = false;
    std::memset(slot.fusedChain.fusedChainOpCodes, 0, sizeof(slot.fusedChain.fusedChainOpCodes));
    std::memset(slot.fusedChain.fusedChainSlots, 0, sizeof(slot.fusedChain.fusedChainSlots));
    std::fill(std::begin(slot.fusedChain.fusedChainSecondaryInputSources), std::end(slot.fusedChain.fusedChainSecondaryInputSources), INT32_MIN);
  }

  // Read release schedule
  plan->releaseAtStep_ = new int*[plan->numSlots_];
  plan->releaseAtStepCounts_ = new int[plan->numSlots_];
  for (int s = 0; s < plan->numSlots_; s++) {
    int count = reader.read<int32_t>();
    plan->releaseAtStepCounts_[s] = count;
    if (count > 0) {
      plan->releaseAtStep_[s] = new int[count];
      reader.readArray(plan->releaseAtStep_[s], count);
    } else {
      plan->releaseAtStep_[s] = nullptr;
    }
  }

  // V3: Read loop regions
  plan->loopRegions_ = nullptr;
  plan->numLoopRegions_ = 0;
  plan->hasControlFlow_ = false;
  if (version >= 3) {
    plan->numLoopRegions_ = reader.read<int32_t>();
    if (plan->numLoopRegions_ > 0) {
      plan->loopRegions_ = new LoopRegion[plan->numLoopRegions_];
      for (int i = 0; i < plan->numLoopRegions_; i++) {
        plan->loopRegions_[i].mergeSlot = reader.read<int32_t>();
        plan->loopRegions_[i].switchSlot = reader.read<int32_t>();
        plan->loopRegions_[i].nextIterSlot = reader.read<int32_t>();
        plan->loopRegions_[i].exitSlot = reader.read<int32_t>();
        plan->loopRegions_[i].bodyStartSlot = reader.read<int32_t>();
        plan->loopRegions_[i].bodyEndSlot = reader.read<int32_t>();
      }
    }
    // Check if any slot has control flow
    for (int s = 0; s < plan->numSlots_; s++) {
      if (plan->slots_[s].cf.controlFlowType != CF_NONE) {
        plan->hasControlFlow_ = true;
        break;
      }
    }
    if (plan->hasControlFlow_) {
      DSP_DIAG(COMPILE, "control flow detected (%d loop regions)",
               plan->numLoopRegions_);
    }
  }

  // Allocate dead-slot tracking for control flow
  plan->slotIsDeadSize_ = plan->totalOutputSlots_;
  plan->slotIsDead_ = new bool[plan->slotIsDeadSize_];
  std::memset(plan->slotIsDead_, 0, sizeof(bool) * plan->slotIsDeadSize_);

  // Read requested output slot indices
  plan->requestedOutputSlotIndices_ = new int[plan->numRequestedOutputs_];
  reader.readArray(plan->requestedOutputSlotIndices_, plan->numRequestedOutputs_);

  // Read external input names (v4+)
  if (version >= 4) {
    plan->externalInputNames_.resize(plan->numExternalInputs_);
    for (int i = 0; i < plan->numExternalInputs_; i++) {
      plan->externalInputNames_[i] = reader.readString();
    }
  }

  // Build externalInputIsVariable_ by scanning slot input source types.
  // Only PLACEHOLDER external inputs need forced H2D sync before CUDA graph replay.
  // SOURCE_VARIABLE (model weights) should NOT be force-synced — their device buffers
  // are authoritative after initial model load and never change during inference.
  plan->externalInputIsVariable_.resize(plan->numExternalInputs_, false);
  for (int s = 0; s < plan->numSlots_; s++) {
    auto& slot = plan->slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < plan->numExternalInputs_ &&
            slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
          plan->externalInputIsVariable_[extIdx] = true;
        }
      }
    }
  }

  // Allocate execution state
  plan->outputSlots_ = new NDArray*[plan->totalOutputSlots_];
  std::memset(plan->outputSlots_, 0, sizeof(NDArray*) * plan->totalOutputSlots_);

  // outputSlots_ owns all slot arrays

  plan->slotIsViewProducer_ = new bool[plan->totalOutputSlots_];
  std::memset(plan->slotIsViewProducer_, 0, sizeof(bool) * plan->totalOutputSlots_);

  // Allocate slot buffer ownership metadata (value-initialized to UNSET)
  plan->slotOwnership_ = new SlotBufferInfo[plan->totalOutputSlots_]();

  // Allocate untracked output cache (for outputs with outputSlotIndices[i] < 0).
  // These are temporary buffers needed by ops but not referenced downstream.
  // Cached here so they can be reused during GPU graph capture (where allocs fail).
  plan->untrackedOutputCacheSize_ = plan->numSlots_ * MAX_OUTPUTS_PER_SLOT;
  plan->untrackedOutputCache_ = new NDArray*[plan->untrackedOutputCacheSize_];
  std::memset(plan->untrackedOutputCache_, 0, sizeof(NDArray*) * plan->untrackedOutputCacheSize_);

  // Pre-allocate context pool
  plan->contextPool_ = new Context*[plan->numSlots_];
  for (int i = 0; i < plan->numSlots_; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // ── Shape static analysis: classify each slot as shape-static or shape-dynamic ──
  // A slot is shape-dynamic if it transitively depends on any placeholder input
  // or is data-dependent. Everything else is shape-static (constants/variables
  // never change shape between executions).
  // Slots are in topological order, so predecessors are already classified.
  {
    // Build reverse mapping: outputSlotIndex -> stepIndex (which slot produced it)
    std::vector<int> outputSlotToStepIndex(plan->totalOutputSlots_, -1);
    for (int s = 0; s < plan->numSlots_; s++) {
      NativeSlot& slot = plan->slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < plan->totalOutputSlots_) {
          outputSlotToStepIndex[si] = s;
        }
      }
    }

    int staticCount = 0, dynamicCount = 0;
    for (int s = 0; s < plan->numSlots_; s++) {
      NativeSlot& slot = plan->slots_[s];
      slot.shapeCache.shapeStatic = true;  // assume static

      // Data-dependent ops always dynamic (output shape depends on runtime values)
      if (slot.flags.isDataDependent || slot.flags.outputShapeDependsOnInputValues) {
        slot.shapeCache.shapeStatic = false;
        dynamicCount++;
        continue;
      }

      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          // External input: placeholders are dynamic, constants/variables are static
          if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            slot.shapeCache.shapeStatic = false;
            break;
          }
        } else {
          // From prior slot output — check if producer is dynamic
          if (srcIdx < plan->totalOutputSlots_) {
            int producerStep = outputSlotToStepIndex[srcIdx];
            if (producerStep >= 0 && !plan->slots_[producerStep].shapeCache.shapeStatic) {
              slot.shapeCache.shapeStatic = false;
              break;
            }
          }
        }
      }

      if (slot.shapeCache.shapeStatic) staticCount++;
      else dynamicCount++;
    }

    DSP_DIAG(SHAPE, "shape analysis: %d static, %d dynamic out of %d slots",
             staticCount, dynamicCount, plan->numSlots_);

    // Count identity ops for diagnostics
    int identityCount = 0;
    for (int i = 0; i < plan->numSlots_; i++) {
      if (plan->slots_[i].flags.isIdentityOp) identityCount++;
    }
    if (identityCount > 0) {
      DSP_DIAG(SHAPE, "%d identity ops (will use fast-path)", identityCount);
    }
  }

  // Build graph segments for GPU graph capture
  plan->buildSegments();

  // Detect and apply fusion candidates
  if (plan->numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(plan->slots_, plan->numSlots_);
    if (!fusions.empty()) {
      DSP_DIAG(FUSION, "detected %d fusion candidates",
               static_cast<int>(fusions.size()));
      for (auto& f : fusions) {
        DSP_DIAG_SLOT(FUSION, f.startSlot, "fusion: slots %d-%d, type=%d, chain=%d",
                      f.startSlot, f.endSlot, static_cast<int>(f.type), f.chainLength);
      }

      int applied = FusionPass::applyFusions(plan->slots_, plan->numSlots_, fusions);
      DSP_DIAG(FUSION, "applied %d of %d fusion candidates (in-place execution)",
               applied, static_cast<int>(fusions.size()));
    }
  }

  // ── Phase 3: Build shared immutable PlanDefinition ─────────────────────
  {
    auto builder = PlanDefinition::Builder();
    builder.setNumSlots(plan->numSlots_)
           .setTotalOutputSlots(plan->totalOutputSlots_)
           .setNumExternalInputs(plan->numExternalInputs_)
           .setNumRequestedOutputs(plan->numRequestedOutputs_)
           .setRequestedOutputSlotIndices(plan->requestedOutputSlotIndices_,
                                          plan->numRequestedOutputs_)
           .setExternalInputNames(plan->externalInputNames_)
           .setExternalInputIsVariable(plan->externalInputIsVariable_)
           .setHasControlFlow(plan->hasControlFlow_)
           .setNumLoopRegions(plan->numLoopRegions_)
           .setBackendPriority(plan->backendPriority_);
    plan->planDef_ = builder.build();
  }

  // ── Phase 4: Create per-instance ExecutionState ────────────────────────
  plan->execState_ = new ExecutionState(plan->totalOutputSlots_);

  // Notify diagnostics that a plan was compiled
  DspDiagnostics::getInstance().beginPlanExecution(
      plan->numSlots_, static_cast<int>(plan->segments_.size()));
  DSP_DIAG(COMPILE, "plan compiled: %d slots, %d segments, planDef refCount=%d",
           plan->numSlots_, static_cast<int>(plan->segments_.size()),
           plan->planDef_ ? plan->planDef_->refCount() : -1);

  return plan;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status NativeDynamicShapePlan::execute(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs,
    void* stream) {

  if (numExternalInputs != numExternalInputs_) {
    DSP_DIAG(EXECUTE, "NativeDynamicShapePlan::execute: expected %d external inputs, got %d",
              numExternalInputs_, numExternalInputs);
    return Status::BAD_ARGUMENTS;
  }

  if (numRequestedOutputs != numRequestedOutputs_) {
    DSP_DIAG(EXECUTE, "NativeDynamicShapePlan::execute: expected %d requested outputs, got %d",
              numRequestedOutputs_, numRequestedOutputs);
    return Status::BAD_ARGUMENTS;
  }

  DspDiagnostics::getInstance().beginStep(executeCount_);

  // Set tl_dspExecutionStream at the start of EVERY DSP execution.
  // This allows syncToSpecial() to use async H2D copies on the DSP stream instead
  // of falling back to stream 0 with full cudaStreamSynchronize.
  // Without this, we get 657k sync calls per decode step.
  //
  // Multi-device safety: tl_dspExecutionStream is thread-local, and each thread
  // executes on a single device at a time. The stream comes from the LaunchContext
  // which is device-specific, so this is safe for multi-device execution.
  //
  // RAII guard: automatically restores the previous tl_dspExecutionStream value
  // when execute() returns (including early returns and exceptions).
  // Platform-specific stream setup + ordering (DspStreamGuard on CUDA, no-op on CPU).
  // executionStatePtr is freed by platformEndExecution at end of execute().
  void* executionStatePtr = platformBeginExecution(stream, shapesFrozen_, executeCount_);

  DSP_DIAG(EXECUTE, "step %d: frozen=%d segs=%d graphCapture=%d ext=%d",
           executeCount_, static_cast<int>(shapesFrozen_),
           static_cast<int>(segments_.size()),
           static_cast<int>(gpuGraphCaptureEnabled_), numExternalInputs);

  std::vector<NDArray*> lifecycleExternalInputs;
  NDArray** lifecycleExternalInputPtrs = externalInputs;
  if (numExternalInputs > 0 && !protectedWeightBuffers_.empty()) {
    lifecycleExternalInputs.assign(externalInputs, externalInputs + numExternalInputs);
    bool filteredAny = false;
    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      DataBuffer* db = arr != nullptr ? arr->dataBuffer() : nullptr;
      bool trackForLifecycle = db != nullptr && protectedWeightBuffers_.count(db) > 0;
      if (!trackForLifecycle) {
        lifecycleExternalInputs[i] = nullptr;
        filteredAny = true;
      }
    }
    if (filteredAny) {
      lifecycleExternalInputPtrs = lifecycleExternalInputs.data();
    }
  }

  platformDumpExternalInputDiagnostics(externalInputs, numExternalInputs, executeCount_);

  // Debug: dump external input at a configured index (ND4J_DSP_TRACE_EXT_INPUT)
  // useful for diagnosing forced-H2D-sync issues where device-authoritative buffers get overwritten
  {
    int traceExt = sd::graph::DspDiagnostics::getInstance().traceExtInput();
    if (traceExt >= 0 && traceExt < numExternalInputs && DSP_DIAG_ENABLED(VERIFY)) {
      NDArray* extArr = externalInputs[traceExt];
      if (extArr != nullptr) {
        DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=%d dtype=%d shape=[%lld] len=%lld "
                 "specialBuf=%p primaryBuf=%p dbPtr=%p pAct=%d sAct=%d",
                 executeCount_, traceExt, (int)extArr->dataType(),
                 (long long)(extArr->rankOf() > 0 ? extArr->sizeAt(0) : 0),
                 (long long)extArr->lengthOf(),
                 DSP_BUF(extArr), extArr->buffer(),
                 static_cast<void*>(extArr->dataBuffer()),
                 extArr->dataBuffer() ? (extArr->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
                 extArr->dataBuffer() ? (extArr->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
        platformDumpExtInputGpuValues(extArr, traceExt, executeCount_, stream);
      }
      // Check if the traced external input shares a buffer with any output slot in the cache
      if (extArr != nullptr && DSP_BUF(extArr) != nullptr && outputSlots_ != nullptr) {
        void* extAddr = DSP_BUF(extArr);
        int aliasCount = 0;
        for (int si = 0; si < totalOutputSlots_; si++) {
          if (outputSlots_[si] != nullptr && DSP_BUF(outputSlots_[si]) == extAddr) {
            DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=%d addr=%p == slotArrayCache[%d] (len=%lld)",
                     traceExt, extAddr, si, (long long)outputSlots_[si]->lengthOf());
            aliasCount++;
          }
        }
        if (aliasCount == 0) {
          DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=%d addr=%p NO alias found in %d output slots",
                   traceExt, extAddr, totalOutputSlots_);
        }
      }
    }
  }

  // Apply pending decode input updates directly to the external input arrays.
  // Graph replay now reads those buffers directly, so no secondary staging step
  // is required after this write.
  if (hasPendingDecodeUpdate_ && isDecodeInputsConfigured()) {
    updateDecodeInputs(externalInputs, numExternalInputs,
                        pendingTokenId_, pendingCachePos_, stream);
  }

  // Frozen graph fast path: if shapes are frozen and a single captured GPU graph
  // covers the entire plan, skip all per-slot/per-segment abstractions.
  // Returns OK if fast path handled execution, MAYBE to fall through.
  auto fastPathResult = platformTryFrozenFastPath(
      externalInputs, numExternalInputs, requestedOutputs, numRequestedOutputs, stream);
  if (fastPathResult != Status::MAYBE) return fastPathResult;

  hasPendingDecodeUpdate_ = false;

  // ── Phase-aware lifecycle validation ─────────────────────────────────────
  // Hard errors (not logs) when buffer lifecycle is violated during frozen execution.
  // This catches: freed buffers, pointer drift, stale ownership, dangling views.
  // When freezeMergeSegments is active, merged segments contain value-dependent ops
  // (reshape, gather, broadcast_to) whose output DataBuffers are re-created on each
  // execution by initializeOutputs. This is correct behavior — shapes are frozen but
  // the allocation path creates fresh arrays. The lifecycle validation (designed for
  // the non-merged case where each segment's slots have stable buffers) incorrectly
  // rejects these buffer replacements as "stale ownership".
  if (planPhase_ >= PlanPhase::SHAPES_FROZEN && executeCount_ > 0) {
    char errMsg[512] = {};
    bool lifecycleOk = validateLifecycleForPhase(
        static_cast<int>(planPhase_),
        slotOwnership_, totalOutputSlots_,
        outputSlots_,
        lifecycleExternalInputPtrs, numExternalInputs,
        protectedWeightBuffers_,
        frozenSnapshot_.valid ? &frozenSnapshot_ : nullptr,
        errMsg, sizeof(errMsg));
    if (!lifecycleOk) {
      DSP_DIAG(FALLBACK, "LIFECYCLE_VALIDATION_FAILED: %s", errMsg);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
          static_cast<int>(Status::KERNEL_FAILURE));
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg);
      return Status::KERNEL_FAILURE;
    }
  }

  if (planPhase_ >= PlanPhase::REPLAYING) {
    // In REPLAYING phase, every replay-eligible segment must still be in
    // backend-specific steady state. If any segment drops out, demote.
    for (size_t si = 0; si < segments_.size(); si++) {
      auto& seg = segments_[si];
      if (!segmentIsFullyReplayingForPlanPhase(seg, slots_)) {
        demotePlanPhase(PlanPhase::POINTERS_STABLE,
            "segment no longer satisfies replay steady state");
        DSP_DIAG(FALLBACK, "  seg[%d-%d] details: backend=%d execPhase=%d "
                  "segExecCount=%d handleReady=%d argStable=%d execCount=%d",
                  seg.def.startSlot, seg.def.endSlot,
                  static_cast<int>(seg.def.selectedBackend),
                  static_cast<int>(seg.exec.currentPhase),
                  seg.exec.executionCount,
                  seg.exec.replayHandle && seg.exec.replayHandle->isReady() ? 1 : 0,
                  seg.exec.argTableStable ? 1 : 0,
                  executeCount_);
        break;
      }
    }
  }

  // Pre-execute setup: clear stale errors, manage attention workspace,
  // flush pending close, invalidate stale cached graphs.
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
  platformPreExecuteSetup(externalInputs, numExternalInputs, stream);

  // Step 1: Initialize output slots
  // When shapes are frozen (after warmup), pre-populate from outputSlots_ so
  // downstream ops can read inputs without each slot individually setting outputSlots_.
  // View-producer slots will be overwritten during execution.
  //
  // Non-capturable (and permanently capture-failed) segments execute slot-by-slot
  // across decode steps. Their shape-driving scalar tensors often keep the same
  // shape while values change (KV length growth), so cross-execution shape cache
  // reuse can become stale and cause later broadcast mismatches. Invalidate these
  // segment-local caches each execute; capturable graph-replay segments keep caches.

  // replayManagedSlots REMOVED: arrays persist (one array per slot), no protection needed.

  // Build protected weight buffer set (once on first execute).
  // DataBuffers from external constants/variables must NEVER be freed. View ops
  // can produce intermediate NDArrays that share a DataBuffer with a model weight.
  // Without this protection, freeing the intermediate frees the weight's GPU memory
  // → CUDA illegal access (error 700) on re-execution.
  // This mirrors the Java-side protectedWeightBuffers in DynamicShapePlanExecutor.
  if (protectedWeightBuffers_.empty()) {
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] != nullptr) {
        auto* db = externalInputs[i]->dataBuffer();
        bool isVariableInput = i < static_cast<int>(externalInputIsVariable_.size()) &&
                               externalInputIsVariable_[i];
        if (db != nullptr && !isVariableInput) {
          protectedWeightBuffers_.insert(db);
        }
      }
    }
    DSP_DIAG(MEMORY, "built protectedWeightBuffers with %zu entries from %d external inputs "
             "(excluding variable/placeholder feeds)",
             protectedWeightBuffers_.size(), numExternalInputs);


  }

  if (numExternalInputs > 0 && !protectedWeightBuffers_.empty()) {
    lifecycleExternalInputs.assign(externalInputs, externalInputs + numExternalInputs);
    bool filteredAny = false;
    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      DataBuffer* db = arr != nullptr ? arr->dataBuffer() : nullptr;
      bool trackForLifecycle = db != nullptr && protectedWeightBuffers_.count(db) > 0;
      if (!trackForLifecycle) {
        lifecycleExternalInputs[i] = nullptr;
        filteredAny = true;
      }
    }
    lifecycleExternalInputPtrs = filteredAny ? lifecycleExternalInputs.data() : externalInputs;
  }

  // ── One array per slot: NO cleanup between executions ────────────────────
  // Same plan = same shapes. Arrays allocated on first execution, reused forever.
  // View ops create lightweight wrappers that are deleted inline when replaced.
  // No free→null→restore cycle. No pendingClose_. No release schedule.
  DSP_DIAG(MEMORY, "execute: arrays persist (exec=%d, frozen=%d, slots=%d)",
           executeCount_, shapesFrozen_ ? 1 : 0, totalOutputSlots_);

  // Non-frozen first execution only: reset segment state for warmup
  if (executeCount_ == 0 && !shapesFrozen_) {
    for (auto& segment : segments_) {
      segment.exec.executionCount = 0;
      segment.exec.compilationFailed = false;
      segment.exec.captureOomRetries = 0;
      segment.exec.captureRetryAfterExec = 0;
      segment.exec.cachedShapeKey = 0;
      segment.exec.capturedInputAddrKey = 0;
      segment.exec.capturedCreateValueKey = 0;
      segment.exec.gapOpsCapturedInGraph = false;
      if (segment.exec.replayHandle) {
        platformCleanupSegmentForRebuild(segment);
      }
    }
  }

  if (!shapesFrozen_) {
    platformClearCastCache();
  }

  // Reset dead-slot flags once per plan execution (not per segment).
  // Dead flags from Switch in one segment must persist to affect ops in later segments.
  if (hasControlFlow_ && slotIsDead_ != nullptr) {
    std::memset(slotIsDead_, 0, sizeof(bool) * slotIsDeadSize_);
  }

  // Timing instrumentation
  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
  PhaseExecutionStats phaseStats;
  bool phaseHandledPostSegments = false;
  bool phaseHandledOutputs = false;

  // Step 1b: Parallel precompilation of all GPU-compilable segments.
  // Skip the first frozen warmup because shapes are still being populated.
  // phaseCompile() itself checks compilationDone_ and returns immediately if already done.
  if (!compilationDone_ && !(shapesFrozen_ && executeCount_ == 0) &&
      graphExecutionMode_ != GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    phaseCompile(externalInputs, numExternalInputs);
  }

  Status phaseStatus = Status::OK;
  if (shapesFrozen_ && executeCount_ == 0) {
    DSP_DIAG(EXECUTE, "PHASE_DISPATCH: phaseWarmup (frozen=%d execCount=%d mode=%d)",
             static_cast<int>(shapesFrozen_), executeCount_, static_cast<int>(graphExecutionMode_));
    phaseStatus = phaseWarmup(externalInputs, numExternalInputs, stream, &phaseStats);
  } else if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    DSP_DIAG(EXECUTE, "PHASE_DISPATCH: phaseSlotBySlot (frozen=%d execCount=%d mode=%d)",
             static_cast<int>(shapesFrozen_), executeCount_, static_cast<int>(graphExecutionMode_));
    phaseStatus = phaseSlotBySlot(externalInputs, numExternalInputs, stream, &phaseStats);
  } else {
    DSP_DIAG(EXECUTE, "PHASE_DISPATCH: phaseReplay (frozen=%d execCount=%d mode=%d)",
             static_cast<int>(shapesFrozen_), executeCount_, static_cast<int>(graphExecutionMode_));
    phaseStatus = phaseReplay(externalInputs, numExternalInputs, requestedOutputs,
                              numRequestedOutputs, stream, &phaseStats);
    phaseHandledPostSegments = true;
    phaseHandledOutputs = true;
  }
  DSP_DIAG(EXECUTE, "PHASE_DISPATCH: phase returned status=%d", static_cast<int>(phaseStatus));
  if (phaseStatus != Status::OK) return phaseStatus;

  auto tSegsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  if (!phaseHandledPostSegments) {
    platformPostSegmentPoolManagement(shapesFrozen_, executeCount_);
  }

  // ── Consistency assertions: verify slot reuse and replay integrity ───
  // These checks run after every execution to catch lifecycle bugs early.
  if (DSP_DIAG_ENABLED(VERIFY)) {
    int nullSlots = 0, liveSlots = 0, viewSlots = 0;
    int replaySegs = 0, slotBySlotSegsCount = 0, compilationFailedSegs = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) { nullSlots++; }
      else {
        liveSlots++;
        auto* db = outputSlots_[i]->dataBuffer();
        if (db != nullptr && protectedWeightBuffers_.count(db) > 0) viewSlots++;
      }
    }
    for (const auto& seg : segments_) {
      if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) replaySegs++;
      else if (seg.exec.compilationFailed) compilationFailedSegs++;
      else slotBySlotSegsCount++;
    }
    DSP_DIAG(VERIFY, "POST_EXEC exec=%d frozen=%d: slots(live=%d null=%d weightView=%d/%d) "
             "segs(replay=%d sbs=%d capFail=%d/%d) graphReplays=%d slotBySlot=%d",
             executeCount_, shapesFrozen_ ? 1 : 0,
             liveSlots, nullSlots, viewSlots, totalOutputSlots_,
             replaySegs, slotBySlotSegsCount, compilationFailedSegs, (int)segments_.size(),
             phaseStats.graphReplaySegs, phaseStats.slotBySlotSegs);
  }

  if (!phaseHandledOutputs) {
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
        requestedOutputs[i] = outputSlots_[slotIdx];
      } else {
        requestedOutputs[i] = nullptr;
      }
    }
  }

  // Diagnostic: dump requested output slot info and argmax for logits comparison
  if (DSP_DIAG_ENABLED(VERIFY) && sd::graph::DspDiagnostics::getInstance().withinExecLimit(executeCount_)) {
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      if (requestedOutputs[i] != nullptr) {
        auto* arr = requestedOutputs[i];
        DSP_DIAG_SLOT(VERIFY, slotIdx,
            "reqOut[%d] len=%lld dt=%d rank=%d",
            i, (long long)arr->lengthOf(), (int)arr->dataType(), arr->rankOf());
        platformDumpLogitsArgmax(executeCount_, stream);
      } else {
        DSP_DIAG_SLOT(VERIFY, slotIdx, "reqOut[%d] nullptr", i);
      }
    }
  }

  if (!phaseHandledOutputs && kvCacheRetentionEnabled_) {
    scatterKvEntries(externalInputs, numExternalInputs, stream);
    kvCachePosition_++;
  }

  auto tOutputsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Step 4: No flush needed — arrays persist (one array per slot)
  auto tFlushDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Log plan-owned array count and total GPU allocation
  DSP_DIAG(MEMORY, "DSP_EXEC_END execCount=%d: planOwnedArrays=%d totalSlots=%d",
            executeCount_, (int)planOwnedArrays_.size(), totalOutputSlots_);

  // Track execution count for shapes-frozen optimization
  if (shapesFrozen_) executeCount_++;

  // Re-classify slot ownership after the first frozen execution (capture step).
  // The capture execution replaces slot arrays (new allocations for different shapes),
  // which invalidates the compile-time ownership classification. Without this,
  // the next execution's lifecycle validation fails with "stale ownership".
  if (shapesFrozen_ && executeCount_ == 2 && slotOwnership_ != nullptr && outputSlots_ != nullptr) {
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) {
        slotOwnership_[i].reset();
        continue;
      }
      auto* db = outputSlots_[i]->dataBuffer();
      if (db == nullptr) {
        slotOwnership_[i].ownership = BufferOwnership::UNSET;
        slotOwnership_[i].dataBuffer = nullptr;
        continue;
      }
      if (protectedWeightBuffers_.count(db) > 0) {
        slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_WEIGHT;
        slotOwnership_[i].dataBuffer = db;
        continue;
      }
      bool isView = false;
      for (int j = 0; j < i; j++) {
        if (outputSlots_[j] != nullptr && outputSlots_[j]->dataBuffer() == db) {
          slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_SLOT;
          slotOwnership_[i].parentSlotIdx = j;
          slotOwnership_[i].dataBuffer = db;
          isView = true;
          break;
        }
      }
      if (!isView) {
        slotOwnership_[i].ownership = BufferOwnership::SLOT_OWNED;
        slotOwnership_[i].dataBuffer = db;
      }
    }
  }

  // ── Plan-level phase advancement ───────────────────────────────────────────
  // Phase transitions are automatic based on observed stability.
  // Snapshot capture and frozenExecutionCount_ increment happen here (need
  // external inputs), then advancePlanPhase() handles the transitions.
  if (shapesFrozen_ && planPhase_ >= PlanPhase::SHAPES_FROZEN) {
    frozenExecutionCount_++;

    // Capture a buffer pointer snapshot after the first frozen execution.
    if (frozenExecutionCount_ == 1 && !frozenSnapshot_.valid) {
      frozenSnapshot_.capture(outputSlots_, totalOutputSlots_,
                               lifecycleExternalInputPtrs, numExternalInputs);
      DSP_DIAG(EXECUTE, "LIFECYCLE: captured buffer pointer snapshot (%d slots, %d extInputs)",
               totalOutputSlots_, numExternalInputs);
      {
        int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
        if (ts >= 0 && ts < totalOutputSlots_ && outputSlots_[ts] != nullptr) {
          DSP_DIAG(MEMORY, "SNAPSHOT_SLOT_%d: arr=%p db=%p special=%p len=%lld",
                   ts, (void*)outputSlots_[ts], (void*)outputSlots_[ts]->dataBuffer(),
                   (void*)outputSlots_[ts]->specialBuffer(),
                   (long long)outputSlots_[ts]->lengthOf());
        }
      }
    }

    advancePlanPhase();
  }

  // Frozen constant detection MUST run BEFORE Triton precompilation.
  // detectFrozenConstants() marks shape_of and other constant-producing slots as
  // FROZEN_CONSTANT. The Triton IR builder checks frozenConstantSlot() and skips
  // these slots — preventing the compiled kernel from overwriting frozen constant
  // device buffers during graph replay. If precompilation runs first, the Triton
  // kernel includes frozen constant ops, and replay corrupts their device data.
  detectFrozenConstants();

  // Eager precompilation: after warmup (executeCount_ just became 1), all shapes
  // are populated in outputSlots_. Compile all Triton modules now so the 2nd
  // execute() goes straight to replay instead of blocking on compilation.
  // compilationDone_ gate ensures this only happens once per plan lifecycle.
  if (!compilationDone_ && shapesFrozen_ && executeCount_ == 1) {
    phaseCompile(externalInputs, numExternalInputs);
  }

  platformDetectAndPrepareBatchedGemm(externalInputs, numExternalInputs, stream);

  // Adaptive segment splitting (GPU only): if a segment's shape key
  // changes for consecutive executions, split it at the midpoint.
  platformMaybeSplitIfEnabled();

  // Print timing breakdown
  if (executionTimingEnabled_) {
    auto segMs = std::chrono::duration_cast<std::chrono::microseconds>(tSegsDone - t0).count();
    auto outMs = std::chrono::duration_cast<std::chrono::microseconds>(tOutputsDone - tSegsDone).count();
    auto flushMs = std::chrono::duration_cast<std::chrono::microseconds>(tFlushDone - tOutputsDone).count();
    auto totalMs = std::chrono::duration_cast<std::chrono::microseconds>(tFlushDone - t0).count();
    DSP_DIAG(TIMING, "segments=%lldus outputs=%lldus flush=%lldus total=%lldus (%d segs, %d slots) | graph=%lldus(%d segs/%d slots) sbs=%lldus(%d segs/%d slots)",
             segMs, outMs, flushMs, totalMs,
             static_cast<int>(segments_.size()), numSlots_,
             phaseStats.graphReplayUs, phaseStats.graphReplaySegs, phaseStats.graphReplaySlots,
             phaseStats.slotBySlotUs, phaseStats.slotBySlotSegs, phaseStats.slotBySlotSlots);
  }

  DspDiagnostics::getInstance().endStep(executeCount_);

  // Cross-stream synchronization + DspStreamGuard cleanup
  platformEndExecution(executionStatePtr, stream, shapesFrozen_, executeCount_);
  executionStatePtr = nullptr;

  return Status::OK;
}

// ─── Statistics ─────────────────────────────────────────────────────────────

int NativeDynamicShapePlan::getNumCapturedGraphSegments() const {
  return platformCountCapturedGraphSegments();
}

int NativeDynamicShapePlan::getTotalGraphReplays() const {
  return totalGraphReplays_;
}

std::string NativeDynamicShapePlan::getSegmentCompilationAudit(int segIdx) const {
  if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return "{}";
  auto& seg = segments_[segIdx];
  std::ostringstream ss;
  ss << "{\"segmentIdx\":" << segIdx
     << ",\"startSlot\":" << seg.def.startSlot
     << ",\"endSlot\":" << seg.def.endSlot
     << ",\"compiledByBackend\":\"" << seg.exec.compiledByBackend << "\""
     << ",\"capturable\":" << (seg.def.isCapturable ? "true" : "false")
     << ",\"compilationFailed\":" << (seg.exec.compilationFailed ? "true" : "false")
     << ",\"executionCount\":" << seg.exec.executionCount
     << "}";
  return ss.str();
}

void NativeDynamicShapePlan::setBackendPriority(const std::vector<std::string>& priority) {
  backendPriority_ = priority;
  // Reset cached backends so new priority takes effect
  gpuGraphBackendChecked_ = false;
  gpuGraphBackend_ = nullptr;
  cpuGraphBackendChecked_ = false;
  cpuGraphBackend_ = nullptr;
  cpuGraphBackendChainBuilt_ = false;
  cpuGraphBackendChain_.clear();
}

// ─── Memory management ─────────────────────────────────────────────────────

// View wrappers deleted inline in slotexec. No batched/deferred close needed.

void NativeDynamicShapePlan::setShapesFrozen(bool frozen) {
  if (frozen) {
    DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "setShapesFrozen(true)");
  }
  bool wasFrozen = shapesFrozen_;
  shapesFrozen_ = frozen;
  if (frozen && !wasFrozen) {
    auto status = phaseFreeze();
    if (status != Status::OK) {
      DSP_DIAG(FALLBACK, "setShapesFrozen(true): phaseFreeze failed with status %d",
               static_cast<int>(status));
    }
  }
  if (!frozen) {
    for (auto* db : protectedWeightBuffers_) {
      if (db != nullptr) db->removeFrozenRef();
    }

    if (outputSlots_ != nullptr) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr && outputSlots_[i]->dataBuffer() != nullptr) {
          outputSlots_[i]->dataBuffer()->removeFrozenRef();
        }
      }
      DSP_DIAG(MEMORY, "setShapesFrozen(false): removed frozen ref from %d output slot DataBuffers", totalOutputSlots_);
    }

    for (int i = 0; i < numSlots_; i++) {
      if (slots_[i].state_ >= NativeSlot::SlotState::FROZEN) {
        slots_[i].state_ = NativeSlot::SlotState::SHAPE_CACHED;
      }
    }
    frozenConstantDetectionDone_ = false;

    planPhase_ = PlanPhase::SLOT_BY_SLOT;
    pointersStable_ = false;
    frozenExecutionCount_ = 0;
    executeCount_ = 0;
    compilationDone_ = false;
    for (auto& seg : segments_) {
      if (seg.exec.replayHandle) {
        platformCleanupSegmentForRebuild(seg);
      }
      seg.exec.executionCount = 0;
      seg.exec.argTableStable = false;
      seg.exec.gapOpsCapturedInGraph = false;
      seg.exec.cachedShapeKey = 0;
      seg.exec.capturedInputAddrKey = 0;
      seg.exec.capturedCreateValueKey = 0;
      seg.exec.compilationFailed = false;
      seg.exec.captureOomRetries = 0;
      seg.exec.captureRetryAfterExec = 0;
      seg.exec.lastReplayExecCount = 0;
      seg.exec.currentPhase = ExecutionPhase::WARMUP;
    }
    clearAllShapeCachesForce();
    frozenSnapshot_.clear();
  }
}

// ─── Phase lifecycle methods ──────────────────────────────────────────────
// Each method encapsulates ALL work for its phase. No scattered logic.

void NativeDynamicShapePlan::advancePlanPhase() {
  // ── Plan-level phase advancement ───────────────────────────────────────────
  // Phase transitions are automatic based on observed stability:
  //   SHAPES_FROZEN → POINTERS_STABLE: after 2+ frozen executions with every
  //                                     replay-eligible segment in pointer-stable state
  //   POINTERS_STABLE → REPLAYING:     when every replay-eligible segment reaches
  //                                     backend-specific replay steady state
  //
  // Caller is responsible for: incrementing frozenExecutionCount_, capturing
  // frozenSnapshot_ on first frozen execution.
  if (!shapesFrozen_ || planPhase_ < PlanPhase::SHAPES_FROZEN) return;

  // Check pointer stability across all replay-eligible segments.
  if (planPhase_ == PlanPhase::SHAPES_FROZEN && frozenExecutionCount_ >= 2) {
    bool allStable = true;
    for (auto& seg : segments_) {
      if (!segmentHasStablePointersForPlanPhase(seg, slots_)) {
        allStable = false;
        break;
      }
    }
    if (allStable) {
      pointersStable_ = true;
      const char* oldPhase = dsp::planPhaseName(planPhase_);
      planPhase_ = PlanPhase::POINTERS_STABLE;
      DSP_DIAG(EXECUTE, "PLAN_PHASE: %s → POINTERS_STABLE (frozenExec=%d)",
               oldPhase, frozenExecutionCount_);
    }
  }

  // Promote to REPLAYING only once every replay-eligible segment has reached
  // steady-state replay. Mixed-phase plans stay at POINTERS_STABLE.
  if (planPhase_ >= PlanPhase::POINTERS_STABLE) {
    bool hasReplayEligibleSegment = false;
    bool allReplaying = true;
    for (auto& seg : segments_) {
      if (!segmentBlocksPlanPhase(seg)) continue;
      hasReplayEligibleSegment = true;
      if (!segmentIsFullyReplayingForPlanPhase(seg, slots_)) {
        allReplaying = false;
        break;
      }
    }
    if (hasReplayEligibleSegment && allReplaying && planPhase_ != PlanPhase::REPLAYING) {
      const char* oldPhase = dsp::planPhaseName(planPhase_);
      planPhase_ = PlanPhase::REPLAYING;
      DSP_DIAG(EXECUTE, "PLAN_PHASE: %s → REPLAYING (frozenExec=%d)",
               oldPhase, frozenExecutionCount_);
    }
  }
}

void NativeDynamicShapePlan::demotePlanPhase(PlanPhase targetPhase, const char* reason) {
  const char* oldPhase = dsp::planPhaseName(planPhase_);
  const char* newPhase = dsp::planPhaseName(targetPhase);
  DSP_DIAG(FALLBACK, "PHASE_DEMOTION: %s → %s: %s (frozenExec=%d)",
           oldPhase, newPhase, reason, frozenExecutionCount_);
  planPhase_ = targetPhase;
  if (targetPhase < PlanPhase::POINTERS_STABLE) {
    pointersStable_ = false;
  }
}

Status NativeDynamicShapePlan::phaseFreeze() {
  DSP_REQUIRE_PLAN_PHASE_EXACT(PlanPhase::SLOT_BY_SLOT, "phaseFreeze");
  auto& env = Environment::getInstance();
  bool mergeSegments = env.dspFreezeMergeSegments();

  // ── Fusion pass (slot-by-slot → freeze transition) ──────────────────
  if (numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(slots_, numSlots_, externalInputRanks_);
    if (!fusions.empty()) {
      DSP_DIAG(FUSION, "detected %d fusion candidates (post-warmup)",
               (int)fusions.size());
      int applied = FusionPass::applyFusions(slots_, numSlots_, fusions);
      DSP_DIAG(FUSION, "applied %d of %d fusion candidates",
               applied, (int)fusions.size());
    }
  }

  DSP_DIAG(SEGMENT, "SEGMENT_MAP: %d segments (frozen first exec)", (int)segments_.size());
  for (int i = 0; i < (int)segments_.size(); i++) {
    auto& s = segments_[i];
    DSP_DIAG(SEGMENT, "  seg[%d]: slots[%d-%d] capturable=%d hasReplay=%d "
             "compilationFailed=%d execCount=%d",
             i, s.def.startSlot, s.def.endSlot, s.def.isCapturable,
             s.exec.replayHandle != nullptr, s.exec.compilationFailed, s.exec.executionCount);
  }

  planPhase_ = PlanPhase::SHAPES_FROZEN;
  pointersStable_ = false;
  frozenExecutionCount_ = 0;

  DSP_DIAG(EXECUTE, "FROZEN_TRANSITION: unfrozen → FROZEN, "
            "%d segments, %d slots, %d extInputs, mergeSegments=%d, recompile=%d",
            (int)segments_.size(), numSlots_, numExternalInputs_,
            mergeSegments ? 1 : 0, env.dspFreezeRecompile() ? 1 : 0);

  MmulHelper::clearCastCache();

  executeCount_ = 0;
  compilationDone_ = false;
  for (auto& seg : segments_) {
    seg.exec.executionCount = 0;
    if (seg.exec.replayHandle) {
      platformCleanupSegmentForRebuild(seg);
    }
    seg.exec.argTableStable = false;
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.cachedShapeKey = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.compilationFailed = false;
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
  }

  for (auto* db : protectedWeightBuffers_) {
    if (db != nullptr) db->addFrozenRef();
  }

  if (outputSlots_ != nullptr) {
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] != nullptr && outputSlots_[i]->dataBuffer() != nullptr) {
        outputSlots_[i]->dataBuffer()->addFrozenRef();
      }
    }
    DSP_DIAG(MEMORY, "phaseFreeze: added frozen ref to %d output slot DataBuffers", totalOutputSlots_);
  }

  return Status::OK;
}

Status NativeDynamicShapePlan::phaseWarmup(NDArray** externalInputs, int numExternalInputs,
                                           void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseWarmup: BEGIN segments=%d extInputs=%d", (int)segments_.size(), numExternalInputs);

  long long slotBySlotUs = 0;
  int slotBySlotSegs = 0;
  int slotBySlotSlots = 0;
  using Clock = std::chrono::high_resolution_clock;

  // Reset segment state for warmup
  for (auto& segment : segments_) {
    segment.exec.executionCount = 0;
    segment.exec.compilationFailed = false;
    segment.exec.captureOomRetries = 0;
    segment.exec.captureRetryAfterExec = 0;
    segment.exec.cachedShapeKey = 0;
    segment.exec.capturedInputAddrKey = 0;
    segment.exec.capturedCreateValueKey = 0;
    segment.exec.gapOpsCapturedInGraph = false;
    if (segment.exec.replayHandle) {
      platformCleanupSegmentForRebuild(segment);
    }
  }

  platformClearCastCache();

  // Execute all segments slot-by-slot to populate shapes
  int segIdx = 0;
  for (auto& segment : segments_) {
    DSP_DIAG(EXECUTE, "phaseWarmup: seg[%d] slots=[%d-%d] capturable=%d starting...",
             segIdx, segment.def.startSlot, segment.def.endSlot,
             static_cast<int>(segment.def.isCapturable));
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    segment.exec.currentPhase = segment.def.isCapturable
        ? ExecutionPhase::WARMUP
        : ExecutionPhase::SLOT_BY_SLOT;

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
    DSP_DIAG(EXECUTE, "phaseWarmup: seg[%d] slots=[%d-%d] completed status=%d",
             segIdx, segment.def.startSlot, segment.def.endSlot, static_cast<int>(status));
    segIdx++;
    if (status != Status::OK) return status;

    // Increment executionCount so that executeSegmentWithGraph sees exec >= 1
    // and proceeds to graph capture instead of repeating warmup.
    segment.exec.executionCount = 1;

    // Capture baseline shape and address keys for EMULATED_REPLAY segments.
    // Without these baselines, the first frozen replay computes keys and compares
    // against zeros — keys never match, argTableStable stays false, and the plan
    // is permanently stuck at SHAPES_FROZEN (frozenExecutionCount_ >= 2 but
    // segmentHasStablePointersForPlanPhase always returns false).
    // Note: computeSegmentShapeKey hashes small input values (<=32 elements) which
    // requires D2H sync, but this only happens once during warmup.
    if (segment.def.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
      segment.exec.cachedShapeKey =
          computeSegmentShapeKey(segment, externalInputs, numExternalInputs);
      segment.exec.capturedInputAddrKey =
          computeSegmentInputAddrKeyPortable(segment, externalInputs, numExternalInputs);
    }

    // Note: We intentionally do NOT call computeSegmentShapeKey for non-EMULATED_REPLAY
    // segments here. It is extremely expensive (per-element D2H sync for small inputs)
    // and unnecessary during warmup. executeSegmentWithGraph will compute the
    // key when it actually needs it for graph capture.

    if (executionTimingEnabled_) {
      auto segUs = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tSegStart).count();
      slotBySlotUs += segUs;
      slotBySlotSegs++;
      slotBySlotSlots += segment.def.endSlot - segment.def.startSlot + 1;
    }

    platformCleanupMigratedInputs();
    auto postStatus = platformCheckPostSegment(segment);
    if (postStatus != Status::OK) return postStatus;
  }

  if (stats != nullptr) {
    stats->slotBySlotUs = slotBySlotUs;
    stats->slotBySlotSegs = slotBySlotSegs;
    stats->slotBySlotSlots = slotBySlotSlots;
  }

  return Status::OK;
}

void NativeDynamicShapePlan::phaseCompile(NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "phaseCompile");
  if (compilationDone_) return;
  // Require at least one warmup execution so slot shape caches are populated.
  // Without shapes, Triton IR builds fail on cross-segment inputs.
  if (executeCount_ < 1) {
    DSP_DIAG(COMPILE, "phaseCompile: deferred (executeCount=%d, shapes not yet populated)", executeCount_);
    return;
  }
  DSP_DIAG(COMPILE, "phaseCompile: BEGIN segments=%d extInputs=%d", (int)segments_.size(), numExternalInputs);
  platformPrecompileSegments(externalInputs, numExternalInputs);
  compilationDone_ = true;
  DSP_DIAG(COMPILE, "phaseCompile: END (compilationDone=true)");
}

Status NativeDynamicShapePlan::phaseSlotBySlot(NDArray** externalInputs, int numExternalInputs,
                                               void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseSlotBySlot: BEGIN segments=%d extInputs=%d", (int)segments_.size(), numExternalInputs);

  long long slotBySlotUs = 0;
  int slotBySlotSegs = 0;
  int slotBySlotSlots = 0;
  using Clock = std::chrono::high_resolution_clock;

  for (auto& segment : segments_) {
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    segment.exec.currentPhase = segment.def.isCapturable
        ? ExecutionPhase::WARMUP
        : ExecutionPhase::SLOT_BY_SLOT;

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
    if (status != Status::OK) return status;

    if (executionTimingEnabled_) {
      auto segUs = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tSegStart).count();
      slotBySlotUs += segUs;
      slotBySlotSegs++;
      slotBySlotSlots += segment.def.endSlot - segment.def.startSlot + 1;
    }

    platformCleanupMigratedInputs();
    auto postStatus = platformCheckPostSegment(segment);
    if (postStatus != Status::OK) return postStatus;
  }

  if (stats != nullptr) {
    stats->slotBySlotUs = slotBySlotUs;
    stats->slotBySlotSegs = slotBySlotSegs;
    stats->slotBySlotSlots = slotBySlotSlots;
  }

  return Status::OK;
}

Status NativeDynamicShapePlan::phaseReplay(NDArray** externalInputs, int numExternalInputs,
                                           NDArray** requestedOutputs, int numRequestedOutputs,
                                           void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseReplay: BEGIN segments=%d extInputs=%d frozen=%d phase=%d execCount=%d",
           (int)segments_.size(), numExternalInputs, shapesFrozen_ ? 1 : 0,
           static_cast<int>(planPhase_), executeCount_);

  size_t poolUsedPreSegs = 0, poolReservedPreSegs = 0;
  platformPreReplayPoolStats(poolUsedPreSegs, poolReservedPreSegs);

  long long graphReplayUs = 0, slotBySlotUs = 0;
  int graphReplaySegs = 0, slotBySlotSegs = 0, graphReplaySlots = 0, slotBySlotSlots = 0;

  using Clock = std::chrono::high_resolution_clock;

  for (auto& segment : segments_) {
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    bool useGraph = platformShouldUseGraph(segment);
    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    bool segUsedGraph = false;
    int segSlots = segment.def.endSlot - segment.def.startSlot + 1;

    if (segment.def.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
      auto status = executeSegmentEmulatedReplay(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
      segUsedGraph = true;
    } else if (useGraph) {
      auto status = platformExecuteSegmentWithBackends(
          segment, externalInputs, numExternalInputs, stream, segUsedGraph);
      if (status != Status::OK) return status;
    } else {
      segment.exec.currentPhase = segment.def.isCapturable
          ? ExecutionPhase::WARMUP
          : ExecutionPhase::SLOT_BY_SLOT;
      auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
    }

    if (executionTimingEnabled_) {
      auto segUs = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tSegStart).count();
      if (segUsedGraph) {
        graphReplayUs += segUs;
        graphReplaySegs++;
        graphReplaySlots += segSlots;
      } else {
        slotBySlotUs += segUs;
        slotBySlotSegs++;
        slotBySlotSlots += segSlots;
      }
    }

    platformCleanupMigratedInputs();
    auto postStatus = platformCheckPostSegment(segment);
    if (postStatus != Status::OK) return postStatus;

    // Trace slot reporting (GPU only)
    platformTraceSlotValues(segment, stream, executeCount_);

    // NaN detection (gated behind tritonVerifyKernels)
    if (shapesFrozen_ && Environment::getInstance().tritonVerifyKernels()) {
      for (int stepIdx = segment.def.startSlot; stepIdx <= segment.def.endSlot; stepIdx++) {
        auto& slot = slots_[stepIdx];
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          int si = slot.wiring.outputSlotIndices[o];
          if (si < 0 || si >= totalOutputSlots_ || outputSlots_[si] == nullptr) continue;
          auto* arr = outputSlots_[si];
          auto* db = arr->dataBuffer();
          if (db == nullptr || DSP_BUF(arr) == nullptr || arr->lengthOf() == 0) continue;
          bool dbClosed = db->isClosed();
          if (dbClosed) {
            DSP_DIAG_SLOT(VERIFY, stepIdx, slot.ident.opName.c_str(),
                    "NaN_CLOSED_DB seg[%d-%d] outSlot=%d DataBuffer CLOSED! "
                    "frozenConst=%d shapeStatic=%d execCount=%d",
                    segment.def.startSlot, segment.def.endSlot, si,
                    slot.frozenConstantSlot() ? 1 : 0, slot.shapeCache.shapeStatic ? 1 : 0, executeCount_);
            continue;
          }
          arr->syncToHost();
          bool hasNaN = arr->hasNaNs();
          if (hasNaN) {
            bool anyInputNaN = false;
            for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
              int srcIdx = slot.wiring.inputSourceIndices[inp];
              NDArray* srcArr = nullptr;
              if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
                srcArr = outputSlots_[srcIdx];
              } else if (srcIdx < 0) {
                int extIdx = -(srcIdx + 1);
                if (extIdx >= 0 && extIdx < numExternalInputs) srcArr = externalInputs[extIdx];
              }
              if (srcArr != nullptr && srcArr->lengthOf() > 0) {
                srcArr->syncToHost();
                bool inpHasNaN = srcArr->hasNaNs();
                if (inpHasNaN) anyInputNaN = true;
              }
            }
            DSP_DIAG_SLOT(VERIFY, stepIdx, slot.ident.opName.c_str(),
                    "NaN_DETECT seg[%d-%d] output[%d]=%d NaN! "
                    "useGraph=%d execCount=%d len=%lld inputsNaN=%d hasReplay=%d "
                    "frozenConst=%d shapeStatic=%d",
                    segment.def.startSlot, segment.def.endSlot, o, si,
                    useGraph ? 1 : 0, executeCount_, (long long)arr->lengthOf(),
                    anyInputNaN ? 1 : 0,
                    segment.exec.replayHandle != nullptr ? 1 : 0,
                    slot.frozenConstantSlot() ? 1 : 0, slot.shapeCache.shapeStatic ? 1 : 0);
            goto nanDetectDoneReplay;
          }
        }
      }
      nanDetectDoneReplay:;
    }
  }

  // Post-segment pool trimming
  platformPostReplayPoolManagement(poolUsedPreSegs, shapesFrozen_, executeCount_);

  // Copy requested outputs
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      requestedOutputs[i] = outputSlots_[slotIdx];
    } else {
      requestedOutputs[i] = nullptr;
    }
  }

  // KV cache retention
  if (kvCacheRetentionEnabled_) {
    scatterKvEntries(externalInputs, numExternalInputs, stream);
    kvCachePosition_++;
  }

  // Timing breakdown
  if (executionTimingEnabled_) {
    DSP_DIAG(TIMING, "replay: graph=%lldus(%d segs/%d slots) sbs=%lldus(%d segs/%d slots)",
             graphReplayUs, graphReplaySegs, graphReplaySlots,
             slotBySlotUs, slotBySlotSegs, slotBySlotSlots);
  }

  if (stats != nullptr) {
    stats->graphReplayUs = graphReplayUs;
    stats->slotBySlotUs = slotBySlotUs;
    stats->graphReplaySegs = graphReplaySegs;
    stats->slotBySlotSegs = slotBySlotSegs;
    stats->graphReplaySlots = graphReplaySlots;
    stats->slotBySlotSlots = slotBySlotSlots;
  }

  return Status::OK;
}

void NativeDynamicShapePlan::clearShapeCaches() {
  // When shapes are frozen, skip clearing entirely after first execution.
  // All cached shapes remain valid since external input shapes are constant.
  if (shapesFrozen_ && executeCount_ > 0) return;

  for (int i = 0; i < numSlots_; i++) {
    if (!slots_[i].shapeCache.shapeStatic) {
      slots_[i].shapeCache.cachedShapeKey = 0;
      slots_[i].shapeCache.cachedOutputShapes.clear();
      // Demote to WARMUP if currently beyond warmup (non-static slots need re-inference)
      if (slots_[i].state_ > NativeSlot::SlotState::WARMUP) {
        slots_[i].state_ = NativeSlot::SlotState::WARMUP;
      }
    }
  }
}

void NativeDynamicShapePlan::clearAllShapeCachesForce() {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "clearAllShapeCachesForce");
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].shapeCache.cachedShapeKey = 0;
    slots_[i].shapeCache.cachedOutputShapes.clear();
    // Force demote all slots to WARMUP
    if (slots_[i].state_ > NativeSlot::SlotState::WARMUP) {
      slots_[i].state_ = NativeSlot::SlotState::WARMUP;
    }
  }
}

// ─── Reset segment execution state ──────────────────────────────────────────

void NativeDynamicShapePlan::resetSegmentExecutionState() {
  demotePlanPhase(PlanPhase::SLOT_BY_SLOT, "resetSegmentExecutionState");
  for (auto& seg : segments_) {
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
    seg.exec.captureOomRetries = 0;
    if (seg.exec.replayHandle) seg.exec.replayHandle.reset();
    seg.exec.argTableStable = false;
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
  }
  compilationDone_ = false;
}

// ─── Release GPU intermediates ───────────────────────────────────────────────


int NativeDynamicShapePlan::releaseGpuIntermediates() {
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: START plan=%p numSlots=%d totalOutputSlots=%d",
           this, numSlots_, totalOutputSlots_);

  // ── Phase demotion: demote to SLOT_BY_SLOT BEFORE freeing any arrays ──────
  // This ensures no code path can observe POINTERS_STABLE/REPLAYING phase
  // while buffers are being freed, which would violate the phase contract
  // (those phases guarantee stable buffer pointers).
  {
    static const char* phaseNames[] = {"SLOT_BY_SLOT", "SHAPES_FROZEN", "POINTERS_STABLE", "REPLAYING"};
    const char* oldPhaseName = phaseNames[static_cast<int>(planPhase_)];
    static const char* reasonNames[] = {"NORMAL_CLOSE", "SESSION_RESET", "OOM_RECOVERY",
                                         "DEVICE_SWITCH", "CAPTURE_FAILURE", "SHAPE_CHANGE", "ERROR_RECOVERY"};
    const char* reasonName = reasonNames[static_cast<int>(destructionReason_)];
    if (planPhase_ != PlanPhase::SLOT_BY_SLOT) {
      DSP_DIAG(FALLBACK, "releaseGpuIntermediates: demoting planPhase_ %s → SLOT_BY_SLOT "
               "reason=%s before freeing arrays", oldPhaseName, reasonName);
      planPhase_ = PlanPhase::SLOT_BY_SLOT;
      pointersStable_ = false;
      frozenExecutionCount_ = 0;
    }
  }

  // ── Step 1: Free per-segment GPU resources (CUDA graphs, capture workspaces,
  //            pinned host pointers) ──────────────────────────────────────────
  // This is the same cleanup as the destructor's platformFreePlanResources(),
  // but we keep the segment metadata (slot ranges, op definitions) intact.
  platformReleaseSegmentGpuResources();

  // ── Step 2: Free non-weight NDArrays from outputSlots_ ─────────────────
  // Only free SLOT_OWNED buffers. Views and weights are externally owned.
  //
  //  Re-classify ownership before freeing. After CUDA graph capture,
  // outputSlots_[] is restored to warmup arrays (line 2716 in gpubackend.cpp)
  // but slotOwnership_[] still reflects capture-time classification. The warmup
  // arrays may have different ownership than the capture arrays:
  //   - Warmup array has unique buffer → should be SLOT_OWNED → must be freed
  //   - Capture array shared buffer with weight → was VIEW_OF_WEIGHT
  // Without re-classification, warmup arrays with unique buffers are skipped
  // (classified as VIEW_OF_WEIGHT from capture), leaking ~1.7 GB per page cycle.
  int freedCount = 0;
  std::unordered_set<NDArray*> deleted;

  if (outputSlots_) {
    // Re-classify ownership for ALL slots based on the CURRENT outputSlots_[] arrays.
    // protectedWeightBuffers_ contains DataBuffers from ALL external inputs (built
    // during execute()). Any slot whose DataBuffer matches an external input is
    // BORROWED (model-owned, never freed by the plan). Everything else is an
    // intermediate (plan-owned, freed here).
    if (slotOwnership_) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        slotOwnership_[i].reset();
        if (outputSlots_[i] == nullptr) continue;
        auto* db = outputSlots_[i]->dataBuffer();
        if (db == nullptr) {
          slotOwnership_[i].ownership = BufferOwnership::UNSET;
          continue;
        }
        if (protectedWeightBuffers_.count(db) > 0) {
          slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_WEIGHT;
          slotOwnership_[i].dataBuffer = db;
          continue;
        }
        bool isViewOfSlot = false;
        for (int j = 0; j < i; j++) {
          if (outputSlots_[j] != nullptr && outputSlots_[j]->dataBuffer() == db) {
            slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_SLOT;
            slotOwnership_[i].parentSlotIdx = j;
            slotOwnership_[i].dataBuffer = db;
            isViewOfSlot = true;
            break;
          }
        }
        if (!isViewOfSlot) {
          slotOwnership_[i].ownership = BufferOwnership::SLOT_OWNED;
          slotOwnership_[i].dataBuffer = db;
        }
      }
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: re-classified ownership for %d slots "
               "(%zu protected external buffers)", totalOutputSlots_, protectedWeightBuffers_.size());
    }

    if (slotOwnership_) {
      // First pass: null out all VIEW_OF_SLOT entries (they'll be invalidated
      // when their parent SLOT_OWNED buffer is freed).
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr &&
            slotOwnership_[i].ownership == BufferOwnership::VIEW_OF_SLOT) {
          // Don't delete — the parent owns the buffer. Just null out.
          outputSlots_[i] = nullptr;
          slotOwnership_[i].reset();
        }
      }
      // Second pass: free SLOT_OWNED buffers that are plan-owned.
      // Only delete arrays the plan created (in planOwnedArrays_).
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr &&
            slotOwnership_[i].ownership == BufferOwnership::SLOT_OWNED) {
          slotOwnership_[i].viewRefCount = 0;
          if (planOwnedArrays_.count(outputSlots_[i]) > 0 &&
              deleted.insert(outputSlots_[i]).second) {
            planOwnedArrays_.erase(outputSlots_[i]);
            delete outputSlots_[i];
            freedCount++;
          }
          outputSlots_[i] = nullptr;
          slotOwnership_[i].reset();
        }
      }
    } else {
      // Fallback: no ownership info — only free plan-owned arrays
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr && planOwnedArrays_.count(outputSlots_[i]) > 0) {
          if (deleted.insert(outputSlots_[i]).second) {
            planOwnedArrays_.erase(outputSlots_[i]);
            delete outputSlots_[i];
            freedCount++;
          }
          outputSlots_[i] = nullptr;
        }
      }
    }
  }
  platformMigrateWeightsAndClearCaches();

  // ── Step 5: Reset execution state so plan re-warms on next execute() ────
  viewProducerDetectionDone_ = false;
  frozenConstantDetectionDone_ = false;
  executeCount_ = 0;
  compilationDone_ = false;
  // Remove frozen reference counts from weight DataBuffers before clearing
  // shapesFrozen_. During frozen execution, addFrozenRef() was called on each
  // protectedWeightBuffer to prevent DataBuffer::migrate() from relocating
  // buffers whose addresses are baked into frozen slot contexts / CUDA graphs.
  // Now that all graphs and frozen state are torn down, migration is safe again.
  if (shapesFrozen_) {
    for (auto* db : protectedWeightBuffers_) {
      if (db != nullptr) {
        db->removeFrozenRef();
      }
    }
    
    // ALSO remove frozen ref count from all output slot DataBuffers.
    // These were protected during frozen execution to prevent SIGSEGV.
    if (outputSlots_ != nullptr) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr && outputSlots_[i]->dataBuffer() != nullptr) {
          outputSlots_[i]->dataBuffer()->removeFrozenRef();
        }
      }
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: removed frozen ref from %d output slot DataBuffers", totalOutputSlots_);
    }
  }
  // Reset shapesFrozen_ so the plan goes through the full warmup-capture-replay
  // lifecycle from scratch. Without this, stale segment state (shapeKey, cachedShapeKey)
  // causes Triton recompilation to be skipped, and the plan tries to replay
  // CUDA graphs that were destroyed, leading to error 700.
  shapesFrozen_ = false;

  // Disable KV cache retention — after releasing intermediates, the output slots
  // that KV scatter reads from are NULL. The next page's decode loop will
  // re-configure retention at the correct position via configureKvCacheRetention().
  // Without this, a restored cached handle runs KV scatter with stale position
  // against empty destination buffers, causing CUDA error 700.
  kvCacheRetentionEnabled_ = false;
  kvCachePosition_ = 0;

  // Clear shape caches so shapes are re-inferred
  clearAllShapeCachesForce();

  // Clear GPU backend failed-segment cache
  clearGpuBackendFailedCache();

  DSP_DIAG(MEMORY, "releaseGpuIntermediates: DONE plan=%p, freed %d arrays. "
           "Plan is now cold — next execute() will re-warm.", this, freedCount);

  return freedCount;
}

int NativeDynamicShapePlan::releaseGpuIntermediates(bool preserveDecodeState) {
  if (preserveDecodeState) {
    // Decode-invariant path: preserve slot arrays, CUDA graphs, cuBLAS workspace,
    // and batch optimization resources. Only reset KV cache position and decode
    // input pending state.
    DSP_DIAG(MEMORY, "releaseGpuIntermediates(preserve=true): preserving decode-invariant state");
    
    // Reset KV cache position for next page
    kvCachePosition_ = 0;
    
    // Clear pending decode update flag
    hasPendingDecodeUpdate_ = false;
    pendingTokenId_ = 0;
    pendingCachePos_ = 0;
    
    // No GPU memory freed — returning 0
    return 0;
  } else {
    // Full release — call the existing method
    return releaseGpuIntermediates();
  }
}

// ─── KV cache retention ──────────────────────────────────────────────────────

void NativeDynamicShapePlan::configureKvCacheRetention(
    const int* mappings, int numMappings, int maxKvLen, int initialPos) {
  // Free existing mappings
  delete[] kvCacheMappings_;

  kvCacheNumMappings_ = numMappings;
  kvCacheMaxLen_ = maxKvLen;
  kvCachePosition_ = initialPos;

  if (numMappings > 0 && mappings != nullptr) {
    kvCacheMappings_ = new KvCacheMapping[numMappings];
    for (int i = 0; i < numMappings; i++) {
      // Java passes either:
      //   >= 0 : requested output index
      //   <= -2: direct absolute slot index encoded as -(slotIdx + 2)
      int outputRef = mappings[i * 3];
      int slotIdx = -1;
      if (outputRef <= -2) {
        slotIdx = -outputRef - 2;
        if (slotIdx < 0 || slotIdx >= totalOutputSlots_) slotIdx = -1;
      } else if (outputRef >= 0 && outputRef < numRequestedOutputs_) {
        slotIdx = requestedOutputSlotIndices_[outputRef];
      }
      kvCacheMappings_[i].presentOutputSlotIdx = slotIdx;
      kvCacheMappings_[i].pastInputExternalIdx = mappings[i * 3 + 1];
      kvCacheMappings_[i].seqDim = mappings[i * 3 + 2];
    }
    kvCacheRetentionEnabled_ = true;

    // Retained for compatibility with the platform hook; replay now uses the
    // canonical KV buffers directly so this is a no-op on every backend.
    platformMarkKvCaptureBuffersNeverSkip();

    DSP_DIAG(KV_CACHE, "KV cache retention configured: %d mappings, maxLen=%d, initialPos=%d",
             numMappings, maxKvLen, initialPos);
  } else {
    kvCacheMappings_ = nullptr;
    kvCacheRetentionEnabled_ = false;
  }
}

int NativeDynamicShapePlan::advanceKvCachePosition() {
  return ++kvCachePosition_;
}

void NativeDynamicShapePlan::resetKvCachePosition(int newPos) {
  kvCachePosition_ = newPos;
}

void NativeDynamicShapePlan::configureDecodeInputs(
    int inputIdsExtIdx, int positionIdsExtIdx,
    int attentionMaskExtIdx, int maxKvLen) {
  decodeInputIdsExtIdx_ = inputIdsExtIdx;
  decodePositionIdsExtIdx_ = positionIdsExtIdx;
  decodeAttentionMaskExtIdx_ = attentionMaskExtIdx;
  decodeMaxKvLen_ = maxKvLen;
  DSP_DIAG(EXECUTE, "Decode inputs configured: inputIds=%d posIds=%d attnMask=%d maxKvLen=%d",
           inputIdsExtIdx, positionIdsExtIdx, attentionMaskExtIdx, maxKvLen);
}

void NativeDynamicShapePlan::updateDecodeInputs(
    NDArray** externalInputs, int numExt,
    long long tokenId, int cachePos, void* stream) {
  // Direct device writes via cudaMemcpyAsync — zero JNI indexer overhead,
  // correct actuality flags, single stream for ordering with graph replay.
  // Note: for non-pinned H2D, cudaMemcpyAsync stages data before returning,
  // so stack-local source values are safe.
  platformUpdateDecodeInputs(externalInputs, numExt, tokenId, cachePos, stream);
}

void NativeDynamicShapePlan::setOutputSlotMaxSizes(const int* slotIndices, const LongType* maxSizes, int numSlots) {
  if (slotIndices == nullptr || maxSizes == nullptr || numSlots <= 0) return;

  outputSlotMaxSizes_.clear();
  maxAllocatedSlots_.clear();

  for (int i = 0; i < numSlots; i++) {
    if (slotIndices[i] >= 0 && slotIndices[i] < totalOutputSlots_ && maxSizes[i] > 0) {
      outputSlotMaxSizes_[slotIndices[i]] = maxSizes[i];
    }
  }
}

void NativeDynamicShapePlan::setKvCachePosition(int pos) {
  kvCachePosition_ = pos;
}

void NativeDynamicShapePlan::setMaxKvCacheLength(int maxLen) {
  maxKvCacheLen_ = maxLen;
}

void NativeDynamicShapePlan::scatterKvEntries(NDArray** externalInputs, int numExt, void* stream) {
  if (!kvCacheRetentionEnabled_ || kvCacheNumMappings_ == 0) return;

  // Scatter present KV outputs into external input (static) buffers.
  // GPU: uses kvScatter kernel + stream management.
  // CPU: uses operator() + assign() fallback.
  void* savedState = platformBeginKvScatter(stream);

  auto resolveLiveArray = [](NDArray* arr) -> NDArray* {
    if (arr == nullptr) return nullptr;
    if (arr->isEmpty()) return arr;
    auto* db = arr->dataBuffer();
    return (db != nullptr && db->isValid()) ? arr : nullptr;
  };

  // Collect valid scatter entries for batched dispatch
  std::vector<sd::ops::helpers::KvScatterEntry> batchEntries;
  batchEntries.reserve(kvCacheNumMappings_);
  // Track arrays for prepareSpecialUse/registerSpecialUse
  std::vector<std::pair<NDArray*, NDArray*>> scatterPairs;
  // Track contiguous-copy temporaries that must be freed after scatter
  std::vector<NDArray*> tempCopies;
  DataType batchDtype = DataType::HALF;  // default; set from first valid entry

  int skipped = 0;
  for (int m = 0; m < kvCacheNumMappings_; m++) {
    KvCacheMapping& mapping = kvCacheMappings_[m];

    int presentSlotIdx = mapping.presentOutputSlotIdx;
    if (presentSlotIdx < 0 || presentSlotIdx >= totalOutputSlots_) { skipped++; continue; }

    NDArray* presentKv = resolveLiveArray(outputSlots_[presentSlotIdx]);
    if (presentKv == nullptr) {
      presentKv = resolveLiveArray(outputSlots_[presentSlotIdx]);
    } else if (outputSlots_[presentSlotIdx] != presentKv) {
      outputSlots_[presentSlotIdx] = presentKv;
    }

    if (presentKv == nullptr) { skipped++; continue; }

    int extIdx = mapping.pastInputExternalIdx;
    if (extIdx < 0 || extIdx >= numExt) { skipped++; continue; }
    NDArray* staticBuf = externalInputs[extIdx];
    if (staticBuf == nullptr) { skipped++; continue; }

    if (presentKv->rankOf() != 4 || staticBuf->rankOf() != 4) { skipped++; continue; }

    // If presentKv has non-contiguous strides (e.g. from a permute view),
    // the scatter kernel's flat pointer arithmetic will read wrong values.
    // Materialize a contiguous copy before scattering.
    NDArray* scatterSrc = presentKv;
    {
      const int rank = presentKv->rankOf();
      bool isContiguous = (rank <= 1);
      if (!isContiguous) {
        const auto* shapePtr = presentKv->shapeOf();
        const auto* stridePtr = presentKv->stridesOf();
        const char order = presentKv->ordering();
        if (order == 'c') {
          LongType expectedStride = 1;
          isContiguous = true;
          for (int i = rank - 1; i >= 0 && isContiguous; --i) {
            if (shapePtr[i] > 1 && stridePtr[i] != expectedStride) {
              isContiguous = false;
            }
            expectedStride *= shapePtr[i];
          }
        } else {
          LongType expectedStride = 1;
          isContiguous = true;
          for (int i = 0; i < rank && isContiguous; ++i) {
            if (shapePtr[i] > 1 && stridePtr[i] != expectedStride) {
              isContiguous = false;
            }
            expectedStride *= shapePtr[i];
          }
        }
      }
      if (!isContiguous) {
        NDArray* contiguousCopy = presentKv->dup();
        if (contiguousCopy == nullptr) {
          NDArray::registerSpecialUse({staticBuf}, {presentKv});
          skipped++;
          continue;
        }
        tempCopies.push_back(contiguousCopy);
        scatterSrc = contiguousCopy;
        DSP_DIAG(KV_CACHE, "  mapping[%d]: presentKv non-contiguous, created contiguous copy %p", m, contiguousCopy);
      }
    }

    // Validate GPU buffer pointers are non-null and sequence dims are non-zero.
    // After resetForNextPage(), restored cached handles may encounter empty
    // destination buffers (shape [1,H,0,D]) whose specialBuffer() is nullptr.
    // Writing to nullptr causes CUDA error 700.
    NDArray::prepareSpecialUse({staticBuf}, {scatterSrc});

    const void* srcBuf = DSP_BUF(scatterSrc);
    void* dstBuf = DSP_BUF(staticBuf);
    auto srcSeq = scatterSrc->sizeAt(2);
    auto dstSeq = staticBuf->sizeAt(2);

    if (srcBuf == nullptr || dstBuf == nullptr || srcSeq <= 0 || dstSeq <= 0) {
      NDArray::registerSpecialUse({staticBuf}, {scatterSrc}); // balance the prepareSpecialUse
      skipped++;
      continue;
    }

    // Validate cachePos is within the destination buffer's bounds
    if (kvCachePosition_ >= dstSeq) {
      NDArray::registerSpecialUse({staticBuf}, {scatterSrc}); // balance the prepareSpecialUse
      skipped++;
      continue;
    }

    sd::ops::helpers::KvScatterEntry entry;
    entry.srcPtr = srcBuf;
    entry.dstPtr = dstBuf;
    entry.heads = scatterSrc->sizeAt(1);
    entry.srcSeqLen = srcSeq;
    entry.dstSeqLen = dstSeq;
    entry.dim = scatterSrc->sizeAt(3);
    entry.lastPos = entry.srcSeqLen - 1;
    entry.cachePos = kvCachePosition_;
    batchEntries.push_back(entry);
    if (scatterPairs.empty()) {
      batchDtype = scatterSrc->dataType();
    }
    scatterPairs.push_back({scatterSrc, staticBuf});
  }

  int scattered = static_cast<int>(batchEntries.size());
  if (scattered > 0) {
    auto* lc = LaunchContext::defaultContext();
    sd::ops::helpers::kvScatterBatched(batchEntries.data(), scattered, batchDtype, lc);

    platformPostKvScatterSync(scattered, kvCachePosition_, kvCacheNumMappings_);
  }

  // Register special use for all pairs
  for (auto& pair : scatterPairs) {
    NDArray::registerSpecialUse({pair.second}, {pair.first});
  }

  // Free temporary contiguous copies
  for (NDArray* temp : tempCopies) {
    if (temp != nullptr) {
      delete temp;
    }
  }
  tempCopies.clear();

  platformEndKvScatter(savedState);

  DSP_DIAG(KV_CACHE, "KV scatter (batched): %d scattered, %d skipped, pos=%d, numMappings=%d, execCount=%lld",
           scattered, skipped, kvCachePosition_, kvCacheNumMappings_, (long long)executeCount_);
  if (DSP_DIAG_ENABLED(KV_CACHE) && scattered > 0) {
    auto& e0 = batchEntries[0];
    DSP_DIAG(KV_CACHE, "  entry[0]: srcPtr=%p dstPtr=%p heads=%lld srcSeqLen=%lld dstSeqLen=%lld dim=%lld lastPos=%lld cachePos=%lld dtype=%d",
             e0.srcPtr, e0.dstPtr, (long long)e0.heads, (long long)e0.srcSeqLen,
             (long long)e0.dstSeqLen, (long long)e0.dim, (long long)e0.lastPos, (long long)e0.cachePos,
             (int)batchDtype);
  }
  if (DSP_DIAG_ENABLED(KV_CACHE) && skipped > 0) {
    // Log which mappings were skipped and why
    for (int m = 0; m < kvCacheNumMappings_ && m < 3; m++) {
      KvCacheMapping& mapping = kvCacheMappings_[m];
      int psi = mapping.presentOutputSlotIdx;
      int exi = mapping.pastInputExternalIdx;
      NDArray* fromSlot = (psi >= 0 && psi < totalOutputSlots_) ? outputSlots_[psi] : nullptr;
      NDArray* fromCache = (psi >= 0 && psi < totalOutputSlots_) ? outputSlots_[psi] : nullptr;
      NDArray* extArr = (exi >= 0 && exi < numExt) ? externalInputs[exi] : nullptr;
      DSP_DIAG(KV_CACHE, "  mapping[%d]: presentSlot=%d outputSlots_=%p slotCache=%p extIdx=%d ext=%p",
               m, psi, fromSlot, fromCache, exi, extArr);
      if (fromSlot != nullptr) {
        auto* db = fromSlot->dataBuffer();
        DSP_DIAG(KV_CACHE, "    fromSlot: empty=%d db=%p dbValid=%d rank=%d",
                 fromSlot->isEmpty(), db, db ? db->isValid() : 0, fromSlot->rankOf());
      }
    }
  }
}

// ─── Backend resolution (one-time, at segment build) ────────────────────────

SelectedBackend NativeDynamicShapePlan::resolveBackendForSegment(bool isCapturable) const {
  if (!isCapturable) return SelectedBackend::SLOT_BY_SLOT;

  switch (graphExecutionMode_) {
    case GraphExecutionMode::GEM_SLOT_BY_SLOT:
      return SelectedBackend::SLOT_BY_SLOT;

    case GraphExecutionMode::GEM_CUDA_GRAPHS:
    case GraphExecutionMode::GEM_HIP_GRAPHS:
    case GraphExecutionMode::GEM_LEVELZERO:
    case GraphExecutionMode::GEM_VULKAN:
    case GraphExecutionMode::GEM_METAL:
      return platformResolveBackend(true);

    case GraphExecutionMode::GEM_TRITON:
    case GraphExecutionMode::GEM_NVRTC_JIT:
    case GraphExecutionMode::GEM_PTX_JIT:
    case GraphExecutionMode::GEM_TPU:
    case GraphExecutionMode::GEM_HEXAGON:
      return platformResolveBackend(false);

    case GraphExecutionMode::GEM_MLX:
    case GraphExecutionMode::GEM_ARM_HYBRID:
    case GraphExecutionMode::GEM_NNAPI:
      return SelectedBackend::CPU_GRAPH;

    case GraphExecutionMode::GEM_EMULATED_REPLAY:
      return SelectedBackend::EMULATED_REPLAY;

    case GraphExecutionMode::GEM_AUTO: {
      return platformResolveBackend(false);
    }

    default:
      return SelectedBackend::SLOT_BY_SLOT;
  }
}

// ─── Graph segmentation for GPU graph capture ───────────────────────────────

void NativeDynamicShapePlan::buildSegments() {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "buildSegments");
  if (numSlots_ == 0) {
    DSP_DIAG(SEGMENT, "buildSegments: skipped (numSlots=0)");
    return;
  }
  DSP_DIAG(SEGMENT, "buildSegments: BEGIN numSlots=%d matmulSeg=%d",
           numSlots_, Environment::getInstance().dspMatmulSegmentation() ? 1 : 0);

  // Segmentation policy:
  //
  // Merge as many consecutive slots as possible into each capturable segment.
  // Each contiguous capturable run (with the same device) becomes ONE segment.
  // At runtime, if a segment's shapes are stable it gets captured once and
  // replayed every step. If shapes change, the segment recompiles via the
  // shape key cache — no physical splitting needed.
  //
  // Capturability: a slot is capturable iff:
  //   1. It is NOT data-dependent (where/unique/nms produce variable-length output)
  //   2. Value-dep-shape ops (reshape/concat/gather whose output SHAPE depends on
  //      runtime VALUES) are now capturable — computeSegmentShapeKey hashes actual
  //      data values of small inputs (≤32 elements), so value changes are detected.
  //      Segments containing these ops have hasValueDepOps=true, which forces shape
  //      key recomputation even when shapes are frozen.

  // ALL ops are capturable. The shapeKey system handles dynamic shapes:
  // - computeSegmentShapeKey hashes input values for small arrays
  // - hasValueDepOps forces recomputation even when frozen
  // - cache miss triggers recompilation with correct shapes
  // The old isDataDependent exclusion was for CUDA graph capture which
  // required fixed output shapes. CPU backends (OneDNN/OpenVINO) and
  // the shapeKey system handle recompilation transparently.
  auto isSlotCapturable = [](const NativeSlot& slot, int) -> bool {
    if (slot.cf.controlFlowType != CF_NONE) return false;
    return true;
  };

  // Matmul segmentation: break segments at matmul/attention op boundaries.
  // This isolates element-wise chains for Triton fusion while matmuls run via cuBLAS.
  const bool matmulSegmentation = Environment::getInstance().dspMatmulSegmentation();

  // Adaptive segment size: frozen shapes can use unlimited segments (the entire
  // decode step becomes one CUDA graph capture). Static shapes get a higher limit.
  // Dynamic shapes keep the conservative limit to bound recompilation scope.
  auto allSlotsStaticShape = [this]() -> bool {
    for (int i = 0; i < numSlots_; i++) {
      if (!slots_[i].shapeCache.shapeStatic) return false;
    }
    return true;
  };
  const int MAX_SEGMENT_SIZE = shapesFrozen_ ? 100000 :
                               allSlotsStaticShape() ? 500 : 200;

  auto isMatmulOrAttention = [this](int idx) -> bool {
    auto* op = slots_[idx].ident.op;
    if (!op || !op->getOpDescriptor()) return false;
    return op->getOpDescriptor()->hasAnyTrait(
        sd::ops::OP_TRAIT_MATMUL | sd::ops::OP_TRAIT_ATTENTION);
  };

  GraphSegment current;
  current.def.startSlot = 0;
  current.def.isCapturable = isSlotCapturable(slots_[0], 0);

  for (int i = 1; i < numSlots_; i++) {
    bool thisCapturable = isSlotCapturable(slots_[i], i);
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);
    int currentSize = i - current.def.startSlot;
    bool sizeLimit = (current.def.isCapturable && currentSize >= MAX_SEGMENT_SIZE);

    // Break at matmul/attention boundaries for Triton fusion
    bool matmulBreak = false;
    if (matmulSegmentation) {
      bool thisIsMatmul = isMatmulOrAttention(i);
      bool prevIsMatmul = isMatmulOrAttention(i - 1);
      if (thisIsMatmul != prevIsMatmul) {
        // Transition detected. Only break if:
        // 1. Going from elementwise→matmul AND the elementwise range has
        //    outputs consumed by slots AFTER the upcoming matmul range, OR
        // 2. Going from matmul→elementwise (always break — matmul is a natural
        //    compilation unit boundary, and the elementwise tail should be
        //    isolated for Triton fusion)
        if (prevIsMatmul && !thisIsMatmul) {
          // matmul→elementwise: always break (isolate elementwise for Triton)
          matmulBreak = true;
        } else {
          // elementwise→matmul: check if any slot in the current elementwise
          // range [current.def.startSlot .. i-1] has outputs consumed by
          // slots beyond i (outside this matmul). If yes, break. If all
          // outputs feed only slot i (the matmul), defer the break.
          bool hasExternalConsumers = false;
          for (int s = current.def.startSlot; s < i; s++) {
            for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
              int outIdx = slots_[s].wiring.outputSlotIndices[o];
              // Check if any consumer of this output is beyond the matmul
              for (int c = i + 1; c < numSlots_; c++) {
                for (int ci = 0; ci < slots_[c].wiring.numInputs; ci++) {
                  if (slots_[c].wiring.inputSourceIndices[ci] == outIdx) {
                    hasExternalConsumers = true;
                    break;
                  }
                }
                if (hasExternalConsumers) break;
              }
              if (hasExternalConsumers) break;
            }
            if (hasExternalConsumers) break;
          }
          matmulBreak = hasExternalConsumers;
        }
      }
    }

    bool cpuTraitBreak = platformShouldBreakSegmentAtTraitBoundary(i, i - 1);

    if (thisCapturable != current.def.isCapturable || deviceChange || sizeLimit || matmulBreak || cpuTraitBreak) {
      // End current segment
      current.def.endSlot = i - 1;
      segments_.push_back(std::move(current));

      // Start new segment
      current = GraphSegment();
      current.def.startSlot = i;
      current.def.isCapturable = thisCapturable;
    }
  }

  // Finalize last segment
  current.def.endSlot = numSlots_ - 1;
  segments_.push_back(std::move(current));

  // Log segment structure
  int capturableCount = 0, totalCapturable = 0;
  int staticCapturableCount = 0, dynamicCapturableCount = 0;
  for (auto& seg : segments_) {
    if (seg.def.isCapturable) {
      capturableCount++;
      int sz = seg.def.endSlot - seg.def.startSlot + 1;
      totalCapturable += sz;
      // A segment is "static" if all its slots have stable shapes
      bool allStatic = true;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot && allStatic; s++)
        allStatic = slots_[s].shapeCache.shapeStatic;
      if (allStatic) staticCapturableCount++;
      else dynamicCapturableCount++;
    }
  }
  DSP_DIAG(SEGMENT, "%d segments (%d capturable: %d static, %d dynamic; covering %d/%d slots)",
           (int)segments_.size(), capturableCount,
           staticCapturableCount, dynamicCapturableCount,
           totalCapturable, numSlots_);

  int maxLoggedSegments = 8;
  int logged = std::min(static_cast<int>(segments_.size()), maxLoggedSegments);
  for (int i = 0; i < logged; i++) {
    const auto& seg = segments_[i];
    int targetDevice = -1;
    if (seg.def.startSlot >= 0 && seg.def.startSlot < numSlots_) {
      targetDevice = slots_[seg.def.startSlot].targetDeviceId;
    }
    DSP_DIAG_SEG(SEGMENT, i, "segment[%d] [%d-%d] capturable=%d targetDeviceId=%d",
                 i, seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.def.isCapturable), targetDevice);
  }
  if ((int)segments_.size() > maxLoggedSegments) {
    DSP_DIAG(SEGMENT, "... %d additional segments not shown in device map",
             static_cast<int>(segments_.size()) - maxLoggedSegments);
  }

  // Propagate outputSlots_, resolve backend, and detect value-dep ops for all segments.
  for (auto& seg : segments_) {
    seg.slotArrayCache = outputSlots_;
    seg.def.selectedBackend = resolveBackendForSegment(seg.def.isCapturable);
    // Scan slots for value-dependent ops — these require shape key recomputation
    // even when shapes are frozen, because input VALUES (not just shapes) affect output shape.
    seg.def.hasValueDepOps = false;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      if (slots_[s].flags.outputShapeDependsOnInputValues) {
        seg.def.hasValueDepOps = true;
        break;
      }
    }
    DSP_DIAG_SEG(SEGMENT, seg.def.startSlot, "segment[%d-%d] selectedBackend=%d hasValueDepOps=%d",
                 seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.def.selectedBackend),
                 seg.def.hasValueDepOps ? 1 : 0);
  }

  // Initialize symbolic shape ranges if enabled
  if (Environment::getInstance().dspSymbolicShapes()) {
    int warmup = Environment::getInstance().dspSymbolicShapeWarmup();
    for (auto& seg : segments_) {
      seg.exec.symbolicShapeEnabled = true;
      seg.exec.symbolicWarmupRemaining = warmup;
      seg.exec.symbolicRangeData = createSegmentShapeProfile(warmup);
    }
  }

  // ── Post-pass: merge unprofitable small segments ──────────────────────────
  // Segments below MIN_PROFITABLE_SIZE that consist entirely of transparent ops
  // (views, shapes, identity, constants) are merged into the preceding segment.
  // This mirrors XLA's DeclusterNodes which removes trivially small clusters.
  static constexpr int MIN_PROFITABLE_SIZE = 4;

  if (segments_.size() > 1) {
    std::vector<GraphSegment> merged;
    merged.reserve(segments_.size());
    merged.push_back(std::move(segments_[0]));

    for (size_t i = 1; i < segments_.size(); i++) {
      auto& seg = segments_[i];
      int sz = seg.def.endSlot - seg.def.startSlot + 1;

      // Check if segment is small AND all ops are transparent (non-materializing)
      bool isSmallTransparent = false;
      if (sz < MIN_PROFITABLE_SIZE && seg.def.isCapturable) {
        isSmallTransparent = true;
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          // A slot is "transparent" if it's a view, identity, shape-only, or constant
          bool isTransparent = slots_[s].flags.isViewCapableOp ||
                               slots_[s].flags.isIdentityOp;
          // Also check op traits for shape/constant generation
          if (!isTransparent) {
            uint32_t traits = 0;
            if (slots_[s].ident.op && slots_[s].ident.op->getOpDescriptor()) {
              traits = slots_[s].ident.op->getOpDescriptor()->getTraits();
            }
            if (traits == 0 && !slots_[s].ident.opName.empty()) {
              traits = sd::ops::getOpTraitsByName(slots_[s].ident.opName);
            }
            isTransparent = (traits & (sd::ops::OP_TRAIT_VIEW_PRODUCING |
                                       sd::ops::OP_TRAIT_IDENTITY |
                                       sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT |
                                       sd::ops::OP_TRAIT_CONSTANT_GENERATION)) != 0;
          }
          if (!isTransparent) {
            isSmallTransparent = false;
            break;
          }
        }
      }

      if (isSmallTransparent && !merged.empty()) {
        // Absorb into preceding segment
        auto& prev = merged.back();
        prev.def.endSlot = seg.def.endSlot;
        // Preserve hasValueDepOps
        if (seg.def.hasValueDepOps) prev.def.hasValueDepOps = true;
        DSP_DIAG(SEGMENT, "Merged small transparent segment [%d-%d] (%d slots) into [%d-%d]",
                 seg.def.startSlot, seg.def.endSlot, sz,
                 prev.def.startSlot, prev.def.endSlot);
      } else {
        merged.push_back(std::move(seg));
      }
    }

    if (merged.size() < segments_.size()) {
      DSP_DIAG(SEGMENT, "Profitability post-pass: %d -> %d segments",
               (int)segments_.size(), (int)merged.size());
      segments_ = std::move(merged);
    }
  }
}

// ─── fromFlatGraph (delegates to NativePlanCompiler) ─────────────────────────

NativeDynamicShapePlan* NativeDynamicShapePlan::fromFlatGraph(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {
  return NativePlanCompiler::compile(graph, variables, requestedOutputs);
}

// ─── GPU graph capture audit and validation ─────────────────────────────────
// Moved to platform dispatch (getHostOnlyOps, printCaptureAudit,
// validateCapturedGraph). These methods are GPU-only and defined in the .cu file.

// ─── CPU Graph compilation audit and validation ─────────────────────────────

bool NativeDynamicShapePlan::validateCompiledCpuGraph(int segmentIndex) const {
  if (lastCompilationAudit_.empty()) return true;  // No audit data = no validation

  bool allOpsCompiled = true;

  for (const auto& entry : lastCompilationAudit_) {
    if (!entry.wasCompiled) {
      allOpsCompiled = false;
      const char* backendName = cpuGraphBackend_ ? cpuGraphBackend_->name() : "unknown";
      DSP_DIAG(COMPILE, "CPU GRAPH VALIDATION FAILURE: slot %d (%s) was NOT compiled by %s backend: %s",
                entry.slotIndex, entry.opName.c_str(), backendName, entry.reason.c_str());
    }
  }

  return allOpsCompiled;
}

}  // namespace graph
}  // namespace sd
