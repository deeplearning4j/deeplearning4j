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
#include <sstream>
#include <graph/gpu/SymbolicShapeRanges.h>
#ifdef SD_CUDA
#include <graph/gpu/CaptureBufferRegistry.h>
#endif
#include <graph/DspDiagnostics.h>
#if HAVE_TRITON && defined(SD_CUDA)
#include <graph/gpu/TritonGraphBackend.h>
#endif
#include <graph/FusionPass.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <graph/GraphBackend.h>
#include <array/DataBuffer.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/helper_hash.h>
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

// ─── NativeSlot move operations ───────────────────────────────────────────────

NativeSlot::NativeSlot(NativeSlot&& other) noexcept
    : opHash(other.opHash), op(other.op), opName(std::move(other.opName)),
      numInputs(other.numInputs), inputSourceIndices(other.inputSourceIndices),
      inputSourceTypes(other.inputSourceTypes), numOutputs(other.numOutputs),
      outputSlotIndices(other.outputSlotIndices),
      iArgs(other.iArgs), numIArgs(other.numIArgs),
      tArgs(other.tArgs), numTArgs(other.numTArgs),
      bArgs(other.bArgs), numBArgs(other.numBArgs),
      dArgs(other.dArgs), numDArgs(other.numDArgs),
      sArgs(other.sArgs), numSArgs(other.numSArgs),
      needsZeroedOutput(other.needsZeroedOutput),
      isDataDependent(other.isDataDependent),
      outputShapeDependsOnInputValues(other.outputShapeDependsOnInputValues),
      needsIntLongSync(other.needsIntLongSync),
      isCustomOp(other.isCustomOp),
      isIdentityOp(other.isIdentityOp),
      inPlaceFused(other.inPlaceFused),
      inPlaceFusedInputIdx(other.inPlaceFusedInputIdx),
      structuralIArgCount(other.structuralIArgCount),
      isFusedChainHead(other.isFusedChainHead),
      fusedChainLength(other.fusedChainLength),
      isFusedChainTail(other.isFusedChainTail),
      targetDeviceId(other.targetDeviceId),
      legacyOpType(other.legacyOpType),
      legacyOpNum(other.legacyOpNum),
      cachedShapeKey(other.cachedShapeKey),
      cachedOutputShapes(std::move(other.cachedOutputShapes)),
      shapeStatic(other.shapeStatic),
      state_(other.state_) {
  std::memcpy(fusedChainOpCodes, other.fusedChainOpCodes, sizeof(fusedChainOpCodes));
  std::memcpy(fusedChainSlots, other.fusedChainSlots, sizeof(fusedChainSlots));
  std::memcpy(fusedChainSecondaryInputSources, other.fusedChainSecondaryInputSources, sizeof(fusedChainSecondaryInputSources));
  other.inputSourceIndices = nullptr;
  other.inputSourceTypes = nullptr;
  other.outputSlotIndices = nullptr;
  other.iArgs = nullptr;
  other.tArgs = nullptr;
  other.bArgs = nullptr;
  other.dArgs = nullptr;
  other.sArgs = nullptr;
}

NativeSlot& NativeSlot::operator=(NativeSlot&& other) noexcept {
  if (this != &other) {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
    delete[] iArgs;
    delete[] tArgs;
    delete[] bArgs;
    delete[] dArgs;
    delete[] sArgs;

    opHash = other.opHash;
    op = other.op;
    opName = std::move(other.opName);
    numInputs = other.numInputs;
    inputSourceIndices = other.inputSourceIndices;
    inputSourceTypes = other.inputSourceTypes;
    numOutputs = other.numOutputs;
    outputSlotIndices = other.outputSlotIndices;
    iArgs = other.iArgs; numIArgs = other.numIArgs;
    tArgs = other.tArgs; numTArgs = other.numTArgs;
    bArgs = other.bArgs; numBArgs = other.numBArgs;
    dArgs = other.dArgs; numDArgs = other.numDArgs;
    sArgs = other.sArgs; numSArgs = other.numSArgs;
    needsZeroedOutput = other.needsZeroedOutput;
    isDataDependent = other.isDataDependent;
    outputShapeDependsOnInputValues = other.outputShapeDependsOnInputValues;
    needsIntLongSync = other.needsIntLongSync;
    isCustomOp = other.isCustomOp;
    isIdentityOp = other.isIdentityOp;
    inPlaceFused = other.inPlaceFused;
    inPlaceFusedInputIdx = other.inPlaceFusedInputIdx;
    structuralIArgCount = other.structuralIArgCount;
    isFusedChainHead = other.isFusedChainHead;
    fusedChainLength = other.fusedChainLength;
    std::memcpy(fusedChainOpCodes, other.fusedChainOpCodes, sizeof(fusedChainOpCodes));
    std::memcpy(fusedChainSlots, other.fusedChainSlots, sizeof(fusedChainSlots));
    std::memcpy(fusedChainSecondaryInputSources, other.fusedChainSecondaryInputSources, sizeof(fusedChainSecondaryInputSources));
    isFusedChainTail = other.isFusedChainTail;
    targetDeviceId = other.targetDeviceId;
    legacyOpType = other.legacyOpType;
    legacyOpNum = other.legacyOpNum;
    cachedShapeKey = other.cachedShapeKey;
    cachedOutputShapes = std::move(other.cachedOutputShapes);
    shapeStatic = other.shapeStatic;
    state_ = other.state_;

    other.inputSourceIndices = nullptr;
    other.inputSourceTypes = nullptr;
    other.outputSlotIndices = nullptr;
    other.iArgs = nullptr;
    other.tArgs = nullptr;
    other.bArgs = nullptr;
    other.dArgs = nullptr;
    other.sArgs = nullptr;
  }
  return *this;
}

// ─── NativeDynamicShapePlan ─────────────────────────────────────────────────

NativeDynamicShapePlan::NativeDynamicShapePlan()
    : slots_(nullptr), numSlots_(0), totalOutputSlots_(0), numExternalInputs_(0),
      releaseAtStep_(nullptr), releaseAtStepCounts_(nullptr),
      requestedOutputSlotIndices_(nullptr), numRequestedOutputs_(0),
      outputSlots_(nullptr), slotArrayCache_(nullptr), slotIsViewProducer_(nullptr),
      contextPool_(nullptr), viewProducerDetectionDone_(false), frozenConstantDetectionDone_(false),
      gpuGraphCaptureEnabled_(false), totalGraphReplays_(0), jitMode_(JitMode::GRAPH_ONLY), graphExecutionMode_(GraphExecutionMode::GEM_AUTO),
      shapesFrozen_(false), executeCount_(0), executionTimingEnabled_(false), traceEnabled_(false),
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

NativeDynamicShapePlan::~NativeDynamicShapePlan() {
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: START plan=%p numSlots=%d totalOutputSlots=%d",
           this, numSlots_, totalOutputSlots_);

  // Finalize diagnostics report
  DspDiagnostics::getInstance().endPlanExecution();
  DspDiagnostics::getInstance().printPlanReport();
  DspDiagnostics::getInstance().flushJsonReport();

#ifdef SD_CUDA
  // Free CUDA event used for cross-stream sync
  if (executionCompleteEvent_ != nullptr) {
    cudaEvent_t evt = *static_cast<cudaEvent_t*>(executionCompleteEvent_);
    cudaEventDestroy(evt);
    delete static_cast<cudaEvent_t*>(executionCompleteEvent_);
    executionCompleteEvent_ = nullptr;
  }
#endif

  // ── Phase 1: Free GPU resources FIRST ─────────────────────────────────
  // Platform GPU resources (capture buffers, replay handles, JIT kernels,
  // cuBLAS workspace, batch-zero) may hold directReference pointers into
  // outputSlots_. Clean them BEFORE freeing slot arrays to avoid dangling
  // pointer access during capture buffer cleanup.
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

  // Free slot arrays (slotArrayCache_ is unified with outputSlots_ — same pointer)
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing outputSlots_ (%d slots)", totalOutputSlots_);
  if (outputSlots_) {
    int cacheCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] != nullptr && deleted.insert(outputSlots_[i]).second) {
        cacheCount++;
        delete outputSlots_[i];
      }
    }
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: deleted %d unique arrays from outputSlots_", cacheCount);
    delete[] outputSlots_;
  }
  // slotArrayCache_ is an alias of outputSlots_ — do NOT delete[] separately

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

  // Free external input staging buffers
  if (externalInputStagingBuffers_ != nullptr) {
    for (int i = 0; i < numExternalInputs_; i++) {
      if (externalInputStagingBuffers_[i] != nullptr) {
        delete externalInputStagingBuffers_[i];
      }
    }
    delete[] externalInputStagingBuffers_;
    externalInputStagingBuffers_ = nullptr;
  }
  externalInputMaxSizes_.clear();
  externalInputUseStaging_.clear();

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
    slot.opHash = reader.read<int64_t>();
    slot.opName = reader.readString();
    slot.numInputs = reader.read<int32_t>();
    slot.numOutputs = reader.read<int32_t>();

    REQUIRE_TRUE(slot.numInputs >= 0 && slot.numInputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numInputs %d", s, slot.numInputs);
    REQUIRE_TRUE(slot.numOutputs >= 0 && slot.numOutputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numOutputs %d", s, slot.numOutputs);

    // Input wiring
    slot.inputSourceIndices = new int[slot.numInputs];
    reader.readArray(slot.inputSourceIndices, slot.numInputs);

    // Validate each inputSourceIndex is in valid range
    for (int i = 0; i < slot.numInputs; i++) {
      REQUIRE_TRUE(slot.inputSourceIndices[i] >= -(plan->numExternalInputs_ + 1) &&
                   slot.inputSourceIndices[i] < plan->totalOutputSlots_, 0,
                   "NativeDynamicShapePlan::fromSerializedPlan: slot %d inputSourceIndices[%d]=%d out of range [%d, %d)",
                   s, i, slot.inputSourceIndices[i], -(plan->numExternalInputs_ + 1), plan->totalOutputSlots_);
    }

    slot.inputSourceTypes = new int8_t[slot.numInputs];
    reader.readArray(slot.inputSourceTypes, slot.numInputs);

    // Output wiring
    slot.outputSlotIndices = new int[slot.numOutputs];
    reader.readArray(slot.outputSlotIndices, slot.numOutputs);

    // iArgs
    slot.numIArgs = reader.read<int32_t>();
    if (slot.numIArgs > 0) {
      slot.iArgs = new LongType[slot.numIArgs];
      reader.readArray(slot.iArgs, slot.numIArgs);
    }

    // tArgs
    slot.numTArgs = reader.read<int32_t>();
    if (slot.numTArgs > 0) {
      slot.tArgs = new double[slot.numTArgs];
      reader.readArray(slot.tArgs, slot.numTArgs);
    }

    // bArgs
    slot.numBArgs = reader.read<int32_t>();
    if (slot.numBArgs > 0) {
      slot.bArgs = new bool[slot.numBArgs];
      reader.readArray(slot.bArgs, slot.numBArgs);
    }

    // dArgs
    slot.numDArgs = reader.read<int32_t>();
    if (slot.numDArgs > 0) {
      slot.dArgs = new DataType[slot.numDArgs];
      // dArgs are serialized as int32
      for (int i = 0; i < slot.numDArgs; i++) {
        slot.dArgs[i] = static_cast<DataType>(reader.read<int32_t>());
      }
    }

    slot.numSArgs = 0;
    if (version >= 5) {
      slot.numSArgs = reader.read<int32_t>();
      if (slot.numSArgs > 0) {
        slot.sArgs = new std::string[slot.numSArgs];
        for (int i = 0; i < slot.numSArgs; i++) {
          slot.sArgs[i] = reader.readString();
        }
      }
    }

    // Flags
    slot.needsZeroedOutput = reader.read<uint8_t>() != 0;
    slot.isDataDependent = reader.read<uint8_t>() != 0;
    slot.outputShapeDependsOnInputValues = reader.read<uint8_t>() != 0;
    slot.needsIntLongSync = reader.read<uint8_t>() != 0;
    slot.isCustomOp = reader.read<uint8_t>() != 0;
    slot.targetDeviceId = reader.read<int32_t>();

    // V2: legacy op type and opNum for ops not registered as DeclarableOp
    slot.legacyOpType = 0;
    slot.legacyOpNum = -1;
    if (version >= 2) {
      slot.legacyOpType = reader.read<int32_t>();
      slot.legacyOpNum = reader.read<int32_t>();
    }

    // V3: control flow metadata
    slot.controlFlowType = CF_NONE;
    slot.loopBackTarget = -1;
    slot.loopRegionIndex = -1;
    if (version >= 3) {
      slot.controlFlowType = static_cast<ControlFlowType>(reader.read<uint8_t>());
      slot.loopBackTarget = reader.read<int32_t>();
      slot.loopRegionIndex = reader.read<int32_t>();
    }

    // Resolve op by name (Java and C++ use different hash functions,
    // so we look up by name string and compute the C++ hash from it)
    slot.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.opName);
    if (!slot.op && slot.legacyOpType > 0 && slot.legacyOpNum >= 0) {
      // Create a legacy op wrapper for ops not in the OpRegistrator
      // (e.g., exp, log, abs, neg, sqrt, sin, cos, etc.)
      sd::ops::DeclarableOp* legacyOp = nullptr;
      switch (slot.legacyOpType) {
        case 1:  // LegacyTransformSameOp
          legacyOp = new sd::ops::LegacyTransformSameOp(slot.legacyOpNum);
          break;
        case 2:  // LegacyTransformStrictOp
          legacyOp = new sd::ops::LegacyTransformStrictOp(slot.legacyOpNum);
          break;
        case 3:  // LegacyTransformFloatOp
          legacyOp = new sd::ops::LegacyTransformFloatOp(slot.legacyOpNum);
          break;
        case 4:  // LegacyTransformBoolOp
          legacyOp = new sd::ops::LegacyTransformBoolOp(slot.legacyOpNum);
          break;
        case 5:  // LegacyScalarOp
          legacyOp = new sd::ops::LegacyScalarOp(slot.legacyOpNum);
          break;
        case 6:  // LegacyPairwiseTransformOp
          legacyOp = new sd::ops::LegacyPairwiseTransformOp(slot.legacyOpNum);
          break;
        case 7:  // LegacyScalarBoolOp
          legacyOp = new sd::ops::LegacyScalarBoolOp(slot.legacyOpNum);
          break;
        default:
          DSP_DIAG(COMPILE, "unknown legacy op type %d for '%s'",
                    slot.legacyOpType, slot.opName.c_str());
          break;
      }
      if (legacyOp) {
        plan->ownedLegacyOps_.push_back(legacyOp);
        slot.op = legacyOp;
        sd_debug("NativeDynamicShapePlan: created legacy op type=%d num=%d for '%s'\n",
                 slot.legacyOpType, slot.legacyOpNum, slot.opName.c_str());
      }
    }
    if (!slot.op && slot.controlFlowType != CF_NONE) {
      // Control flow ops dont need a DeclarableOp — dispatched by CF engine
      sd_debug("NativeDynamicShapePlan: CF op '%s' (type=%d) — no DeclarableOp needed\n",
               slot.opName.c_str(), static_cast<int>(slot.controlFlowType));
    } else if (!slot.op) {
      DSP_DIAG(COMPILE, "NativeDynamicShapePlan: op not found for name '%s' (serialized hash: %lld, legacyType: %d, legacyNum: %d)",
                slot.opName.c_str(), slot.opHash, slot.legacyOpType, slot.legacyOpNum);
      delete plan;
      return nullptr;
    }


    // Use the C++ hash for internal computations (shape key, etc.)
    slot.opHash = sd::ops::HashHelper::getInstance().getLongHash(slot.opName);
    // Classify identity ops for fast-path skipping
    slot.isIdentityOp = (normalizeOpName(slot.opName) == "identity");
    {
      auto normalized = normalizeOpName(slot.opName);
      slot.isViewCapableOp = (normalized == "reshape" || normalized == "reshape_no_copy" ||
                              normalized == "expand_dims" || normalized == "squeeze" ||
                              normalized == "permute" || normalized == "strided_slice");
    }
    // View-capable ops share input buffer → no zeroing needed
    if (slot.isViewCapableOp) slot.needsZeroedOutput = false;

    // Set structural iArg count from table (consistent with NativePlanCompiler)
    slot.structuralIArgCount = getStructuralIArgCount(normalizeOpName(slot.opName));

    // Initialize fusion fields (will be set by FusionPass::applyFusions later)
    slot.inPlaceFused = false;
    slot.inPlaceFusedInputIdx = -1;
    slot.isFusedChainHead = false;
    slot.fusedChainLength = 0;
    slot.isFusedChainTail = false;
    std::memset(slot.fusedChainOpCodes, 0, sizeof(slot.fusedChainOpCodes));
    std::memset(slot.fusedChainSlots, 0, sizeof(slot.fusedChainSlots));
    std::fill(std::begin(slot.fusedChainSecondaryInputSources), std::end(slot.fusedChainSecondaryInputSources), INT32_MIN);
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
      if (plan->slots_[s].controlFlowType != CF_NONE) {
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
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < plan->numExternalInputs_ &&
            slot.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
          plan->externalInputIsVariable_[extIdx] = true;
        }
      }
    }
  }

  // Allocate execution state
  plan->outputSlots_ = new NDArray*[plan->totalOutputSlots_];
  std::memset(plan->outputSlots_, 0, sizeof(NDArray*) * plan->totalOutputSlots_);

  // slotArrayCache_ unified with outputSlots_ (same pointer, Phase 2)
  plan->slotArrayCache_ = plan->outputSlots_;

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
      for (int i = 0; i < slot.numOutputs; i++) {
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < plan->totalOutputSlots_) {
          outputSlotToStepIndex[si] = s;
        }
      }
    }

    int staticCount = 0, dynamicCount = 0;
    for (int s = 0; s < plan->numSlots_; s++) {
      NativeSlot& slot = plan->slots_[s];
      slot.shapeStatic = true;  // assume static

      // Data-dependent ops always dynamic (output shape depends on runtime values)
      if (slot.isDataDependent || slot.outputShapeDependsOnInputValues) {
        slot.shapeStatic = false;
        dynamicCount++;
        continue;
      }

      for (int i = 0; i < slot.numInputs; i++) {
        int srcIdx = slot.inputSourceIndices[i];
        if (srcIdx < 0) {
          // External input: placeholders are dynamic, constants/variables are static
          if (slot.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            slot.shapeStatic = false;
            break;
          }
        } else {
          // From prior slot output — check if producer is dynamic
          if (srcIdx < plan->totalOutputSlots_) {
            int producerStep = outputSlotToStepIndex[srcIdx];
            if (producerStep >= 0 && !plan->slots_[producerStep].shapeStatic) {
              slot.shapeStatic = false;
              break;
            }
          }
        }
      }

      if (slot.shapeStatic) staticCount++;
      else dynamicCount++;
    }

    DSP_DIAG(SHAPE, "shape analysis: %d static, %d dynamic out of %d slots",
             staticCount, dynamicCount, plan->numSlots_);

    // Count identity ops for diagnostics
    int identityCount = 0;
    for (int i = 0; i < plan->numSlots_; i++) {
      if (plan->slots_[i].isIdentityOp) identityCount++;
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

  // CRITICAL FIX: Set tl_dspExecutionStream at the start of EVERY DSP execution.
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
#ifdef SD_CUDA
  // Declare outside the if so it lives for the entire function scope.
  // On CPU builds, DspStreamGuard is a no-op (accepts void*).
  std::unique_ptr<sd::graph::DspStreamGuard> dspStreamGuardPtr;
  if (stream != nullptr) {
    dspStreamGuardPtr = std::make_unique<sd::graph::DspStreamGuard>(
        *static_cast<cudaStream_t*>(stream));
  }
#endif

  DSP_DIAG(EXECUTE, "step %d: frozen=%d segs=%d graphCapture=%d ext=%d",
           executeCount_, static_cast<int>(shapesFrozen_),
           static_cast<int>(segments_.size()),
           static_cast<int>(gpuGraphCaptureEnabled_), numExternalInputs);

#ifdef SD_CUDA
  // Sync DSP stream at the start of execution to ensure all async CUDA operations
  // Sync DSP stream on the first execution and shape transitions to ensure
  // Java-side inter-step operations (DataBuffer closes, H2D copies) complete
  // before array manipulation. On frozen replay steps (executeCount_ > 1),
  // the previous step's end-of-execution sync already guarantees this.
  if (!shapesFrozen_ || executeCount_ <= 1) {
    if (stream != nullptr) {
      cudaStream_t cudaStr = *static_cast<cudaStream_t*>(stream);
      cudaStreamSynchronize(cudaStr);
    }
  }
#endif

  // Staging buffer copy deferred - staging buffers not yet allocated

  // Debug: dump external input at index 1331 (slot -1332) — useful for diagnosing
  // forced-H2D-sync issues where device-authoritative buffers get overwritten
  if (sd::Environment::getInstance().isDebug() && numExternalInputs > 1331) {
    NDArray* ext1331 = externalInputs[1331];
    if (ext1331 != nullptr) {
      DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=1331 dtype=%d shape=[%lld] len=%lld "
               "specialBuf=%p primaryBuf=%p dbPtr=%p pAct=%d sAct=%d",
               executeCount_, (int)ext1331->dataType(),
               (long long)(ext1331->rankOf() > 0 ? ext1331->sizeAt(0) : 0),
               (long long)ext1331->lengthOf(),
               ext1331->specialBuffer(), ext1331->buffer(),
               static_cast<void*>(ext1331->dataBuffer()),
               ext1331->dataBuffer() ? (ext1331->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
               ext1331->dataBuffer() ? (ext1331->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
#ifdef SD_CUDA
      if (ext1331->specialBuffer() != nullptr && ext1331->lengthOf() > 0
          && ext1331->dataType() == FLOAT32) {
        int dumpCount = std::min((int)ext1331->lengthOf(), 8);
        std::vector<float> hostBuf(dumpCount);
        cudaDeviceSynchronize();
        cudaMemcpy(hostBuf.data(), ext1331->specialBuffer(), dumpCount * 4, cudaMemcpyDeviceToHost);
        std::string valStr;
        for (int v = 0; v < dumpCount; v++) {
          if (v > 0) valStr += ",";
          char buf[32]; snprintf(buf, sizeof(buf), "%.6f", hostBuf[v]); valStr += buf;
        }
        DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=1331 GPU values: %s",
                 executeCount_, valStr.c_str());
      }
#endif
    }
    // Check if externalInputs[1331] shares a buffer with any output slot in the cache
    if (ext1331 != nullptr && ext1331->specialBuffer() != nullptr && slotArrayCache_ != nullptr) {
      void* extAddr = ext1331->specialBuffer();
      int aliasCount = 0;
      for (int si = 0; si < totalOutputSlots_; si++) {
        if (slotArrayCache_[si] != nullptr && slotArrayCache_[si]->specialBuffer() == extAddr) {
          DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=1331 addr=%p == slotArrayCache[%d] (len=%lld)",
                   extAddr, si, (long long)slotArrayCache_[si]->lengthOf());
          aliasCount++;
        }
      }
      if (aliasCount == 0) {
        DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=1331 addr=%p NO alias found in %d output slots",
                 extAddr, totalOutputSlots_);
      }
    }
  }

  // Apply pending decode input updates to external input arrays (if configured).
  // This updates the source arrays that the frozen path's D2D copies read from.
  // The frozen path additionally writes directly to capture buffers (which the
  // graph actually reads from) to handle cases where Java skipped feedDict updates.
  // For the non-frozen fallback path, this is the only update needed.
  if (hasPendingDecodeUpdate_ && isDecodeInputsConfigured()) {
    updateDecodeInputs(externalInputs, numExternalInputs,
                        pendingTokenId_, pendingCachePos_, stream);
    // Do NOT clear hasPendingDecodeUpdate_ here — the frozen fast path
    // also needs it to write directly to capture buffers.
  }

  // Frozen graph fast path: if shapes are frozen and a single captured GPU graph
  // covers the entire plan, skip all per-slot/per-segment abstractions.
  // Returns OK if fast path handled execution, MAYBE to fall through.
  auto fastPathResult = platformTryFrozenFastPath(
      externalInputs, numExternalInputs, requestedOutputs, numRequestedOutputs, stream);
  if (fastPathResult != Status::MAYBE) return fastPathResult;

  // Frozen path didn't handle execution — clear the pending flag for fallback path.
  // (External inputs were already updated by updateDecodeInputs above.)
  hasPendingDecodeUpdate_ = false;

  // Pre-execute setup: clear stale errors, manage attention workspace,
  // flush pending close, invalidate stale cached graphs.
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
  platformPreExecuteSetup(externalInputs, numExternalInputs, stream);

  // Step 1: Initialize output slots
  // When shapes are frozen (after warmup), pre-populate from slotArrayCache_ so
  // downstream ops can read inputs without each slot individually setting outputSlots_.
  // View-producer slots will be overwritten during execution.
  //
  // Non-capturable (and permanently capture-failed) segments execute slot-by-slot
  // across decode steps. Their shape-driving scalar tensors often keep the same
  // shape while values change (KV length growth), so cross-execution shape cache
  // reuse can become stale and cause later broadcast mismatches. Invalidate these
  // segment-local caches each execute; capturable graph-replay segments keep caches.
  // Determine if any segment has a replay handle (used for pre-populate decision below)
  bool hasAnyReplayHandle = false;
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle != nullptr) {
      hasAnyReplayHandle = true;
      break;
    }
  }

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
        if (db != nullptr) {
          protectedWeightBuffers_.insert(db);
        }
      }
    }
    DSP_DIAG(MEMORY, "built protectedWeightBuffers with %zu entries from %d external inputs",
             protectedWeightBuffers_.size(), numExternalInputs);

    // External inputs are WEIGHT — never freed by the plan.
    // Output slot ownership is set when ops produce outputs during execution
    // via classifyAndUpdateOwnership() in executeSegmentSlotBySlot().
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

#ifdef SD_CUDA
  // Clear MmulHelper's thread-local persistent cast cache between non-frozen executions.
  // During non-frozen execution (dynamic shapes), CUDA graph capture never happens, so
  // the persistent cast buffers are never reused. Without clearing, each matmul pushes
  // new persistent NDArrays with GPU memory (~1-2 MB each for weight matrix casts),
  // leaking ~225 MB per decoder step across ~150 matmul ops.
  if (!shapesFrozen_) {
    MmulHelper::clearCastCache();
  }
#endif

  // Reset dead-slot flags once per plan execution (not per segment).
  // Dead flags from Switch in one segment must persist to affect ops in later segments.
  if (hasControlFlow_ && slotIsDead_ != nullptr) {
    std::memset(slotIsDead_, 0, sizeof(bool) * slotIsDeadSize_);
  }

  // Timing instrumentation
  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Step 1b: Parallel precompilation of all GPU-compilable segments.
  // On GPU: fires async compilation threads for all eligible segments.
  // On CPU: no-op.
  platformPrecompileSegments(externalInputs, numExternalInputs);

  // Step 2: Execute segments
#ifdef SD_CUDA
  size_t poolUsedPreSegs = 0, poolReservedPreSegs = 0;
  sd::memory::CudaMemoryPool::getInstance().getStats(0, poolUsedPreSegs, poolReservedPreSegs);
  DSP_DIAG(MEMORY, "pre-segments: pool used=%zuMB reserved=%zuMB",
           poolUsedPreSegs / (1024*1024), poolReservedPreSegs / (1024*1024));

  // cuBLAS workspace: do NOT zero between frozen executions.  During capture,
  // the workspace accumulates cuBLAS plan/descriptor data as segments capture
  // sequentially.  Later segments' graphs omit H2D nodes for plans already
  // cached by earlier segments.  Zeroing would destroy those inherited plans,
  // causing GEMM kernels that lack H2D re-uploads to read zeros and hang.
  // Since shapes and buffer addresses are stable in frozen state, all cuBLAS
  // plans remain valid across executions — the workspace is self-consistent.
  if (shapesFrozen_ && executeCount_ > 0 && hasAnyReplayHandle &&
      cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
    DSP_DIAG(MEMORY, "pre-segments: cuBLAS workspace PRESERVED (%zuMB) — plans stable in frozen state",
             cublasWorkspaceSize_ / (1024*1024));
  }
#endif
  // One-time segment map dump on first frozen execution
  if (shapesFrozen_ && executeCount_ == 0) {
    DSP_DIAG(SEGMENT, "SEGMENT_MAP: %d segments (frozen first exec)", (int)segments_.size());
    for (int i = 0; i < (int)segments_.size(); i++) {
      auto& s = segments_[i];
      DSP_DIAG(SEGMENT, "  seg[%d]: slots[%d-%d] capturable=%d hasReplay=%d "
               "compilationFailed=%d execCount=%d",
               i, s.startSlot, s.endSlot, s.isCapturable,
               s.exec.replayHandle != nullptr, s.exec.compilationFailed, s.exec.executionCount);
    }
  }

  int segmentIdx = 0;
  long long graphReplayUs = 0, slotBySlotUs = 0;
  int graphReplaySegs = 0, slotBySlotSegs = 0, graphReplaySlots = 0, slotBySlotSlots = 0;
  for (auto& segment : segments_) {
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }

    // Migrate inputs that are on a different device than this segment's target
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    bool useGraph = platformShouldUseGraph(segment);

    // Set initial execution phase before dispatch
    if (segment.exec.executionCount == 0) {
      segment.exec.currentPhase = ExecutionPhase::WARMUP;
    }

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    bool segUsedGraph = false;
    int segSlots = segment.endSlot - segment.startSlot + 1;

    if (segment.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
      // Emulated graph replay: slot-by-slot with full replay lifecycle diagnostics
      auto status = executeSegmentEmulatedReplay(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
      // Emulated replay counts as "graph" for timing purposes (tracks replay overhead)
      segUsedGraph = true;
    } else if (useGraph) {
      // Platform dispatch: selected backend executes segment
      auto status = platformExecuteSegmentWithBackends(
          segment, externalInputs, numExternalInputs, stream, segUsedGraph);
      if (status != Status::OK) return status;
    } else {
      // No graph backend applicable — slot-by-slot execution
      segment.exec.currentPhase = segment.isCapturable
          ? ExecutionPhase::WARMUP    // Capturable but not yet ready for graph
          : ExecutionPhase::SLOT_BY_SLOT;  // Non-capturable, always slot-by-slot
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

    // Restore original arrays in outputSlots_ and delete migrated copies.
    // Must happen AFTER segment execution but BEFORE post-segment checks so
    // downstream segments see the original (source-device) arrays, not the
    // migrated copies which are about to be deleted.
    platformCleanupMigratedInputs();

    // Post-segment check: on GPU, detects sticky errors from async execution.
    // On CPU, always returns OK.
    auto postStatus = platformCheckPostSegment(segment);
    if (postStatus != Status::OK) return postStatus;

#ifdef SD_CUDA
    // Targeted slot trace: after every segment, check a specific slot's GPU data.
    // Set ND4J_DSP_TRACE_SLOT=<slotIndex> to enable (e.g., 299).
    // Reports: slot pointer, DataBuffer state, first 4 floats from GPU memory.
    {
      static int traceSlot = -1;
      static bool traceSlotInit = false;
      if (!traceSlotInit) {
        const char* env = std::getenv("ND4J_DSP_TRACE_SLOT");
        if (env != nullptr) traceSlot = std::atoi(env);
        traceSlotInit = true;
      }
      if (traceSlot >= 0 && traceSlot < totalOutputSlots_ && shapesFrozen_) {
        auto* arr = outputSlots_[traceSlot];
        if (arr != nullptr) {
          auto* db = arr->dataBuffer();
          void* gpuPtr = arr->specialBuffer();
          float firstVals[4] = {0, 0, 0, 0};
          if (gpuPtr != nullptr && arr->lengthOf() > 0 && arr->dataType() == FLOAT32) {
            int n = std::min((int)arr->lengthOf(), 4);
            // Raw cudaMemcpy — bypasses DataBuffer sync to avoid actuality counter corruption
            cudaStream_t execStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
            if (execStr != nullptr) cudaStreamSynchronize(execStr);
            cudaMemcpy(firstVals, gpuPtr, n * sizeof(float), cudaMemcpyDeviceToHost);
          }
          bool allZero = (firstVals[0] == 0.0f && firstVals[1] == 0.0f &&
                         firstVals[2] == 0.0f && firstVals[3] == 0.0f);
          bool hasNaN = (std::isnan(firstVals[0]) || std::isnan(firstVals[1]) ||
                        std::isnan(firstVals[2]) || std::isnan(firstVals[3]));
          // Always log when execCount_ > 0 to trace data transitions
          if (allZero || hasNaN || executeCount_ > 0) {
            const char* tag = hasNaN ? "NaN" : (allZero ? "ZERO" : "OK");
            DSP_DIAG(VERIFY, "SLOT_TRACE %s after seg[%d-%d]: slot=%d "
                    "arr=%p gpuPtr=%p db=%p closed=%d pAct=%d sAct=%d "
                    "vals=[%.6f,%.6f,%.6f,%.6f] execCount=%d",
                    tag,
                    segment.startSlot, segment.endSlot, traceSlot,
                    (void*)arr, gpuPtr, (void*)db,
                    db ? db->isClosed() : -1,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    firstVals[0], firstVals[1], firstVals[2], firstVals[3],
                    executeCount_);
          }
        } else {
          // Slot is null — log when this first happens
          static int lastNullSegEnd = -1;
          if (lastNullSegEnd != segment.endSlot) {
            DSP_DIAG(VERIFY, "SLOT_TRACE NULL after seg[%d-%d]: slot=%d "
                    "outputSlots_[%d]=nullptr cache=%p execCount=%d",
                    segment.startSlot, segment.endSlot, traceSlot, traceSlot,
                    (slotArrayCache_ && traceSlot < totalOutputSlots_) ?
                      (void*)slotArrayCache_[traceSlot] : nullptr,
                    executeCount_);
            lastNullSegEnd = segment.endSlot;
          }
        }
      }
    }
#endif

    // NaN detection: check output slots for NaN when verify mode is enabled.
    // GATED behind tritonVerifyKernels because syncToHost() on every output slot
    // in every segment causes thousands of GPU→CPU syncs per token (~4592 segments),
    // which is the single biggest performance bottleneck when left always-on.
    // Enable with: ND4J_TRITON_VERIFY_KERNELS=true
    if (shapesFrozen_ && Environment::getInstance().tritonVerifyKernels()) {
      for (int stepIdx = segment.startSlot; stepIdx <= segment.endSlot; stepIdx++) {
        auto& slot = slots_[stepIdx];
        for (int o = 0; o < slot.numOutputs; o++) {
          int si = slot.outputSlotIndices[o];
          if (si < 0 || si >= totalOutputSlots_ || outputSlots_[si] == nullptr) continue;
          auto* arr = outputSlots_[si];
          auto* db = arr->dataBuffer();
#if defined(SD_CUDA)
          if (db == nullptr || db->special() == nullptr || arr->lengthOf() == 0) continue;
#else
          if (db == nullptr || db->primary() == nullptr || arr->lengthOf() == 0) continue;
#endif
          bool dbClosed = db->isClosed();
          if (dbClosed) {
            fprintf(stdout, "[DSP_DIAG] [NaN_CLOSED_DB] seg[%d-%d] slot=%d opName=%s outSlot=%d "
                    "DataBuffer CLOSED! frozenConst=%d shapeStatic=%d execCount=%d\n",
                    segment.startSlot, segment.endSlot, stepIdx,
                    slot.opName.empty() ? "?" : slot.opName.c_str(), si,
                    slot.frozenConstantSlot() ? 1 : 0, slot.shapeStatic ? 1 : 0, executeCount_);
            fflush(stdout);
            continue;
          }
          // Check FULL array for NaN, not just element 0
          arr->syncToHost();
          bool hasNaN = arr->hasNaNs();
          if (hasNaN) {
            // Check inputs for this slot — full NaN check on each input
            bool anyInputNaN = false;
            for (int inp = 0; inp < slot.numInputs; inp++) {
              int srcIdx = slot.inputSourceIndices[inp];
              NDArray* srcArr = nullptr;
              if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
                srcArr = outputSlots_[srcIdx];
              } else if (srcIdx < 0) {
                int extIdx = -(srcIdx + 1);
                if (extIdx >= 0 && extIdx < numExternalInputs) srcArr = externalInputs[extIdx];
              }
              if (srcArr != nullptr && srcArr->lengthOf() > 0) {
                auto* srcDb = srcArr->dataBuffer();
                bool srcClosed = (srcDb != nullptr) ? srcDb->isClosed() : false;
                // Find which step produces this output slot for frozen/shapeStatic info
                int srcStepIdx = -1;
                bool srcFrozenConst = false, srcShapeStatic = false;
                if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
                  for (int ss = 0; ss < numSlots_; ss++) {
                    for (int oo = 0; oo < slots_[ss].numOutputs; oo++) {
                      if (slots_[ss].outputSlotIndices[oo] == srcIdx) {
                        srcStepIdx = ss;
                        srcFrozenConst = slots_[ss].frozenConstantSlot();
                        srcShapeStatic = slots_[ss].shapeStatic;
                        goto foundSrcStep;
                      }
                    }
                  }
                  foundSrcStep:;
                }
                srcArr->syncToHost();
                bool inpHasNaN = srcArr->hasNaNs();
                bool inpHasInf = !srcArr->isFinite();
                float inpVal = srcArr->e<float>(0);
                if (inpHasNaN) anyInputNaN = true;
                fprintf(stdout, "[DSP_DIAG] [NaN_TRACE] slot=%d input[%d] srcIdx=%d firstVal=%.6f "
                        "anyNaN=%d anyInf=%d addr=%p len=%lld dbClosed=%d "
                        "srcStep=%d srcFrozenConst=%d srcShapeStatic=%d\n",
                        stepIdx, inp, srcIdx, inpVal, inpHasNaN ? 1 : 0, inpHasInf ? 1 : 0,
                        srcArr->specialBuffer(), (long long)srcArr->lengthOf(), srcClosed ? 1 : 0,
                        srcStepIdx, srcFrozenConst ? 1 : 0, srcShapeStatic ? 1 : 0);
              } else {
                fprintf(stdout, "[DSP_DIAG] [NaN_TRACE] slot=%d input[%d] srcIdx=%d arr=%p (null or empty)\n",
                        stepIdx, inp, srcIdx, (void*)srcArr);
              }
            }
            fprintf(stdout, "[DSP_DIAG] [NaN_DETECT] seg[%d-%d] slot=%d opName=%s output[%d]=%d NaN! "
                    "useGraph=%d execCount=%d len=%lld addr=%p inputsNaN=%d hasReplay=%d "
                    "frozenConst=%d shapeStatic=%d\n",
                    segment.startSlot, segment.endSlot, stepIdx,
                    slot.opName.empty() ? "?" : slot.opName.c_str(), o, si,
                    useGraph ? 1 : 0, executeCount_, (long long)arr->lengthOf(),
                    arr->specialBuffer(), anyInputNaN ? 1 : 0,
                    segment.exec.replayHandle != nullptr ? 1 : 0,
                    slot.frozenConstantSlot() ? 1 : 0, slot.shapeStatic ? 1 : 0);
            fflush(stdout);
            goto nanDetectDone; // only report first NaN
          }
        }
      }
      nanDetectDone:;
    }

    segmentIdx++;
  }

  auto tSegsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

#ifdef SD_CUDA
  {
    size_t poolUsedPostSegs = 0, poolReservedPostSegs = 0;
    sd::memory::CudaMemoryPool::getInstance().getStats(0, poolUsedPostSegs, poolReservedPostSegs);
    long long deltaMB = static_cast<long long>(poolUsedPostSegs - poolUsedPreSegs) / (1024LL*1024);
    DSP_DIAG(MEMORY, "post-segments: pool used=%zuMB reserved=%zuMB (delta=%lldMB from pre-segs)",
             poolUsedPostSegs / (1024*1024), poolReservedPostSegs / (1024*1024), deltaMB);
  }

  // Periodically trim the async memory pool to release freed memory back to the
  // CUDA driver. Without this, cudaFreeAsync returns memory to the pool's reserved
  // set but does NOT return it to cudaFree-able driver memory. Over many decode
  // steps, the gap between pool-reserved and pool-used grows.
  // Trim on the first frozen execution and then every trimInterval steps (default 5).
  // The end-of-execution cudaStreamSynchronize (below) ensures all pending frees
  // complete before the next step, so we do NOT need a redundant sync here.
  if (shapesFrozen_) {
    int trimInterval = Environment::getInstance().dspTrimInterval();
    if (trimInterval > 0 && (executeCount_ == 0 || (executeCount_ % trimInterval) == 0)) {
      int trimDeviceId = 0;
      cudaGetDevice(&trimDeviceId);
      sd::memory::CudaMemoryPool::getInstance().trimPool(trimDeviceId);
      DSP_DIAG(MEMORY, "post-segments: trimmed pool on device %d (frozen exec=%d, interval=%d)",
               trimDeviceId, executeCount_, trimInterval);
    }
  }
#endif

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
             graphReplaySegs, slotBySlotSegs);

    // Verify slotArrayCache_ alias is still consistent with outputSlots_
    if (slotArrayCache_ != outputSlots_) {
      DSP_DIAG(VERIFY, "ERROR: slotArrayCache_ (%p) != outputSlots_ (%p) — alias broken!",
               (void*)slotArrayCache_, (void*)outputSlots_);
    }
  }

  // Step 3: Copy requested outputs
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      requestedOutputs[i] = outputSlots_[slotIdx];
    } else {
      requestedOutputs[i] = nullptr;
    }
  }

  // Diagnostic: dump requested output slot info and argmax for logits comparison
  if (DSP_DIAG_ENABLED(VERIFY) && executeCount_ <= 4) {
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      if (requestedOutputs[i] != nullptr) {
        auto* arr = requestedOutputs[i];
        DSP_DIAG_SLOT(VERIFY, slotIdx,
            "reqOut[%d] len=%lld dt=%d rank=%d",
            i, (long long)arr->lengthOf(), (int)arr->dataType(), arr->rankOf());
#ifdef SD_CUDA
        // For logits-sized outputs, find argmax on GPU side
        void* sbuf = arr->specialBuffer();
        if (sbuf && arr->dataType() == FLOAT32 && arr->lengthOf() >= 49280) {
          auto len = arr->lengthOf();
          std::vector<float> fullBuf(len);
          cudaMemcpy(fullBuf.data(), sbuf, len * sizeof(float), cudaMemcpyDeviceToHost);
          float maxVal = -1e30f;
          int maxIdx = -1;
          for (int j = 0; j < (int)len; j++) {
            if (fullBuf[j] > maxVal) { maxVal = fullBuf[j]; maxIdx = j; }
          }
          DSP_DIAG_SLOT(VERIFY, slotIdx,
              "logits maxIdx=%d maxVal=%.4f v@44=%.4f v@15539=%.4f",
              maxIdx, maxVal, fullBuf[44], fullBuf[15539]);
        }
#endif
      } else {
        DSP_DIAG_SLOT(VERIFY, slotIdx, "reqOut[%d] nullptr", i);
      }
    }
  }

  // Step 3.5: KV cache retention — C++ side scatters present KV into static buffers
  if (kvCacheRetentionEnabled_) {
    scatterKvEntries(externalInputs, numExternalInputs, stream);
    kvCachePosition_++;
  }

  auto tOutputsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Step 4: No flush needed — arrays persist (one array per slot)
  auto tFlushDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Track execution count for shapes-frozen optimization
  if (shapesFrozen_) executeCount_++;

  // Eager precompilation: after warmup (executeCount_ just became 1), all shapes
  // are populated in outputSlots_. Compile all Triton modules now so the 2nd
  // execute() goes straight to replay instead of blocking on compilation.
  if (shapesFrozen_ && executeCount_ == 1) {
    platformPrecompileSegments(externalInputs, numExternalInputs);
  }

  // Frozen constant detection (extracted to NativeDynamicShapePlan_slotexec.cpp)
  detectFrozenConstants();

#ifdef SD_CUDA
  // Batched GEMM group detection: after first shapes-frozen warmup, scan for
  // same-shape matmul slots that can be batched. Uses cachedOutputShapes from
  // the warmup pass (persists even after arrays are released).
  // Only detect when NOT using graph capture — batched GEMM replaces individual
  // matmul ops during slot-by-slot execution, not during graph replay.
  // Running detection + cudaMalloc during graph capture steps causes interference.
  if (shapesFrozen_ && executeCount_ == 1 && batchedGemmGroups_.empty() &&
      Environment::getInstance().dspBatchedGemm() &&
      !gpuGraphCaptureEnabled_) {
    detectBatchedGemmGroups(externalInputs, numExternalInputs);
    if (!batchedGemmGroups_.empty()) {
      cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
      prepareBatchedGemmDevice(execStream);
    }
  }
#endif

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
             graphReplayUs, graphReplaySegs, graphReplaySlots,
             slotBySlotUs, slotBySlotSegs, slotBySlotSlots);
  }

  DspDiagnostics::getInstance().endStep(executeCount_);

  // Cross-stream synchronization: the DSP execution stream must complete before
  // Java reads outputs on the default stream (stream 0) for argmax/sampling.
  // On frozen replay steps, use a lightweight CUDA event (~0.1ms) instead of full
  // cudaStreamSynchronize (~1.4ms). The event is recorded on the DSP stream and
  // the default stream waits on it before any output reads.
  // On non-frozen steps, use full sync for safety (shape transitions may free arrays).
#ifdef SD_CUDA
  if (stream != nullptr) {
    cudaStream_t cudaStr = *static_cast<cudaStream_t*>(stream);
    if (shapesFrozen_ && executeCount_ > 1) {
      // Lightweight event-based sync for frozen replay steps
      if (executionCompleteEvent_ == nullptr) {
        cudaEvent_t evt;
        cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        executionCompleteEvent_ = static_cast<void*>(new cudaEvent_t(evt));
      }
      cudaEvent_t evt = *static_cast<cudaEvent_t*>(executionCompleteEvent_);
      cudaEventRecord(evt, cudaStr);
      // Make the default stream (stream 0) wait for the DSP stream to finish
      cudaStreamWaitEvent(nullptr, evt, 0);
    } else {
      // Full sync for non-frozen steps (shape transitions, warmup, capture)
      cudaStreamSynchronize(cudaStr);
    }
  }
  // tl_dspExecutionStream is restored automatically by DspStreamGuard (dspStreamGuardPtr)
  // when it goes out of scope at function exit.
#endif

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
     << ",\"startSlot\":" << seg.startSlot
     << ",\"endSlot\":" << seg.endSlot
     << ",\"compiledByBackend\":\"" << seg.exec.compiledByBackend << "\""
     << ",\"capturable\":" << (seg.isCapturable ? "true" : "false")
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

// flushPendingClose REMOVED: arrays persist (one array per slot).
// View wrappers deleted inline in slotexec. No batched/deferred close needed.

void NativeDynamicShapePlan::clearShapeCaches() {
  // When shapes are frozen, skip clearing entirely after first execution.
  // All cached shapes remain valid since external input shapes are constant.
  if (shapesFrozen_ && executeCount_ > 0) return;

  for (int i = 0; i < numSlots_; i++) {
    if (!slots_[i].shapeStatic) {
      slots_[i].cachedShapeKey = 0;
      slots_[i].cachedOutputShapes.clear();
      // Demote to WARMUP if currently beyond warmup (non-static slots need re-inference)
      if (slots_[i].state_ > NativeSlot::SlotState::WARMUP) {
        slots_[i].state_ = NativeSlot::SlotState::WARMUP;
      }
    }
  }
}

void NativeDynamicShapePlan::clearAllShapeCachesForce() {
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].cachedShapeKey = 0;
    slots_[i].cachedOutputShapes.clear();
    // Force demote all slots to WARMUP
    if (slots_[i].state_ > NativeSlot::SlotState::WARMUP) {
      slots_[i].state_ = NativeSlot::SlotState::WARMUP;
    }
  }
}

// ─── Release GPU intermediates ───────────────────────────────────────────────

#ifdef SD_CUDA
// Diagnostic helper: report cudaMemGetInfo + pool stats at each step boundary.
// Always prints via sd_printf so we can trace the ~800 MB per-page leak.
static void logGpuMemState(const char* label) {
  size_t freeMem = 0, totalMem = 0;
  cudaMemGetInfo(&freeMem, &totalMem);
  size_t usedMem = totalMem - freeMem;

  // Query the default memory pool for current device
  cudaMemPool_t pool = nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  cudaDeviceGetDefaultMemPool(&pool, deviceId);

  uint64_t poolUsed = 0, poolReserved = 0;
  if (pool != nullptr) {
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &poolUsed);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &poolReserved);
  }

  // Also count direct allocations from CudaMemoryPool
  auto& memPool = memory::CudaMemoryPool::getInstance();
  size_t directAllocBytes = 0;
  {
    // Sum direct allocation sizes
    // We can't access directAllocations_ directly, but we know the weight migration
    // moves weights there. Use pinnedHostBytesUsed as a proxy for pinned host.
  }

  sd_printf("  [GPU-MEM %s] dev%d: used=%zu MB, free=%zu MB, total=%zu MB | "
            "pool: used=%llu MB, reserved=%llu MB, reclaimable=%llu MB\n",
            label, deviceId,
            usedMem / (1024*1024), freeMem / (1024*1024), totalMem / (1024*1024),
            poolUsed / (1024ULL*1024), poolReserved / (1024ULL*1024),
            (poolReserved - poolUsed) / (1024ULL*1024));
}
#endif

int NativeDynamicShapePlan::releaseGpuIntermediates() {
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: START plan=%p numSlots=%d totalOutputSlots=%d",
           this, numSlots_, totalOutputSlots_);

  // ── Step 1: Free per-segment GPU resources (CUDA graphs, capture buffers,
  //            capture workspaces, pinned host pointers) ──────────────────────
  // This is the same cleanup as the destructor's platformFreePlanResources(),
  // but we keep the segment metadata (slot ranges, op definitions) intact.
#ifdef SD_CUDA
  logGpuMemState("STEP-0-ENTRY");
  bool usePool = Environment::getInstance().dspCapturePoolEnabled() &&
                 captureBufferRegistry_ != nullptr;

  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      // Free capture buffer NDArrays
      for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
        if (!cb.directReference) delete cb.buffer;
      }
      seg.exec.replayHandle->getCaptureBuffers().clear();
      // Free capture workspace
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(
            usePool ? captureBufferRegistry_ : nullptr,
            seg.startSlot);
      }
      // Free pinned host pointers
      seg.exec.replayHandle->freeHostPointers();
      seg.exec.replayHandle->clearExternalAddresses();
      seg.exec.replayHandle.reset();
    }
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.argTableStable = false;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.compilationFailed = false;
    seg.exec.executionCount = 0;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
    // Reset stale state that would cause skipped recompilation/recapture:
    seg.exec.cachedShapeKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
    seg.exec.compiledByBackend.clear();
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
    seg.exec.jitShapeKey = 0;
    seg.exec.jitCompileFailed = false;
    seg.exec.segBatchZeroEntries.clear();
    // Reset compile-time shape key so Triton recompilation is triggered
    seg.shapeKey = 0;
  }

  // Release pool-managed capture buffers
  logGpuMemState("STEP-1-AFTER-SEGMENTS");
  if (usePool) {
    auto* registry = static_cast<CaptureBufferRegistry*>(captureBufferRegistry_);
    registry->releaseAll();
    delete registry;
    captureBufferRegistry_ = nullptr;
  }

  // Free cuBLAS workspace (256 MB)
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }

  // Free batch-zero, batch-D2D, and batched-GEMM device arrays
  freeBatchZeroResources();
  freeBatchD2DResources();
  freeBatchedGemmResources();
  logGpuMemState("STEP-1-AFTER-BATCH-RESOURCES");
#else
  // CPU path: reset segment execution state (no CUDA graphs or GPU resources)
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      seg.exec.replayHandle.reset();
    }
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.argTableStable = false;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.compilationFailed = false;
    seg.exec.executionCount = 0;
    // Reset stale state that would cause skipped recompilation/recapture:
    seg.exec.cachedShapeKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
    seg.exec.compiledByBackend.clear();
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
    seg.exec.segBatchZeroEntries.clear();
    seg.shapeKey = 0;
  }
#endif

  // ── Step 2: Free non-weight NDArrays from outputSlots_ ─────────────────
  // Only free SLOT_OWNED buffers. Views, weights, and capture buffers are
  // either already freed (capture buffers above) or externally owned.
  //
  // CRITICAL: Re-classify ownership before freeing. After CUDA graph capture,
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
    // This corrects stale ownership from CUDA graph capture-time classification.
    if (slotOwnership_) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        slotOwnership_[i].reset();
        if (outputSlots_[i] == nullptr) continue;
        auto* db = outputSlots_[i]->dataBuffer();
        if (db == nullptr) {
          slotOwnership_[i].ownership = BufferOwnership::UNSET;
          continue;
        }
        // Check if buffer belongs to a protected weight
        if (protectedWeightBuffers_.count(db) > 0) {
          slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_WEIGHT;
          slotOwnership_[i].dataBuffer = db;
          continue;
        }
        // Check if buffer is shared with an earlier slot (view of another slot)
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
          // Also check if a LATER slot shares this buffer (this slot is the owner)
          slotOwnership_[i].ownership = BufferOwnership::SLOT_OWNED;
          slotOwnership_[i].dataBuffer = db;
        }
      }
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: re-classified ownership for %d slots", totalOutputSlots_);
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
      // Second pass: free SLOT_OWNED buffers. Force viewRefCount to 0 since
      // all views were cleared in the first pass — no live references remain.
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr &&
            slotOwnership_[i].ownership == BufferOwnership::SLOT_OWNED) {
          slotOwnership_[i].viewRefCount = 0;  // All views already nulled above
          if (deleted.insert(outputSlots_[i]).second) {
            delete outputSlots_[i];
            freedCount++;
          }
          outputSlots_[i] = nullptr;
          slotOwnership_[i].reset();
        }
      }
    } else {
      // Fallback: no ownership info — use protectedWeightBuffers_ to decide
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr) {
          auto* db = outputSlots_[i]->dataBuffer();
          bool isWeight = (db != nullptr && protectedWeightBuffers_.count(db) > 0);
          if (!isWeight && deleted.insert(outputSlots_[i]).second) {
            delete outputSlots_[i];
            freedCount++;
          }
          if (!isWeight) {
            outputSlots_[i] = nullptr;
          }
        }
      }
    }
  }
#ifdef SD_CUDA
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: freed %d unique intermediate NDArrays", freedCount);
  sd_printf("  [GPU-MEM] Step 2: freed %d unique intermediate NDArrays\n", freedCount);
  logGpuMemState("STEP-2-AFTER-INTERMEDIATES");

  // ── Step 3: Free untracked output cache ─────────────────────────────────
  if (untrackedOutputCache_) {
    for (int i = 0; i < untrackedOutputCacheSize_; i++) {
      delete untrackedOutputCache_[i];
      untrackedOutputCache_[i] = nullptr;
    }
  }

  // ── Step 4: Clear MmulHelper cast cache (thread-local FP16→FP32 staging) ─
  MmulHelper::clearCastCache();
  logGpuMemState("STEP-4-AFTER-CAST-CACHE");

  // ── Step 4b: Migrate weight buffers out of async pool, then trim ────────
  // The CUDA async memory pool uses a single default pool per device for BOTH
  // model weights (long-lived) and intermediates (short-lived). When intermediates
  // are freed, weight allocations scattered across pool blocks prevent
  // cudaMemPoolTrimTo from reclaiming the freed memory. This causes a progressive
  // ~800-1200 MB leak per page cycle.
  //
  // Fix: After freeing intermediates and syncing, migrate each weight DataBuffer
  // from its pool allocation (cudaMallocAsync) to a direct allocation (cudaMalloc).
  // This removes all weight pointers from the pool, allowing trimPool to fully
  // reclaim all freed intermediate memory. The migration is a one-time cost per
  // reset (~2-4 GB memcpy) that eliminates the fragmentation root cause.
  {
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: cudaDeviceSynchronize failed: %s",
               cudaGetErrorString(syncErr));
      cudaGetLastError();  // clear sticky error
    }
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    // Migrate weight buffers from async pool to direct cudaMalloc
    int migratedCount = 0;
    int skippedDirect = 0;
    int skippedNonDevice = 0;
    int failedMigrations = 0;
    size_t migratedBytes = 0;
    size_t totalWeightBytes = 0;
    auto& pool = memory::CudaMemoryPool::getInstance();

    sd_printf("  [GPU-MEM] Weight migration: %zu protected weight buffers to check\n",
              protectedWeightBuffers_.size());

    for (auto* db : protectedWeightBuffers_) {
      if (db == nullptr || db->special() == nullptr) continue;

      size_t bufSize = db->getLenInBytes();
      totalWeightBytes += bufSize;

      // Skip buffers that are already direct allocations (from a previous migration)
      if (pool.isDirectAllocation(db->special())) {
        skippedDirect++;
        continue;
      }

      // Skip host (pinned) allocations — these aren't in the pool
      // (detected by checking if the pointer is in hostAllocations_)
      cudaPointerAttributes ptrAttrs;
      cudaError_t attrErr = cudaPointerGetAttributes(&ptrAttrs, db->special());
      if (attrErr != cudaSuccess) {
        cudaGetLastError();
        continue;  // Can't query pointer — skip
      }
      if (ptrAttrs.type != cudaMemoryTypeDevice) {
        skippedNonDevice++;
        continue;  // Not device memory (host/managed/unregistered) — skip
      }

      if (bufSize == 0) continue;

      // Allocate a new direct (non-pool) buffer via cudaMalloc
      void* directPtr = nullptr;
      cudaError_t allocErr = cudaMalloc(&directPtr, bufSize);
      if (allocErr != cudaSuccess || directPtr == nullptr) {
        // Can't allocate — skip this buffer (it stays in the pool)
        cudaGetLastError();
        failedMigrations++;
        sd_printf("  [GPU-MEM] Weight migration FAILED for %zu bytes (%zu MB): %s\n",
                  bufSize, bufSize / (1024*1024), cudaGetErrorString(allocErr));
        continue;
      }

      // Copy weight data from pool buffer to direct buffer
      cudaError_t copyErr = cudaMemcpy(directPtr, db->special(), bufSize, cudaMemcpyDeviceToDevice);
      if (copyErr != cudaSuccess) {
        // Copy failed — free the new buffer and skip
        cudaFree(directPtr);
        cudaGetLastError();
        DSP_DIAG(MEMORY, "releaseGpuIntermediates: weight migration memcpy failed for %zu bytes: %s",
                 bufSize, cudaGetErrorString(copyErr));
        continue;
      }

      // Free the old pool-based buffer via cudaFreeAsync (returns memory to pool)
      void* oldPtr = db->special();
      cudaFreeAsync(oldPtr, nullptr);

      // Update the DataBuffer to point to the new direct buffer.
      // replaceSpecialBuffer swaps _specialBuffer without calling deleteSpecial(),
      // which would try to free the already-freed old pointer.
      db->replaceSpecialBuffer(directPtr, true);

      // Register the new pointer as a direct allocation so CudaMemoryPool::free()
      // routes it to cudaFree instead of cudaFreeAsync
      pool.registerDirectAllocation(directPtr, bufSize);

      migratedCount++;
      migratedBytes += bufSize;
    }

    if (migratedCount > 0) {
      // Sync again to ensure all cudaFreeAsync from old pool buffers complete
      cudaDeviceSynchronize();
    }

    sd_printf("  [GPU-MEM] Weight migration summary: total=%zu MB, migrated=%d (%zu MB), "
              "skippedDirect=%d, skippedNonDevice=%d, failed=%d\n",
              totalWeightBytes / (1024*1024), migratedCount, migratedBytes / (1024*1024),
              skippedDirect, skippedNonDevice, failedMigrations);

    pool.trimPool(deviceId);
    logGpuMemState("STEP-4b-AFTER-MIGRATION-AND-TRIM");

    // ── Step 4c: Clear shape and TAD caches ────────────────────────────────
    // DirectShapeTrie and DirectTadTrie permanently cache every unique
    // (shape+strides+dtype+order) and (shape+TAD-dimensions) combination.
    // In the VLM decoder, seqKV increments by 1 each decode step, creating
    // ~hundreds of unique shapes per page. Each entry allocates GPU memory
    // via replicatePointer() → CudaMemoryPool::allocate(). Neither cache
    // ever evicts entries — they grow monotonically, accounting for ~600-700 MB
    // of non-reclaimable pool memory per page cycle.
    //
    // Fix: Clear both caches between pages. The entries will be recreated
    // on-demand during the next page's execution. The GPU memory freed here
    // returns to the async pool, which we then reclaim via a second trimPool().
    {
      auto shapeEntriesBefore = ConstantShapeHelper::getInstance().getCachedEntries();
      auto tadEntriesBefore = ConstantTadHelper::getInstance().getCachedEntries();

      ConstantShapeHelper::getInstance().clearCache();
      ConstantTadHelper::getInstance().clearCache();

      auto shapeEntriesAfter = ConstantShapeHelper::getInstance().getCachedEntries();
      auto tadEntriesAfter = ConstantTadHelper::getInstance().getCachedEntries();

      sd_printf("  [GPU-MEM] Shape/TAD cache clear: shapes %lld->%lld, TADs %lld->%lld\n",
                static_cast<long long>(shapeEntriesBefore), static_cast<long long>(shapeEntriesAfter),
                static_cast<long long>(tadEntriesBefore), static_cast<long long>(tadEntriesAfter));

      // Sync so all cudaFreeAsync calls from cache clearing complete
      cudaDeviceSynchronize();
      // Trim again to reclaim pool memory freed by cache clearing
      pool.trimPool(deviceId);
      logGpuMemState("STEP-4c-AFTER-CACHE-CLEAR-AND-TRIM");
    }
  }
#endif

  // ── Step 5: Reset execution state so plan re-warms on next execute() ────
  viewProducerDetectionDone_ = false;
  frozenConstantDetectionDone_ = false;
  executeCount_ = 0;
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

  // Clear Triton compiled kernel cache (singleton) to free CUmodule GPU memory.
  // The Triton cache accumulates ~100-150MB/page of CUmodule handles because
  // cache keys include shapeKey which changes per page. Kernels re-load from
  // disk cache in <100ms, so the re-compilation cost is minimal.
#if HAVE_TRITON && defined(SD_CUDA)
  {
    std::vector<std::pair<int,int>> segRanges;
    segRanges.reserve(segments_.size());
    for (auto& seg : segments_) {
      segRanges.emplace_back(seg.startSlot, seg.endSlot);
    }
    if (!segRanges.empty()) {
      TritonGraphBackend::getInstance().invalidateCacheForSegments(segRanges);
    }
  }
#endif

  DSP_DIAG(MEMORY, "releaseGpuIntermediates: DONE plan=%p, freed %d arrays. "
           "Plan is now cold — next execute() will re-warm.", this, freedCount);

  return freedCount;
}

int NativeDynamicShapePlan::releaseGpuIntermediates(bool preserveDecodeState) {
  if (preserveDecodeState) {
    // Decode-invariant path: preserve staging buffers, slot arrays, CUDA graphs,
    // cuBLAS workspace, and batch optimization resources. Only reset KV cache
    // position and decode input pending state.
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

    // Mark capture buffers for KV inputs as "always copy" — their data changes
    // each step via kvScatter even though the GPU pointer stays the same.
    // No-op on CPU builds (no capture buffers).
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
#ifdef SD_CUDA
  cudaStream_t cudaStr = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);

  DSP_DIAG(EXECUTE, "updateDecodeInputs: ENTER tokenId=%lld cachePos=%d numExt=%d idsIdx=%d posIdx=%d maskIdx=%d",
           tokenId, cachePos, numExt, decodeInputIdsExtIdx_, decodePositionIdsExtIdx_, decodeAttentionMaskExtIdx_);

  // input_ids[0] = tokenId
  if (decodeInputIdsExtIdx_ >= 0 && decodeInputIdsExtIdx_ < numExt) {
    NDArray* ids = externalInputs[decodeInputIdsExtIdx_];
    DSP_DIAG(EXECUTE, "updateDecodeInputs: ids NDArray=%p specialBuf=%p len=%lld",
             ids, ids ? ids->specialBuffer() : nullptr, ids ? (long long)ids->lengthOf() : -1);
    if (ids != nullptr && ids->specialBuffer() != nullptr) {
      LongType val = static_cast<LongType>(tokenId);
      cudaMemcpyAsync(ids->specialBuffer(), &val, sizeof(LongType),
                      cudaMemcpyHostToDevice, cudaStr);
      ids->dataBuffer()->writeSpecial();
    }
  }

  // position_ids[0] = cachePos
  if (decodePositionIdsExtIdx_ >= 0 && decodePositionIdsExtIdx_ < numExt) {
    NDArray* pos = externalInputs[decodePositionIdsExtIdx_];
    DSP_DIAG(EXECUTE, "updateDecodeInputs: pos NDArray=%p specialBuf=%p len=%lld",
             pos, pos ? pos->specialBuffer() : nullptr, pos ? (long long)pos->lengthOf() : -1);
    if (pos != nullptr && pos->specialBuffer() != nullptr) {
      LongType val = static_cast<LongType>(cachePos);
      cudaMemcpyAsync(pos->specialBuffer(), &val, sizeof(LongType),
                      cudaMemcpyHostToDevice, cudaStr);
      pos->dataBuffer()->writeSpecial();
    }
  }

  // attention_mask[cachePos - 1] = 1  (unmask the position filled by the PREVIOUS step's scatter)
  // cachePos is the NEXT write position — not yet filled. The position just filled is cachePos - 1.
  // Positions 0..cachePos-1 are valid; cachePos itself will be filled AFTER this forward pass.
  if (decodeAttentionMaskExtIdx_ >= 0 && decodeAttentionMaskExtIdx_ < numExt && cachePos > 0) {
    NDArray* mask = externalInputs[decodeAttentionMaskExtIdx_];
    int writePos = cachePos - 1;
    DSP_DIAG(EXECUTE, "updateDecodeInputs: mask NDArray=%p specialBuf=%p len=%lld cachePos=%d writePos=%d",
             mask, mask ? mask->specialBuffer() : nullptr, mask ? (long long)mask->lengthOf() : -1,
             cachePos, writePos);
    if (mask != nullptr && mask->specialBuffer() != nullptr) {
      LongType one = 1;
      auto maskLen = mask->lengthOf();
      if (writePos < maskLen) {
        auto* dst = static_cast<LongType*>(mask->specialBuffer()) + writePos;
        DSP_DIAG(EXECUTE, "updateDecodeInputs: mask dst=%p (base=%p + %d * %d)",
                 dst, mask->specialBuffer(), writePos, (int)sizeof(LongType));
        cudaMemcpyAsync(dst, &one, sizeof(LongType),
                        cudaMemcpyHostToDevice, cudaStr);
        mask->dataBuffer()->writeSpecial();
      } else {
        DSP_DIAG(EXECUTE, "updateDecodeInputs: SKIP attn_mask write writePos=%d maskLen=%lld (OOB)",
                 writePos, (long long)maskLen);
      }
    }
  }

  DSP_DIAG(EXECUTE, "updateDecodeInputs: tokenId=%lld cachePos=%d", tokenId, cachePos);
#endif
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

void NativeDynamicShapePlan::setExternalInputMaxSizes(const int* extIndices, const LongType* maxSizes, int numInputs) {
  if (extIndices == nullptr || maxSizes == nullptr || numInputs <= 0) return;
  if (externalInputMaxSizes_.empty()) {
    externalInputMaxSizes_.resize(numExternalInputs_, 0);
    externalInputUseStaging_.resize(numExternalInputs_, false);
  }

  for (int i = 0; i < numInputs; i++) {
    if (extIndices[i] >= 0 && extIndices[i] < numExternalInputs_ && maxSizes[i] > 0) {
      externalInputMaxSizes_[extIndices[i]] = maxSizes[i];
      externalInputUseStaging_[extIndices[i]] = true;
    }
  }

  // Staging buffer allocation deferred - requires proper NDArray allocation API
  // For now, just track the max sizes for future use
  DSP_DIAG(MEMORY, "setExternalInputMaxSizes: configured %d inputs for staging (allocation deferred)", numInputs);
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
  DataType batchDtype = DataType::HALF;  // default; set from first valid entry

  int skipped = 0;
  for (int m = 0; m < kvCacheNumMappings_; m++) {
    KvCacheMapping& mapping = kvCacheMappings_[m];

    int presentSlotIdx = mapping.presentOutputSlotIdx;
    if (presentSlotIdx < 0 || presentSlotIdx >= totalOutputSlots_) { skipped++; continue; }

    NDArray* presentKv = resolveLiveArray(outputSlots_[presentSlotIdx]);
    if (presentKv == nullptr) {
      presentKv = resolveLiveArray(slotArrayCache_[presentSlotIdx]);
    } else if (slotArrayCache_[presentSlotIdx] != presentKv) {
      slotArrayCache_[presentSlotIdx] = presentKv;
    }

    if (presentKv == nullptr) { skipped++; continue; }

    int extIdx = mapping.pastInputExternalIdx;
    if (extIdx < 0 || extIdx >= numExt) { skipped++; continue; }
    NDArray* staticBuf = externalInputs[extIdx];
    if (staticBuf == nullptr) { skipped++; continue; }

    if (presentKv->rankOf() != 4 || staticBuf->rankOf() != 4) { skipped++; continue; }

    // Validate GPU buffer pointers are non-null and sequence dims are non-zero.
    // After resetForNextPage(), restored cached handles may encounter empty
    // destination buffers (shape [1,H,0,D]) whose specialBuffer() is nullptr.
    // Writing to nullptr causes CUDA error 700.
    NDArray::prepareSpecialUse({staticBuf}, {presentKv});

    const void* srcBuf = presentKv->specialBuffer();
    void* dstBuf = staticBuf->specialBuffer();
    auto srcSeq = presentKv->sizeAt(2);
    auto dstSeq = staticBuf->sizeAt(2);

    if (srcBuf == nullptr || dstBuf == nullptr || srcSeq <= 0 || dstSeq <= 0) {
      NDArray::registerSpecialUse({staticBuf}, {presentKv}); // balance the prepareSpecialUse
      skipped++;
      continue;
    }

    // Validate cachePos is within the destination buffer's bounds
    if (kvCachePosition_ >= dstSeq) {
      NDArray::registerSpecialUse({staticBuf}, {presentKv}); // balance the prepareSpecialUse
      skipped++;
      continue;
    }

    sd::ops::helpers::KvScatterEntry entry;
    entry.srcPtr = srcBuf;
    entry.dstPtr = dstBuf;
    entry.heads = presentKv->sizeAt(1);
    entry.srcSeqLen = srcSeq;
    entry.dstSeqLen = dstSeq;
    entry.dim = presentKv->sizeAt(3);
    entry.lastPos = entry.srcSeqLen - 1;
    entry.cachePos = kvCachePosition_;
    batchEntries.push_back(entry);
    if (scatterPairs.empty()) {
      batchDtype = presentKv->dataType();
    }
    scatterPairs.push_back({presentKv, staticBuf});
  }

  int scattered = static_cast<int>(batchEntries.size());
  if (scattered > 0) {
    auto* lc = LaunchContext::defaultContext();
    sd::ops::helpers::kvScatterBatched(batchEntries.data(), scattered, batchDtype, lc);

#ifdef SD_CUDA
    // Diagnostic: sync after KV scatter to catch latent errors from scatter kernel
    {
      cudaError_t scatterErr = cudaDeviceSynchronize();
      if (scatterErr != cudaSuccess) {
        sd_printf("KV SCATTER CUDA ERROR: cudaDeviceSynchronize after kvScatterBatched "
                  "returned error %d (%s). scattered=%d pos=%d numMappings=%d\n",
                  static_cast<int>(scatterErr), cudaGetErrorString(scatterErr),
                  scattered, kvCachePosition_, kvCacheNumMappings_);
        // Log first entry pointers for debugging
        if (!batchEntries.empty()) {
          auto& e = batchEntries[0];
          sd_printf("  entry[0]: srcPtr=%p dstPtr=%p heads=%lld srcSeqLen=%lld dstSeqLen=%lld cachePos=%lld\n",
                    e.srcPtr, e.dstPtr, (long long)e.heads, (long long)e.srcSeqLen,
                    (long long)e.dstSeqLen, (long long)e.cachePos);
        }
        cudaGetLastError(); // clear sticky error
      }
    }
#endif
  }

  // Register special use for all pairs
  for (auto& pair : scatterPairs) {
    NDArray::registerSpecialUse({pair.second}, {pair.first});
  }

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
      NDArray* fromCache = (psi >= 0 && psi < totalOutputSlots_) ? slotArrayCache_[psi] : nullptr;
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
#ifdef SD_CUDA
      return SelectedBackend::CUDA_GRAPHS;
#else
      return SelectedBackend::SLOT_BY_SLOT;
#endif

    case GraphExecutionMode::GEM_TRITON:
    case GraphExecutionMode::GEM_NVRTC_JIT:
    case GraphExecutionMode::GEM_PTX_JIT:
    case GraphExecutionMode::GEM_TPU:
    case GraphExecutionMode::GEM_HEXAGON:
#ifdef SD_CUDA
      return SelectedBackend::GPU_COMPILER;
#else
      return SelectedBackend::CPU_GRAPH;
#endif

    case GraphExecutionMode::GEM_MLX:
    case GraphExecutionMode::GEM_ARM_HYBRID:
    case GraphExecutionMode::GEM_NNAPI:
      return SelectedBackend::CPU_GRAPH;

    case GraphExecutionMode::GEM_EMULATED_REPLAY:
      return SelectedBackend::EMULATED_REPLAY;

    case GraphExecutionMode::GEM_AUTO: {
      // Resolve best available backend. Check order: GPU compiler → CUDA graphs → CPU graph → slot-by-slot
      // GPU compiler is checked lazily by getGpuGraphBackend() on first execution.
      // At build time we can check if CUDA graphs are enabled as a strong signal.
#ifdef SD_CUDA
      // For AUTO, we prefer GPU_COMPILER (Triton/NVRTC/PTX). The actual backend
      // is resolved lazily by getGpuGraphBackend(). If it returns nullptr at
      // execution time, we fall back to CUDA_GRAPHS if enabled.
      // We can't fully resolve at build time because backend availability may
      // depend on runtime state. Mark as GPU_COMPILER optimistically — the
      // dispatcher will handle nullptr gpuBackend gracefully.
      return SelectedBackend::GPU_COMPILER;
#else
      return SelectedBackend::CPU_GRAPH;
#endif
    }

    default:
      return SelectedBackend::SLOT_BY_SLOT;
  }
}

// ─── Graph segmentation for GPU graph capture ───────────────────────────────

void NativeDynamicShapePlan::buildSegments() {
  if (numSlots_ == 0) return;

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

  auto isSlotCapturable = [](const NativeSlot& slot) -> bool {
    // Control flow ops are never capturable — execution path is data-dependent
    if (slot.controlFlowType != CF_NONE) return false;
    if (slot.isDataDependent) return false;
    // Value-dependent-shape ops (broadcast_to, reshape, etc.) are now capturable
    // because computeSegmentShapeKey hashes actual DATA VALUES of small inputs
    // (≤32 elements). If a shape-determining input value changes between steps,
    // the shape key changes → graph is invalidated → re-captured with correct shape.
    // No hardcoded op list needed.
    return true;
  };

  GraphSegment current;
  current.startSlot = 0;
  current.isCapturable = isSlotCapturable(slots_[0]);

  for (int i = 1; i < numSlots_; i++) {
    bool thisCapturable = isSlotCapturable(slots_[i]);
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);

    bool capturabilityChanged = (thisCapturable != current.isCapturable);

    if (capturabilityChanged || deviceChange) {
      // End current segment
      current.endSlot = i - 1;
      segments_.push_back(std::move(current));

      // Start new segment
      current = GraphSegment();
      current.startSlot = i;
      current.isCapturable = thisCapturable;
    }
  }

  // Finalize last segment
  current.endSlot = numSlots_ - 1;
  segments_.push_back(std::move(current));

  // Log segment structure
  int capturableCount = 0, totalCapturable = 0;
  int staticCapturableCount = 0, dynamicCapturableCount = 0;
  for (auto& seg : segments_) {
    if (seg.isCapturable) {
      capturableCount++;
      int sz = seg.endSlot - seg.startSlot + 1;
      totalCapturable += sz;
      // A segment is "static" if all its slots have stable shapes
      bool allStatic = true;
      for (int s = seg.startSlot; s <= seg.endSlot && allStatic; s++)
        allStatic = slots_[s].shapeStatic;
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
    if (seg.startSlot >= 0 && seg.startSlot < numSlots_) {
      targetDevice = slots_[seg.startSlot].targetDeviceId;
    }
    DSP_DIAG_SEG(SEGMENT, i, "segment[%d] [%d-%d] capturable=%d targetDeviceId=%d",
                 i, seg.startSlot, seg.endSlot, static_cast<int>(seg.isCapturable), targetDevice);
  }
  if ((int)segments_.size() > maxLoggedSegments) {
    DSP_DIAG(SEGMENT, "... %d additional segments not shown in device map",
             static_cast<int>(segments_.size()) - maxLoggedSegments);
  }

  // Propagate slotArrayCache_, resolve backend, and detect value-dep ops for all segments.
  for (auto& seg : segments_) {
    seg.slotArrayCache = slotArrayCache_;
    seg.selectedBackend = resolveBackendForSegment(seg.isCapturable);
    // Scan slots for value-dependent ops — these require shape key recomputation
    // even when shapes are frozen, because input VALUES (not just shapes) affect output shape.
    seg.hasValueDepOps = false;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      if (slots_[s].outputShapeDependsOnInputValues) {
        seg.hasValueDepOps = true;
        break;
      }
    }
    DSP_DIAG_SEG(SEGMENT, seg.startSlot, "segment[%d-%d] selectedBackend=%d hasValueDepOps=%d",
                 seg.startSlot, seg.endSlot, static_cast<int>(seg.selectedBackend),
                 seg.hasValueDepOps ? 1 : 0);
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

#ifdef SD_CUDA
  // Initialize capture buffer pool registry if enabled
  if (Environment::getInstance().dspCapturePoolEnabled() && captureBufferRegistry_ == nullptr) {
    captureBufferRegistry_ = new CaptureBufferRegistry();
    DSP_DIAG(MEMORY, "CaptureBufferRegistry: initialized for plan with %d segments",
             static_cast<int>(segments_.size()));
  }
#endif
}

// ─── Rebuild segments for frozen shapes ───────────────────────────────────────
//
// When shapes are frozen, value-dependent-shape ops (reshape, gather, slice, etc.)
// are safe to capture — their input values never change, so output shapes are constant.
// Only truly data-dependent ops (where/unique/NMS with variable-length output)
// remain non-capturable.
//
// This merges ALL consecutive slots into a single segment, breaking only on:
//   1. Data-dependent ops (variable-length output)
//   2. Device boundaries
//
void NativeDynamicShapePlan::rebuildSegmentsForFrozenShapes() {
  DSP_DIAG(SEGMENT, "rebuildSegmentsForFrozenShapes: ENTER numSlots=%d oldSegments=%d executeCount=%d shapesFrozen=%d",
           numSlots_, (int)segments_.size(), executeCount_, shapesFrozen_ ? 1 : 0);

  // Destroy existing cached graphs (they reference old segment boundaries)
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      DSP_DIAG_SEG(SEGMENT, seg.startSlot,
                   "destroying replay handle for seg[%d-%d] state=%d replays=%d",
                   seg.startSlot, seg.endSlot, (int)seg.exec.replayHandle->getState(),
                   seg.exec.replayHandle->getStatistics().replayCount);
    }
    platformCleanupSegmentForRebuild(seg);
  }

  int oldSegCount = (int)segments_.size();
  segments_.clear();

  if (numSlots_ == 0) return;

  // When shapes are frozen, value-dependent-shape ops (reshape, gather, slice, etc.)
  // are safe to capture — their input values never change, so output shapes are constant.
  // Data-dependent ops (1-input Where/Unique/NMS) remain non-capturable because they
  // have variable-length output. 3-input Where (element-wise select) IS capturable.
  // Break segments on: data-dependent ops, device boundaries, and max size limit.
  //
  // Max segment size prevents mega-graphs that cause:
  //   1. Address instability (workspace allocs change address on replay → SIGSEGV)
  //   2. Slow replay (390ms for 3781-slot graph vs 72ms for 129 smaller graphs)
  // Cap at ~150 slots per segment — matches typical transformer layer size.
  // MAX_FROZEN_SEGMENT_SIZE caps graph size. Set to INT_MAX to allow single
  // mega-graph capture of the entire decoder (3781+ slots).
  // With frozen shapes, all value-dependent ops are safe to capture.
  // NOTE: create (ConstantOfShape) ops are captured but their input values are
  // hashed and validated before replay (see computeCreateOpValueKey).
  // If values change → graph is invalidated and re-captured.
  static constexpr int MAX_FROZEN_SEGMENT_SIZE = INT_MAX;

  auto isSlotCapturableFrozen = [this](int idx) -> bool {
    return !slots_[idx].isDataDependent && slots_[idx].controlFlowType == CF_NONE;
  };

  // Matmul segmentation: when enabled, break segments at matmul/attention ops.
  // This creates separate segments for element-wise chains between matmuls,
  // allowing Triton to fuse each chain into a single kernel (like pytorch.compile).
  // Matmul ops themselves run via GPU BLAS fallback within their own tiny segments.
  const bool matmulSegmentation = Environment::getInstance().dspMatmulSegmentation();

  auto isMatmulOrAttention = [this](int idx) -> bool {
    const std::string& name = slots_[idx].opName;
    return name == "matmul" || name == "mmul" || name == "batched_gemm"
        || name == "tensormmul" || name == "fp8_matmul" || name == "smooth_quant"
        || name == "awq_matmul" || name == "column_parallel_linear"
        || name == "row_parallel_linear" || name == "multi_lora_matmul"
        || name == "fused_gemm_swiglu" || name == "multi_head_attention";
  };

  GraphSegment current;
  current.startSlot = 0;
  current.isCapturable = isSlotCapturableFrozen(0);

  for (int i = 1; i < numSlots_; i++) {
    bool thisCapturable = isSlotCapturableFrozen(i);
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);
    int currentSize = i - current.startSlot;
    bool sizeLimit = (current.isCapturable && currentSize >= MAX_FROZEN_SEGMENT_SIZE);

    // When matmul segmentation is enabled, break before and after matmul/attention ops.
    // This isolates element-wise chains for Triton fusion while matmuls run via GPU BLAS.
    bool matmulBreak = false;
    if (matmulSegmentation) {
      bool thisIsMatmul = isMatmulOrAttention(i);
      bool prevIsMatmul = isMatmulOrAttention(i - 1);
      // Break when transitioning from non-matmul to matmul or matmul to non-matmul
      if (thisIsMatmul != prevIsMatmul) matmulBreak = true;
    }

    if (thisCapturable != current.isCapturable || deviceChange || sizeLimit || matmulBreak) {
      current.endSlot = i - 1;
      segments_.push_back(std::move(current));
      current = GraphSegment();
      current.startSlot = i;
      current.isCapturable = thisCapturable;
    }
  }
  current.endSlot = numSlots_ - 1;
  segments_.push_back(std::move(current));

  // Reset frozen context for all slots since segment boundaries changed.
  // Demote any slot that was FROZEN or FROZEN_CONSTANT back to SHAPE_CACHED
  // (shape cache is still valid, but frozen context must be rebuilt).
  for (int i = 0; i < numSlots_; i++) {
    if (slots_[i].state_ >= NativeSlot::SlotState::FROZEN) {
      slots_[i].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }
  }

  // Log the result and any data-dependent ops that prevent full merge
  int capturableSlots = 0;
  int dataDepCount = 0;
  for (auto& seg : segments_) {
    if (seg.isCapturable) capturableSlots += (seg.endSlot - seg.startSlot + 1);
  }
  for (int i = 0; i < numSlots_; i++) {
    if (slots_[i].isDataDependent) {
      dataDepCount++;
      if (dataDepCount <= 10) {
        DSP_DIAG_SLOT(SEGMENT, i, "slot %d op='%s' is data-dependent",
                      i, slots_[i].opName.c_str());
      }
    }
  }
  if (dataDepCount > 10) {
    DSP_DIAG(SEGMENT, "... and %d more data-dependent slots",
             dataDepCount - 10);
  }
  // Log per-segment summary
  if (DSP_DIAG_ENABLED(SEGMENT)) {
    for (int si = 0; si < (int)segments_.size(); si++) {
      auto& seg = segments_[si];
      int segSize = seg.endSlot - seg.startSlot + 1;
      DSP_DIAG_SEG(SEGMENT, seg.startSlot,
                   "segment[%d] slots[%d-%d] size=%d capturable=%d",
                   si, seg.startSlot, seg.endSlot, segSize, seg.isCapturable ? 1 : 0);
    }
  }

  DSP_DIAG(SEGMENT, "rebuildSegmentsForFrozenShapes: %d -> %d segments (%d/%d slots capturable, %d data-dep, matmulSeg=%d)",
           oldSegCount, (int)segments_.size(), capturableSlots, numSlots_, dataDepCount,
           static_cast<int>(matmulSegmentation));

  // Propagate slotArrayCache_, resolve backend, and detect value-dep ops for all rebuilt segments.
  for (auto& seg : segments_) {
    seg.slotArrayCache = slotArrayCache_;
    seg.selectedBackend = resolveBackendForSegment(seg.isCapturable);
    // Scan slots for value-dependent ops — even in frozen mode, segments containing
    // these ops must recompute the shape key (not use cached) because input VALUES
    // could change even when shapes are stable.
    seg.hasValueDepOps = false;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      if (slots_[s].outputShapeDependsOnInputValues) {
        seg.hasValueDepOps = true;
        break;
      }
    }
  }

  // CRITICAL FIX: When shapes are frozen, skip symbolic shape warmup.
  // Frozen shapes are constant, so symbolic shape ranges are unnecessary.
  // Use standard FNV-1a shape key which will be stable.
  if (Environment::getInstance().dspSymbolicShapes()) {
    int warmup = Environment::getInstance().dspSymbolicShapeWarmup();
    // Free old profiles before rebuild
    for (auto& seg : segments_) {
      if (seg.exec.symbolicRangeData != nullptr) {
        freeSegmentShapeProfile(static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData));
        seg.exec.symbolicRangeData = nullptr;
      }
    }
    
    // After rebuildSegmentsForFrozenShapes(), segments are merged.
    // For frozen shapes, skip symbolic shape entirely - use standard FNV-1a key.
    for (auto& seg : segments_) {
      seg.exec.symbolicShapeEnabled = false;  // Disable symbolic shapes for frozen decode
      seg.exec.symbolicWarmupRemaining = 0;
      seg.exec.symbolicRangeData = nullptr;
    }
    
    DSP_DIAG(SEGMENT, "Disabled symbolic shapes for %d frozen segments (using FNV-1a key)",
             (int)segments_.size());
  }

  // PRE-WARMUP: Disable in-place fusion for ops that consume frozen constant outputs.
  // This MUST happen before the warmup execution. In-place fusion allows downstream ops
  // to overwrite their input buffer with their output. If the input is a frozen constant
  // (e.g., reduce_mean of shape_of), the cached frozen value gets corrupted during warmup.
  // detectFrozenConstants() runs AFTER the warmup (executeCount_==1), but by then the
  // cached values are already wrong. Fix: pre-identify frozen candidates using the same
  // graph analysis (SHAPE_ONLY_OUTPUT trait + transitive consumer analysis) and disable
  // in-place fusion for their consumers before the warmup runs.

  // Propagate external dependency through the graph (same logic as detectFrozenConstants).
  std::vector<bool> dependsOnExternal(totalOutputSlots_, false);
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    // Shape-only-output ops don't propagate external dependency
    bool isShapeOnly = (sl.op && sl.op->getOpDescriptor() &&
                        sl.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT));
    if (isShapeOnly) continue; // doesn't propagate dependency

    bool anyExternal = false;
    for (int i = 0; i < sl.numInputs; i++) {
      int srcIdx = sl.inputSourceIndices[i];
      if (srcIdx < 0 || (srcIdx < totalOutputSlots_ && dependsOnExternal[srcIdx])) {
        anyExternal = true;
        break;
      }
    }
    if (anyExternal) {
      for (int o = 0; o < sl.numOutputs; o++) {
        int si = sl.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) dependsOnExternal[si] = true;
      }
    }
  }

  // Collect output slot indices of frozen candidate slots
  std::unordered_set<int> frozenOutputSlots;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    bool allConstant = true;
    for (int o = 0; o < sl.numOutputs; o++) {
      int si = sl.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && dependsOnExternal[si]) {
        allConstant = false;
        break;
      }
    }
    if (allConstant && !sl.isDataDependent) {
      for (int o = 0; o < sl.numOutputs; o++) {
        int si = sl.outputSlotIndices[o];
        if (si >= 0) frozenOutputSlots.insert(si);
      }
    }
  }

  DSP_DIAG(FUSION, "frozen constant analysis: %zu frozen output slots out of %d total",
           frozenOutputSlots.size(), totalOutputSlots_);

  // Disable in-place fusion for any op whose in-place input comes from a frozen output
  int disabledInPlace = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    if (sl.inPlaceFused && sl.inPlaceFusedInputIdx >= 0 &&
        sl.inPlaceFusedInputIdx < sl.numInputs) {
      int srcSlot = sl.inputSourceIndices[sl.inPlaceFusedInputIdx];
      if (srcSlot >= 0 && frozenOutputSlots.count(srcSlot)) {
        DSP_DIAG_SLOT(FUSION, s,
                      "disabled in-place fusion: slot %d (%s) consumes frozen slot %d",
                      s, sl.opName.c_str(), srcSlot);
        sl.inPlaceFused = false;
        sl.inPlaceFusedInputIdx = -1;
        disabledInPlace++;
      }
    }
  }
  if (disabledInPlace > 0) {
    DSP_DIAG(FUSION, "rebuildSegments: disabled %d in-place fusions that would corrupt %zu frozen outputs",
             disabledInPlace, frozenOutputSlots.size());
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

void NativeDynamicShapePlan::printCompilationAudit() const {
  if (lastCompilationAudit_.empty()) {
    DSP_DIAG(BACKEND, "no compilation audit data");
    return;
  }

  const char* backendName = cpuGraphBackend_ ? cpuGraphBackend_->name() : "unknown";

  DSP_DIAG(BACKEND, "CPU GRAPH COMPILATION AUDIT (%s backend), %zu total ops",
           backendName, lastCompilationAudit_.size());

  int skippedCount = 0;
  int compiledCount = 0;

  for (const auto& entry : lastCompilationAudit_) {
    if (entry.wasCompiled) {
      compiledCount++;
      if (entry.reason.empty()) {
        DSP_DIAG_SLOT(BACKEND, entry.slotIndex, "[slot %3d] %s COMPILED",
                      entry.slotIndex, entry.opName.c_str());
      } else {
        DSP_DIAG_SLOT(BACKEND, entry.slotIndex, "[slot %3d] %s COMPILED (%s)",
                      entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    } else {
      skippedCount++;
      DSP_DIAG_SLOT(BACKEND, entry.slotIndex, "[slot %3d] %s *** SKIPPED *** (%s)",
                     entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
    }
  }

  DSP_DIAG(COMPILE, "compilation audit: %d compiled, %d skipped of %zu total (%s backend)",
           compiledCount, skippedCount, lastCompilationAudit_.size(), backendName);
  if (skippedCount > 0) {
    DSP_DIAG(FALLBACK, "%d ops skipped by %s backend - segment will fallback to slot-by-slot",
             skippedCount, backendName);
  }
}

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
