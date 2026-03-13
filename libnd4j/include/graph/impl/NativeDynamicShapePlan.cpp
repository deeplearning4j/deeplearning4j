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

#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <sstream>
#include <graph/gpu/SymbolicShapeRanges.h>
#ifdef SD_CUDA
#include <graph/gpu/CaptureBufferRegistry.h>
#endif
#include <graph/DspDiagnostics.h>
#include <graph/FusionPass.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <graph/GraphBackend.h>
#include <array/DataBuffer.h>
#include <helpers/ConstantShapeHelper.h>
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
      isFusedChainHead(other.isFusedChainHead),
      fusedChainLength(other.fusedChainLength),
      isFusedChainTail(other.isFusedChainTail),
      targetDeviceId(other.targetDeviceId),
      legacyOpType(other.legacyOpType),
      legacyOpNum(other.legacyOpNum),
      cachedShapeKey(other.cachedShapeKey),
      cachedOutputShapes(std::move(other.cachedOutputShapes)),
      shapeCacheValid(other.shapeCacheValid),
      shapeStatic(other.shapeStatic) {
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
    shapeCacheValid = other.shapeCacheValid;
    shapeStatic = other.shapeStatic;

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
      slotViewOutputs_(nullptr),
      contextPool_(nullptr), viewProducerDetectionDone_(false), frozenConstantDetectionDone_(false),
      pendingCloseBytes_(0), gpuGraphCaptureEnabled_(false), totalGraphReplays_(0), jitMode_(JitMode::GRAPH_ONLY), graphExecutionMode_(GraphExecutionMode::GEM_AUTO),
      shapesFrozen_(false), executeCount_(0), executionTimingEnabled_(false), traceEnabled_(false),
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false),
      gpuGraphBackend_(nullptr), gpuGraphBackendChecked_(false),
      untrackedOutputCache_(nullptr), untrackedOutputCacheSize_(0),
      kvCacheRetentionEnabled_(false), kvCachePosition_(0), kvCacheMaxLen_(0),
      kvCacheNumMappings_(0), kvCacheMappings_(nullptr),
      maxKvCacheLen_(0),
      hasControlFlow_(false), loopRegions_(nullptr), numLoopRegions_(0),
      slotIsDead_(nullptr), slotIsDeadSize_(0)
      {}

NativeDynamicShapePlan::~NativeDynamicShapePlan() {
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: START plan=%p numSlots=%d totalOutputSlots=%d",
           this, numSlots_, totalOutputSlots_);

  // Finalize diagnostics report
  DspDiagnostics::getInstance().endPlanExecution();
  DspDiagnostics::getInstance().printPlanReport();
  DspDiagnostics::getInstance().flushJsonReport();

  // Free slots
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

  // Unified dedup set across slotArrayCache_, slotViewOutputs_, and pendingClose_
  // to prevent double-free. Arrays can end up in multiple collections when:
  //   - executeSingleKernel pre-allocates into both outputSlots_ and slotArrayCache_
  //   - release schedule later adds the same array to pendingClose_
  //   - flushPendingClose deletes it but slotArrayCache_ retains dangling pointer
  std::unordered_set<NDArray*> deleted;

  // Free cached output slot arrays
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing slotArrayCache_ (%d slots)", totalOutputSlots_);
  if (slotArrayCache_) {
    int cacheCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotArrayCache_[i] != nullptr && deleted.insert(slotArrayCache_[i]).second) {
        cacheCount++;
        delete slotArrayCache_[i];
      }
    }
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: deleted %d unique arrays from slotArrayCache_", cacheCount);
    delete[] slotArrayCache_;
  }

  // Free output slots array (pointers, not the arrays themselves)
  delete[] outputSlots_;

  // Free view producer flags
  delete[] slotIsViewProducer_;

  // Free zero-copy view outputs (cross-dedup with slotArrayCache_)
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing slotViewOutputs_");
  if (slotViewOutputs_) {
    int viewCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotViewOutputs_[i] != nullptr && deleted.insert(slotViewOutputs_[i]).second) {
        viewCount++;
        delete slotViewOutputs_[i];
      }
    }
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: deleted %d unique arrays from slotViewOutputs_", viewCount);
    delete[] slotViewOutputs_;
  }

  // Free context pool
  if (contextPool_) {
    for (int i = 0; i < numSlots_; i++) {
      delete contextPool_[i];
    }
    delete[] contextPool_;
  }

  // Flush any remaining pending close (cross-dedup with slotArrayCache_ and views)
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing pendingClose_ (%d items)",
           static_cast<int>(pendingClose_.size()));
  for (auto* arr : pendingClose_) {
    if (arr != nullptr && deleted.insert(arr).second) {
      delete arr;
    }
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

  // Free symbolic shape range profiles from all segments
  for (auto& seg : segments_) {
    if (seg.symbolicRangeData != nullptr) {
      freeSegmentShapeProfile(static_cast<SegmentShapeProfile*>(seg.symbolicRangeData));
      seg.symbolicRangeData = nullptr;
    }
  }

  // Free platform-specific GPU resources (capture buffers, workspace, JIT kernels,
  // math library workspace, batch-zero). No-op on CPU builds.
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing platform GPU resources");
  platformFreePlanResources();
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

  // Allocate slots
  plan->slots_ = new NativeSlot[plan->numSlots_];

  // Read per-slot data
  for (int s = 0; s < plan->numSlots_; s++) {
    NativeSlot& slot = plan->slots_[s];
    slot.opHash = reader.read<int64_t>();
    slot.opName = reader.readString();
    slot.numInputs = reader.read<int32_t>();
    slot.numOutputs = reader.read<int32_t>();

    // Input wiring
    slot.inputSourceIndices = new int[slot.numInputs];
    reader.readArray(slot.inputSourceIndices, slot.numInputs);
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
                              normalized == "expand_dims" || normalized == "squeeze");
    }
    // View-capable ops share input buffer → no zeroing needed
    if (slot.isViewCapableOp) slot.needsZeroedOutput = false;

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

  plan->slotArrayCache_ = new NDArray*[plan->totalOutputSlots_];
  std::memset(plan->slotArrayCache_, 0, sizeof(NDArray*) * plan->totalOutputSlots_);

  plan->slotIsViewProducer_ = new bool[plan->totalOutputSlots_];
  std::memset(plan->slotIsViewProducer_, 0, sizeof(bool) * plan->totalOutputSlots_);

  plan->slotViewOutputs_ = new NDArray*[plan->totalOutputSlots_];
  std::memset(plan->slotViewOutputs_, 0, sizeof(NDArray*) * plan->totalOutputSlots_);

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

  // Notify diagnostics that a plan was compiled
  DspDiagnostics::getInstance().beginPlanExecution(
      plan->numSlots_, static_cast<int>(plan->segments_.size()));
  DSP_DIAG(COMPILE, "plan compiled: %d slots, %d segments",
           plan->numSlots_, static_cast<int>(plan->segments_.size()));

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

  DSP_DIAG(EXECUTE, "step %d: frozen=%d segs=%d graphCapture=%d ext=%d",
           executeCount_, static_cast<int>(shapesFrozen_),
           static_cast<int>(segments_.size()),
           static_cast<int>(gpuGraphCaptureEnabled_), numExternalInputs);

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

  // Frozen graph fast path: if shapes are frozen and a single captured GPU graph
  // covers the entire plan, skip all per-slot/per-segment abstractions.
  // Returns OK if fast path handled execution, MAYBE to fall through.
  auto fastPathResult = platformTryFrozenFastPath(
      externalInputs, numExternalInputs, requestedOutputs, numRequestedOutputs, stream);
  if (fastPathResult != Status::MAYBE) return fastPathResult;

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
  if (!shapesFrozen_ || executeCount_ == 0) {
    for (auto& segment : segments_) {
      // Keep caches for segments that can replay (GPU: instantiated graph,
      // CPU: capturable and not failed). Platform dispatch handles the check.
      if (platformShouldKeepSegmentCache(segment)) continue;
      for (int stepIdx = segment.startSlot; stepIdx <= segment.endSlot; stepIdx++) {
        auto& slot = slots_[stepIdx];
        if (slot.shapeStatic) continue;
        slot.cachedShapeKey = 0;
        slot.cachedOutputShapes.clear();
        slot.shapeCacheValid = false;
        slot.frozenContextReady = false;
        slot.frozenConstantSlot = false;
      }
    }
  }

  if (shapesFrozen_ && executeCount_ > 0) {
    std::memcpy(outputSlots_, slotArrayCache_, sizeof(NDArray*) * totalOutputSlots_);
  } else {
    std::memset(outputSlots_, 0, sizeof(NDArray*) * totalOutputSlots_);
  }

  // Timing instrumentation
  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Step 1b: Parallel precompilation of all GPU-compilable segments.
  // On GPU: fires async compilation threads for all eligible segments.
  // On CPU: no-op.
  platformPrecompileSegments(externalInputs, numExternalInputs);

  // Step 2: Execute segments
  int segmentIdx = 0;
  long long graphReplayUs = 0, slotBySlotUs = 0;
  int graphReplaySegs = 0, slotBySlotSegs = 0, graphReplaySlots = 0, slotBySlotSlots = 0;
  for (auto& segment : segments_) {
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }

    bool useGraph = platformShouldUseGraph(segment);

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    bool segUsedGraph = false;
    int segSlots = segment.endSlot - segment.startSlot + 1;

    if (useGraph) {
      // Platform dispatch handles the full backend cascade:
      // GPU: compiler backend → JIT → graph capture/replay → slot-by-slot
      // CPU: GPU backend → CPU graph → slot-by-slot
      auto status = platformExecuteSegmentWithBackends(
          segment, externalInputs, numExternalInputs, stream, segUsedGraph);
      if (status != Status::OK) return status;
    } else {
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

    // Post-segment check: on GPU, detects sticky errors from async execution.
    // On CPU, always returns OK.
    auto postStatus = platformCheckPostSegment(segment);
    if (postStatus != Status::OK) return postStatus;

    segmentIdx++;
  }

  auto tSegsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

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

  // Step 4: Final flush
  flushPendingClose(stream);

  auto tFlushDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Track execution count for shapes-frozen optimization
  if (shapesFrozen_) executeCount_++;

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
     << ",\"compiledByBackend\":\"" << seg.compiledByBackend << "\""
     << ",\"capturable\":" << (seg.isCapturable ? "true" : "false")
     << ",\"captureFailed\":" << (seg.captureFailed ? "true" : "false")
     << ",\"executionCount\":" << seg.executionCount
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
}

// ─── Memory management ─────────────────────────────────────────────────────

void NativeDynamicShapePlan::flushPendingClose(void* stream) {
  // Delete evicted NDArrays (shape mismatch during slot cache reuse).
  // NDArray destructor handles GPU memory deallocation via DataBuffer::deleteSpecial().
  // Deduplicate: the same NDArray pointer can be pushed multiple times when
  // identity ops share pointers across slots and both slots get evicted.
  // Without dedup, the second delete corrupts glibc heap metadata → SIGABRT.
  std::unordered_set<NDArray*> seen;
  for (auto* arr : pendingClose_) {
    if (arr != nullptr) seen.insert(arr);
  }
  // Nullify slotArrayCache_ entries pointing to arrays about to be freed.
  // Without this, memcpy(outputSlots_, slotArrayCache_) at the start of the
  // next step copies freed pointers into outputSlots_, and the destructor
  // double-frees them.
  if (slotArrayCache_ && !seen.empty()) {
    for (int si = 0; si < totalOutputSlots_; si++) {
      if (slotArrayCache_[si] != nullptr && seen.count(slotArrayCache_[si])) {
        slotArrayCache_[si] = nullptr;
      }
    }
  }
  for (auto* arr : seen) {
    delete arr;
  }
  pendingClose_.clear();
  pendingCloseBytes_ = 0;
}

void NativeDynamicShapePlan::clearShapeCaches() {
  // When shapes are frozen, skip clearing entirely after first execution.
  // All cached shapes remain valid since external input shapes are constant.
  if (shapesFrozen_ && executeCount_ > 0) return;

  for (int i = 0; i < numSlots_; i++) {
    if (!slots_[i].shapeStatic) {
      slots_[i].cachedShapeKey = 0;
      slots_[i].cachedOutputShapes.clear();
      slots_[i].shapeCacheValid = false;
    }
  }
}

void NativeDynamicShapePlan::clearAllShapeCachesForce() {
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].cachedShapeKey = 0;
    slots_[i].cachedOutputShapes.clear();
    slots_[i].shapeCacheValid = false;
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
      // Java passes requested output index; convert to absolute slot index
      int reqOutputIdx = mappings[i * 3];
      int slotIdx = (reqOutputIdx >= 0 && reqOutputIdx < numRequestedOutputs_)
                    ? requestedOutputSlotIndices_[reqOutputIdx] : -1;
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

    NDArray::prepareSpecialUse({staticBuf}, {presentKv});

    sd::ops::helpers::KvScatterEntry entry;
    entry.srcPtr = presentKv->specialBuffer();
    entry.dstPtr = staticBuf->specialBuffer();
    entry.heads = presentKv->sizeAt(1);
    entry.srcSeqLen = presentKv->sizeAt(2);
    entry.dstSeqLen = staticBuf->sizeAt(2);
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
  }

  // Register special use for all pairs
  for (auto& pair : scatterPairs) {
    NDArray::registerSpecialUse({pair.second}, {pair.first});
  }

  platformEndKvScatter(savedState);

  DSP_DIAG(KV_CACHE, "KV scatter (batched): %d scattered, %d skipped, pos=%d", scattered, skipped, kvCachePosition_);
}

// ─── Graph segmentation for GPU graph capture ───────────────────────────────

void NativeDynamicShapePlan::buildSegments() {
  if (numSlots_ == 0) return;

  // Segmentation policy:
  //
  // Merge as many consecutive slots as possible into each capturable segment.
  // Each contiguous capturable run (with the same device) becomes ONE segment.
  // At runtime, if a segment's shapes are stable it gets captured once and
  // replayed every step. If a segment's shapes change repeatedly (e.g. KV-growing
  // attention concat), maybeSplitUnstableSegments() splits it at all value-dep
  // op boundaries. Stable sub-segments get captured; unstable ones become
  // permanently slot-by-slot, minimizing overhead.
  //
  // Capturability: a slot is capturable iff:
  //   1. It is NOT data-dependent (where/unique/nms produce variable-length output)
  //   2. It is NOT a value-dep-shape op (reshape/concat/gather whose output SHAPE
  //      depends on runtime VALUES). Such ops always run slot-by-slot because the
  //      segment shape key hashes input SHAPES only — it cannot detect when a
  //      value-dep op's output shape changes. Replaying a captured graph with stale
  //      output shapes produces wrong results.
  //
  // At runtime, capturable segments with stable shape keys capture once and replay.
  // Segments whose shapes change every step (e.g. KV-growing attention) will hit
  // INSTABILITY_THRESHOLD and be permanently marked slot-by-slot (captureFailed)
  // via maybeSplitUnstableSegments() → no value-dep ops found → captureFailed.

  auto isSlotCapturable = [](const NativeSlot& slot) -> bool {
    // Control flow ops are never capturable — execution path is data-dependent
    if (slot.controlFlowType != CF_NONE) return false;
    if (slot.isDataDependent) return false;
    // Value-dep-shape ops must always run slot-by-slot, regardless of input source.
    // Their output shapes depend on runtime VALUES (not just shapes), so the segment
    // shape key (which hashes input shapes) can't detect when their output shapes change.
    // A graph replay with stale output shapes produces wrong results.
    // The adaptive splitting will detect KV-growing segments (no value-dep ops,
    // shapes change every step) and permanently mark them slot-by-slot via captureFailed.
    if (slot.outputShapeDependsOnInputValues) return false;
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

  // Propagate slotArrayCache_ to all segments so GPU backends can update
  // the cache when pre-allocating output arrays (prevents memory leaks).
  for (auto& seg : segments_) {
    seg.slotArrayCache = slotArrayCache_;
  }

  // Initialize symbolic shape ranges if enabled
  if (Environment::getInstance().dspSymbolicShapes()) {
    int warmup = Environment::getInstance().dspSymbolicShapeWarmup();
    for (auto& seg : segments_) {
      seg.symbolicShapeEnabled = true;
      seg.symbolicWarmupRemaining = warmup;
      seg.symbolicRangeData = createSegmentShapeProfile(warmup);
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
  // Destroy existing cached graphs (they reference old segment boundaries)
  for (auto& seg : segments_) {
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

  // Reset frozen context for all slots since segment boundaries changed
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].frozenContextReady = false;
    slots_[i].frozenConstantSlot = false;
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
  DSP_DIAG(SEGMENT, "rebuildSegmentsForFrozenShapes: %d -> %d segments (%d/%d slots capturable, %d data-dep, matmulSeg=%d)",
           oldSegCount, (int)segments_.size(), capturableSlots, numSlots_, dataDepCount,
           static_cast<int>(matmulSegmentation));

  // Propagate slotArrayCache_ to all rebuilt segments so GPU backends can update
  // the cache when pre-allocating output arrays (prevents memory leaks).
  for (auto& seg : segments_) {
    seg.slotArrayCache = slotArrayCache_;
  }

  // PRE-WARMUP: Disable in-place fusion for ops that consume frozen constant outputs.
  // This MUST happen before the warmup execution. In-place fusion allows downstream ops
  // to overwrite their input buffer with their output. If the input is a frozen constant
  // (e.g., reduce_mean of shape_of), the cached frozen value gets corrupted during warmup.
  // detectFrozenConstants() runs AFTER the warmup (executeCount_==1), but by then the
  // cached values are already wrong. Fix: pre-identify frozen candidates using the same
  // graph analysis (VALUE_INDEPENDENT_OPS + transitive consumer analysis) and disable
  // in-place fusion for their consumers before the warmup runs.
  static const std::unordered_set<std::string> VALUE_INDEPENDENT_OPS = {
      "shape_of", "size_at", "rank",
      "zeros_like", "zeros_as", "zeroslike",
      "ones_like", "ones_as", "oneslike",
      "create",
  };

  auto normalizeOp = [](const std::string& opName) -> std::string {
    std::string n = opName;
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return n;
  };

  // Propagate external dependency through the graph (same logic as detectFrozenConstants).
  std::vector<bool> dependsOnExternal(totalOutputSlots_, false);
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    auto normalized = normalizeOp(sl.opName);
    if (VALUE_INDEPENDENT_OPS.count(normalized) > 0) continue; // doesn't propagate dependency

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

  // Disable in-place fusion for any op whose in-place input comes from a frozen output
  int disabledInPlace = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    if (sl.inPlaceFused && sl.inPlaceFusedInputIdx >= 0 &&
        sl.inPlaceFusedInputIdx < sl.numInputs) {
      int srcSlot = sl.inputSourceIndices[sl.inPlaceFusedInputIdx];
      if (srcSlot >= 0 && frozenOutputSlots.count(srcSlot)) {
        sl.inPlaceFused = false;
        sl.inPlaceFusedInputIdx = -1;
        disabledInPlace++;
      }
    }
  }
  if (disabledInPlace > 0) {
    DSP_DIAG(SEGMENT, "rebuildSegments: disabled %d in-place fusions that would corrupt %zu frozen outputs",
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
