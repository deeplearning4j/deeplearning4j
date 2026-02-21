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
#include <graph/FusionPass.h>
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
#include <ops/declarable/LegacyPairwiseTransformOp.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <system/Environment.h>
#ifdef SD_CUDA
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <ops/declarable/helpers/kv_scatter.h>
#endif

// Include CPU graph backends conditionally
#include <config.h>
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
#endif
// Include GPU graph backend (Triton) conditionally
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

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
      needsZeroedOutput(other.needsZeroedOutput),
      isDataDependent(other.isDataDependent),
      outputShapeDependsOnInputValues(other.outputShapeDependsOnInputValues),
      needsIntLongSync(other.needsIntLongSync),
      isCustomOp(other.isCustomOp),
      isIdentityOp(other.isIdentityOp),
      inPlaceFused(other.inPlaceFused),
      inPlaceFusedInputIdx(other.inPlaceFusedInputIdx),
      targetDeviceId(other.targetDeviceId),
      legacyOpType(other.legacyOpType),
      legacyOpNum(other.legacyOpNum),
      cachedShapeKey(other.cachedShapeKey),
      cachedOutputShapes(std::move(other.cachedOutputShapes)),
      shapeCacheValid(other.shapeCacheValid),
      shapeStatic(other.shapeStatic) {
  other.inputSourceIndices = nullptr;
  other.inputSourceTypes = nullptr;
  other.outputSlotIndices = nullptr;
  other.iArgs = nullptr;
  other.tArgs = nullptr;
  other.bArgs = nullptr;
  other.dArgs = nullptr;
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
    needsZeroedOutput = other.needsZeroedOutput;
    isDataDependent = other.isDataDependent;
    outputShapeDependsOnInputValues = other.outputShapeDependsOnInputValues;
    needsIntLongSync = other.needsIntLongSync;
    isCustomOp = other.isCustomOp;
    isIdentityOp = other.isIdentityOp;
    inPlaceFused = other.inPlaceFused;
    inPlaceFusedInputIdx = other.inPlaceFusedInputIdx;
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
      pendingCloseBytes_(0), cudaGraphsEnabled_(false), totalGraphReplays_(0),
      shapesFrozen_(false), executeCount_(0), executionTimingEnabled_(false), traceEnabled_(false),
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false),
      gpuGraphBackend_(nullptr), gpuGraphBackendChecked_(false),
      untrackedOutputCache_(nullptr), untrackedOutputCacheSize_(0),
      kvCacheRetentionEnabled_(false), kvCachePosition_(0), kvCacheMaxLen_(0),
      kvCacheNumMappings_(0), kvCacheMappings_(nullptr),
      maxKvCacheLen_(0)
#ifdef SD_CUDA
      , cublasWorkspaceBuffer_(nullptr), cublasWorkspaceSize_(0)
#endif
      {}

NativeDynamicShapePlan::~NativeDynamicShapePlan() {
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

  // Free cached output slot arrays
  if (slotArrayCache_) {
    for (int i = 0; i < totalOutputSlots_; i++) {
      delete slotArrayCache_[i];
    }
    delete[] slotArrayCache_;
  }

  // Free output slots array (pointers, not the arrays themselves)
  delete[] outputSlots_;

  // Free view producer flags
  delete[] slotIsViewProducer_;

  // Free zero-copy view outputs
  if (slotViewOutputs_) {
    for (int i = 0; i < totalOutputSlots_; i++) {
      delete slotViewOutputs_[i];
    }
    delete[] slotViewOutputs_;
  }

  // Free context pool
  if (contextPool_) {
    for (int i = 0; i < numSlots_; i++) {
      delete contextPool_[i];
    }
    delete[] contextPool_;
  }

  // Flush any remaining pending close (evicted NDArrays)
  for (auto* arr : pendingClose_) {
    delete arr;
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

#ifdef SD_CUDA
  // Free capture buffers and workspace from all segments
  for (auto& seg : segments_) {
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();
    if (seg.captureWorkspacePtr != nullptr) {
      cudaFree(seg.captureWorkspacePtr);
      seg.captureWorkspacePtr = nullptr;
      seg.captureWorkspaceBytes = 0;
    }
  }

  // Free pre-allocated cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }
#endif
}

// ─── Deserialization from binary plan ─────────────────────────────────────────

static const uint32_t DSP_MAGIC = 0x44535031;  // "DSP1"
static const int32_t DSP_VERSION_MAX = 2;  // Max supported version

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
    sd_printf("NativeDynamicShapePlan: invalid magic 0x%08x (expected 0x%08x)\n", magic, DSP_MAGIC);
    return nullptr;
  }

  int32_t version = reader.read<int32_t>();
  if (version != 1 && version != 2) {
    sd_printf("NativeDynamicShapePlan: unsupported version %d (expected 1 or 2)\n", version);
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
        default:
          sd_printf("NativeDynamicShapePlan: unknown legacy op type %d for '%s'\n",
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
    if (!slot.op) {
      sd_printf("NativeDynamicShapePlan: op not found for name '%s' (serialized hash: %lld, legacyType: %d, legacyNum: %d)\n",
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

  // Read requested output slot indices
  plan->requestedOutputSlotIndices_ = new int[plan->numRequestedOutputs_];
  reader.readArray(plan->requestedOutputSlotIndices_, plan->numRequestedOutputs_);

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
  // Cached here so they can be reused during CUDA graph capture (where allocs fail).
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

    sd_printf("NativeDynamicShapePlan: shape analysis: %d static, %d dynamic out of %d slots\n",
              staticCount, dynamicCount, plan->numSlots_);

    // Count identity ops for diagnostics
    int identityCount = 0;
    for (int i = 0; i < plan->numSlots_; i++) {
      if (plan->slots_[i].isIdentityOp) identityCount++;
    }
    if (identityCount > 0) {
      sd_printf("NativeDynamicShapePlan: %d identity ops (will use fast-path)\n", identityCount);
    }
  }

  // Build graph segments for CUDA Graphs
  plan->buildSegments();

  // Detect and apply fusion candidates
  if (plan->numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(plan->slots_, plan->numSlots_);
    if (!fusions.empty()) {
      sd_printf("NativeDynamicShapePlan: detected %d fusion candidates\n",
                static_cast<int>(fusions.size()));
      for (auto& f : fusions) {
        sd_printf("  fusion: slots %d-%d, type=%d, chain=%d\n",
                  f.startSlot, f.endSlot, static_cast<int>(f.type), f.chainLength);
      }

      int applied = FusionPass::applyFusions(plan->slots_, plan->numSlots_, fusions);
      sd_printf("NativeDynamicShapePlan: applied %d of %d fusion candidates (in-place execution)\n",
                applied, static_cast<int>(fusions.size()));
    }
  }

  return plan;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status NativeDynamicShapePlan::execute(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs,
    void* stream) {

  if (numExternalInputs != numExternalInputs_) {
    sd_printf("NativeDynamicShapePlan::execute: expected %d external inputs, got %d\n",
              numExternalInputs_, numExternalInputs);
    return Status::BAD_ARGUMENTS;
  }

  if (numRequestedOutputs != numRequestedOutputs_) {
    sd_printf("NativeDynamicShapePlan::execute: expected %d requested outputs, got %d\n",
              numRequestedOutputs_, numRequestedOutputs);
    return Status::BAD_ARGUMENTS;
  }

  if (traceEnabled_) {
    sd_printf("NativeDSP::execute: executeCount=%d shapesFrozen=%d numSegments=%d cudaGraphs=%d numExt=%d\n",
              executeCount_, static_cast<int>(shapesFrozen_), static_cast<int>(segments_.size()),
              static_cast<int>(cudaGraphsEnabled_), numExternalInputs);
  }

#ifdef SD_CUDA
  // ═══════════════════════════════════════════════════════════════════════════
  // FROZEN GRAPH FAST PATH: "1 slot, 1 graph"
  // When shapes are frozen, we have 1 segment with 1 captured CUDA graph,
  // and the graph has been successfully replayed at least once, skip ALL
  // per-slot and per-segment abstractions. The entire decoder becomes a
  // single atomic operation: copy inputs → launch graph → return outputs.
  // ═══════════════════════════════════════════════════════════════════════════
  if (shapesFrozen_ && executeCount_ > 1 && segments_.size() == 1 &&
      segments_[0].cachedGraph != nullptr &&
      segments_[0].cachedGraph->getState() == cuda::GraphState::INSTANTIATED) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

    // Clear stale CUDA errors
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    cudaGetLastError();

    GraphSegment& seg = segments_[0];
    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;

    // Copy external inputs into fixed-address capture buffers.
    // Skip inputs whose GPU address hasn't changed since last copy (static model weights).
    bool ok = true;
    int copiedCount = 0;
    int skippedCount = 0;
    for (auto& cb : seg.captureBuffers) {
      // Direct reference: graph captured the external buffer's address directly.
      // No copy needed — KV scatter writes to the same buffer the graph reads from.
      if (cb.directReference) {
        skippedCount++;
        continue;
      }

      NDArray* src = nullptr;
      if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExternalInputs) {
        src = externalInputs[cb.externalInputIndex];
      } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
        src = slotArrayCache_[cb.crossSegmentSlotIdx];
      }
      if (src == nullptr || cb.buffer == nullptr) { ok = false; break; }

      size_t srcBytes = src->lengthOf() * src->sizeOfT();
      if (srcBytes != cb.capturedSize) { ok = false; break; }

      if (srcBytes > 0) {
        // Skip copy if source GPU pointer hasn't changed (static weight — same buffer every step)
        const void* currentPtr = src->specialBuffer();
        if (cb.initialCopyDone && currentPtr == cb.lastSourcePtr && !cb.neverSkipCopy) {
          skippedCount++;
          continue;
        }

        auto dt = src->dataType();
        bool hostMirror = (dt == INT32 || dt == INT64 || dt == BOOL)
                          && src->lengthOf() > 0 && src->lengthOf() <= 32;
        if (hostMirror) {
          src->syncToHost();
          std::memcpy(cb.buffer->buffer(), src->buffer(), srcBytes);
          cb.buffer->tickWriteHost();
          cb.buffer->syncToDevice();
        } else {
          src->syncToDevice();
          auto copyErr = cudaMemcpyAsync(cb.buffer->specialBuffer(), src->specialBuffer(),
                                         srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
          if (copyErr != cudaSuccess) { cudaGetLastError(); ok = false; break; }
        }
        cb.lastSourcePtr = currentPtr;
        cb.initialCopyDone = true;
        copiedCount++;
      }
    }
    if (traceEnabled_ && executeCount_ <= 5) {
      sd_printf("NativeDSP::frozenFastPath: copied=%d skipped=%d total=%d\n",
                copiedCount, skippedCount, copiedCount + skippedCount);
    }

    if (ok) {
      auto tCopyDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
      // No sync needed before graph launch — capture buffer copies are on the
      // same stream (cudaStr), so they are ordered before the graph launch.
      if (seg.cachedGraph->launchAsync(cudaStr)) {
        auto tLaunchDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
        // Return outputs directly from slotArrayCache_ — no slot iteration needed
        for (int i = 0; i < numRequestedOutputs_; i++) {
          int slotIdx = requestedOutputSlotIndices_[i];
          if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
            requestedOutputs[i] = slotArrayCache_[slotIdx];
          } else {
            requestedOutputs[i] = nullptr;
          }
        }
        totalGraphReplays_++;
        seg.executionCount++;
        executeCount_++;

        // C++ KV scatter using direct kvScatter CUDA kernel (no operator() allocations)
        // KV scatter now runs on the SAME execution stream as the graph, so no
        // cross-stream sync is needed — CUDA stream ordering guarantees correctness.
        if (kvCacheRetentionEnabled_) {
          // No sync needed — scatter runs on same stream as graph (ordered)
          scatterKvEntries(externalInputs, numExternalInputs, stream);
          kvCachePosition_++;
        }
        auto tScatterDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
        // Single sync: wait for graph + scatter to complete so Java can read outputs
        if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
        auto tSyncDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

        // Periodic flush (every 10 steps) and trim
        if (executeCount_ % 10 == 0) {
          flushPendingClose(stream);
        }

        if (executionTimingEnabled_) {
          auto copyUs = std::chrono::duration_cast<std::chrono::microseconds>(tCopyDone - t0).count();
          auto launchUs = std::chrono::duration_cast<std::chrono::microseconds>(tLaunchDone - tCopyDone).count();
          auto scatterUs = std::chrono::duration_cast<std::chrono::microseconds>(tScatterDone - tLaunchDone).count();
          auto syncUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncDone - tScatterDone).count();
          auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncDone - t0).count();
          sd_printf("DSP timing: copy=%lldus launch=%lldus scatter=%lldus sync=%lldus total=%lldus "
                    "(copied=%d skipped=%d)\n",
                    copyUs, launchUs, scatterUs, syncUs, totalUs, copiedCount, skippedCount);
        }
        return Status::OK;
      }
    }
    // Fast path failed — fall through to full execution path
    if (traceEnabled_) {
      sd_printf("NativeDSP::execute: frozen fast path failed (ok=%d), falling back to full path\n",
                static_cast<int>(ok));
    }
  }
#endif

  // Step 0: Clear stale CUDA errors and error references from prior execution.
  // Async GPU errors from the previous execute() call may not surface until the
  // next CUDA API call, causing false failures. Clear them proactively.
  // Also clear the custom error reference so failed graph captures from prior
  // executions don't cause false positives at stream sync time.
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
#ifdef SD_CUDA
  cudaGetLastError();

  // Clear attention workspace when no graphs are cached yet.
  // This ensures fresh workspace allocations for new shapes (e.g., prefill → decode transition).
  // When graphs ARE cached, preserve the workspace — captured CUDA graphs recorded the
  // workspace buffer pointers (stable virtual addresses via Pool Memory Model), and
  // clearing would corrupt graph replay. The per-segment clear() calls inside
  // executeSegmentWithGraph() were removed to prevent:
  //   1. Premature freeing of async GPU buffers from other segments (causes KERNEL_FAILURE at step 1)
  //   2. Corruption of already-captured segment graphs when capturing subsequent segments
  {
    bool anyGraphCached = false;
    for (const auto& seg : segments_) {
      if (seg.cachedGraph != nullptr) { anyGraphCached = true; break; }
    }
    if (!anyGraphCached) {
      AttentionWorkspace::getInstance()->clear();
    }
  }

  // Pre-execution flush: free arrays evicted during the previous call's warmup.
  // Without this, 2-3 calls' worth of evicted arrays accumulate before the next
  // mid-execution flush (every 100 steps), causing OOM on shape buffer allocation.
  flushPendingClose(stream);

  // Clear any CUDA errors from workspace clear or flush operations.
  // NDArray destructors (from AttentionWorkspace::clear() and flushPendingClose)
  // call cudaFreeAsync which may post transient stream-ordering errors. These
  // are benign — the exec stream was synced by Java's commit() before this call,
  // so all GPU work on those buffers is complete. Clear errors here so the
  // per-segment checks only catch real errors from segment execution.
  cudaGetLastError();

  // Free captured graphs for segments whose shapes have changed (detectable from
  // external inputs only — cross-segment inputs aren't available yet since
  // outputSlots_ hasn't been populated). This reclaims graph memory before segment
  // execution begins, preventing OOM when many segments have stale cached graphs.
  // Skip pre-check when shapes are frozen — shapes can't change, so cached graphs
  // remain valid. Without this skip, the pre-check incorrectly invalidates graphs
  // because outputSlots_ at this point contains stale entries from the PREVIOUS
  // execution (some cross-segment slots were released by releaseAtStep_ processing).
  // The shape key computed here differs from the capture-time key, causing spurious
  // invalidation and re-capture every step (~5s overhead per step).
  if (!shapesFrozen_ || executeCount_ == 0) {
    for (auto& segment : segments_) {
      if (segment.cachedGraph) {
        LongType segShapeKey = computeSegmentShapeKey(segment, externalInputs, numExternalInputs);
        if (segment.cachedShapeKey != segShapeKey) {
          segment.cachedGraph.reset();
        }
      }
    }
  }
#endif

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
#ifdef SD_CUDA
      // Keep caches for segments with an instantiated graph that can replay.
      // For all others (uncaptured, uncapturable, or capture-failed), force
      // re-shape each execute to avoid stale decode-step shape reuse.
      if (segment.cachedGraph != nullptr && !segment.captureFailed) continue;
#else
      if (segment.isCapturable && !segment.captureFailed) continue;
#endif
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

  // Step 2: Execute segments
  int segmentIdx = 0;
  long long graphReplayUs = 0, slotBySlotUs = 0;
  int graphReplaySegs = 0, slotBySlotSegs = 0, graphReplaySlots = 0, slotBySlotSlots = 0;
  for (auto& segment : segments_) {
    // Set graph execution flag for ALL graph backends (CUDA Graphs, oneDNN Graph, ACL).
    // This tells DataBuffer::syncToPrimary to skip D2H transfers during graph execution,
    // preventing stream conflicts (CUDA) and unnecessary data movement (CPU graphs).
    bool useGraph = false;

#ifdef SD_CUDA
    // For CUDA builds (including ZLUDA): try graph execution if either
    // CUDA Graphs are enabled OR Triton GPU backend is available.
    // Under ZLUDA+AMD: CUDA Graphs may not work, but Triton compiles
    // natively for AMD via HIP and can still fuse ops.
    //
    // When shapes are frozen (static KV cache, fixed seq_len), ALL segments become
    // safe to capture regardless of their isCapturable flag. "Value-dependent shape
    // ops" (gather, where, tile, reshape) were marked non-capturable because their
    // output SHAPES can depend on input VALUES. With frozen shapes those values are
    // constant — output shapes never change — so these ops are safe to capture.
    // The capture-buffer mechanism correctly refreshes external input data each replay.
    // If any segment fails capture it falls back permanently to slot-by-slot.
    //
    // Only enable all-segment capture after at least one full frozen execution
    // (executeCount_ > 0). Step 1 (executeCount_==0) is the warm-up pass where
    // outputSlots_ starts zeroed; passing non-capturable segments through
    // executeSegmentWithGraph with null cross-segment inputs causes spurious CUDA errors.
    bool tryCapture = (segment.isCapturable || (shapesFrozen_ && executeCount_ > 0))
                      && !segment.captureFailed;
    if (tryCapture && (cudaGraphsEnabled_ || getGpuGraphBackend() != nullptr)) {
      useGraph = true;
    }
#else
    if (segment.isCapturable && !segment.captureFailed &&
        (getCpuGraphBackend() != nullptr || getGpuGraphBackend() != nullptr)) {
      useGraph = true;
    }
#endif

    // NOTE: tl_graphExecutionActive is NOT set here. It is set only inside
    // executeSegmentWithGraph() when actual CUDA graph capture begins (not during
    // warmup). Setting it during warmup would cause sync guards to skip essential
    // H2D transfers, leaving data un-synced for subsequent capture.

#ifdef SD_CUDA
    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    bool segUsedGraph = false;
    int segSlots = segment.endSlot - segment.startSlot + 1;

    if (traceEnabled_) {
      sd_printf("NativeDSP::execute: seg[%d-%d] useGraph=%d isCapturable=%d captureFailed=%d hasGraph=%d\n",
                segment.startSlot, segment.endSlot, static_cast<int>(useGraph),
                static_cast<int>(segment.isCapturable), static_cast<int>(segment.captureFailed),
                static_cast<int>(segment.cachedGraph != nullptr));
    }

    if (useGraph) {
      // Try Triton GPU compiler first (fused kernels, best perf).
      // Under ZLUDA+AMD this uses HIP directly, bypassing ZLUDA.
      auto* gpuBackend = getGpuGraphBackend();
      bool tritonHandled = false;
      if (gpuBackend) {
        tl_graphExecutionActive = true;
        auto status = executeSegmentWithGpuGraph(segment, externalInputs, numExternalInputs, stream);
        tl_graphExecutionActive = false;
        if (status == Status::OK) { tritonHandled = true; segUsedGraph = true; }
      }
      if (!tritonHandled) {
        if (cudaGraphsEnabled_) {
          // Fall back to CUDA Graphs (captured replay).
          // tl_graphExecutionActive is managed inside executeSegmentWithGraph()
          // — only set to true during the actual capture phase, not warmup.
          auto status = executeSegmentWithGraph(segment, externalInputs, numExternalInputs, stream);
          if (status != Status::OK) {
            // Graph path failed — degrade this segment permanently to slot-by-slot.
            // This makes graph failures non-fatal: other segments can still use graphs.
            if (traceEnabled_) {
              sd_printf("NativeDSP::execute: graph path failed for seg[%d-%d] status=%d, "
                        "falling back to slot-by-slot\n",
                        segment.startSlot, segment.endSlot, static_cast<int>(status));
            }
            segment.captureFailed = true;
            // Clear sticky CUDA errors from the failed graph attempt
            cudaGetLastError();
            // Clear outputSlots AND slotArrayCache for this segment's range.
            // View-capable ops during failed capture may have cached views whose
            // DataBuffers point into freed capture buffers.
            for (int s = segment.startSlot; s <= segment.endSlot; s++) {
              for (int o = 0; o < slots_[s].numOutputs; o++) {
                int si = slots_[s].outputSlotIndices[o];
                if (si >= 0 && si < totalOutputSlots_) {
                  outputSlots_[si] = nullptr;
                  slotArrayCache_[si] = nullptr;
                }
              }
            }
            // Invalidate shape caches for this segment so slot-by-slot recomputes them
            for (int s = segment.startSlot; s <= segment.endSlot; s++) {
              auto& slot = slots_[s];
              slot.shapeCacheValid = false;
              slot.cachedShapeKey = 0;
              slot.cachedOutputShapes.clear();
              slot.frozenContextReady = false;
              slot.frozenConstantSlot = false;
            }
            status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
            if (status != Status::OK) return status;
          } else {
            // Check if this segment actually replayed a graph (cachedGraph exists and didn't fail)
            segUsedGraph = (segment.cachedGraph != nullptr && !segment.captureFailed);
          }
        } else {
          // No graph backends handled this segment — slot-by-slot fallback
          auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
          if (status != Status::OK) return status;
        }
      }
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

    // Check for sticky CUDA errors via cudaGetLastError (non-blocking).
    // Avoid per-segment cudaStreamSynchronize — with many segments (e.g., 89),
    // 89 syncs add ~240ms of overhead. Errors are caught at execute() exit.
    {
      auto lastErr = cudaGetLastError();
      if (lastErr != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf), "CUDA error after segment [%d-%d] (execCount=%d shapesFrozen=%d): %d (%s)",
                 segment.startSlot, segment.endSlot,
                 executeCount_, static_cast<int>(shapesFrozen_),
                 static_cast<int>(lastErr), cudaGetErrorString(lastErr));
        sd_printf("NativeDynamicShapePlan: %s\n", buf);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(lastErr));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
        return Status::KERNEL_FAILURE;
      }
    }
#else
    if (useGraph) {
      // Try Triton GPU compiler first (for native HIP/Level Zero GPU builds)
      auto* gpuBackend = getGpuGraphBackend();
      bool tritonHandled = false;
      if (gpuBackend) {
        tl_graphExecutionActive = true;
        auto status = executeSegmentWithGpuGraph(segment, externalInputs, numExternalInputs, stream);
        tl_graphExecutionActive = false;
        if (status == Status::OK) tritonHandled = true;
      }
      if (!tritonHandled) {
        // Fall back to CPU graph backend (oneDNN/ACL)
        tl_graphExecutionActive = true;
        auto status = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
        tl_graphExecutionActive = false;
        if (status != Status::OK) {
          // Fall back to slot-by-slot if CPU graph execution also fails
          status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
          if (status != Status::OK) return status;
        }
      }
    } else {
      auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
    }
#endif
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

  // ── Frozen constant detection ────────────────────────────────────────────
  // After the warmup execution (executeCount_ just went from 0 to 1), identify
  // slots whose output never changes between decode steps. These slots are
  // skipped entirely during subsequent executions (including graph capture),
  // removing their kernels, memsets, and memcpys from the captured graph.
  // A slot is frozen constant if ALL its inputs come from non-external sources
  // (constants/variables or other frozen constant slots).
  // Additionally, "value-independent" ops produce the same output every step
  // regardless of input values when shapes are frozen (e.g., shape_of returns
  // the shape which is frozen, zeros_like/ones_like fill based on shape only).
  // These ops break the external dependency chain — their outputs are constant
  // even though their inputs may change.
  if (shapesFrozen_ && executeCount_ == 1 && !frozenConstantDetectionDone_) {
    frozenConstantDetectionDone_ = true;

    // Ops whose output depends ONLY on input shapes, not input values.
    // When shapes are frozen, these produce identical output every step.
    // Include all synonyms since op names come from ONNX import and may vary.
    static const std::unordered_set<std::string> VALUE_INDEPENDENT_OPS = {
        "shape_of", "size_at", "rank",
        "zeros_like", "zeros_as", "zeroslike",
        "ones_like", "ones_as", "oneslike",
        "create",
    };

    std::vector<bool> dependsOnExternal(totalOutputSlots_, false);
    std::vector<bool> isValueIndependentSlot(numSlots_, false);

    // Propagate external dependency through the graph (topological order).
    // Value-independent ops do NOT propagate dependency — their outputs
    // are constant when shapes are frozen.
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];

      // Check if this op is value-independent
      auto normalized = normalizeOpName(sl.opName);
      if (VALUE_INDEPENDENT_OPS.count(normalized) > 0) {
        isValueIndependentSlot[s] = true;
        // Do NOT propagate external dependency through this op
        continue;
      }

      bool anyInputDependsOnExternal = false;
      for (int i = 0; i < sl.numInputs; i++) {
        int srcIdx = sl.inputSourceIndices[i];
        if (srcIdx < 0) {
          // External/placeholder input — changes each step
          anyInputDependsOnExternal = true;
          break;
        }
        if (srcIdx < totalOutputSlots_ && dependsOnExternal[srcIdx]) {
          anyInputDependsOnExternal = true;
          break;
        }
      }
      if (anyInputDependsOnExternal) {
        for (int o = 0; o < sl.numOutputs; o++) {
          int si = sl.outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_) {
            dependsOnExternal[si] = true;
          }
        }
      }
    }

    int frozenConstCount = 0;
    int valueIndepCount = 0;
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      bool allOutputsConstant = true;
      for (int o = 0; o < sl.numOutputs; o++) {
        int si = sl.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && dependsOnExternal[si]) {
          allOutputsConstant = false;
          break;
        }
      }
      if (allOutputsConstant && !sl.isDataDependent) {
        sl.frozenConstantSlot = true;
        frozenConstCount++;
        if (isValueIndependentSlot[s]) valueIndepCount++;
      }
    }
    sd_printf("NativeDSP: frozen constant detection: %d/%d slots are frozen constants (%d value-independent)\n",
              frozenConstCount, numSlots_, valueIndepCount);
  }

  // Adaptive segment splitting: if a segment's shape key changes for
  // INSTABILITY_THRESHOLD consecutive executions, split it at the midpoint.
  // Stable halves capture; unstable halves split further until MIN_SPLIT_SIZE.
#ifdef SD_CUDA
  if (cudaGraphsEnabled_ && !shapesFrozen_) maybeSplitUnstableSegments();
#endif

  // Print timing breakdown
  if (executionTimingEnabled_) {
    auto segMs = std::chrono::duration_cast<std::chrono::microseconds>(tSegsDone - t0).count();
    auto outMs = std::chrono::duration_cast<std::chrono::microseconds>(tOutputsDone - tSegsDone).count();
    auto flushMs = std::chrono::duration_cast<std::chrono::microseconds>(tFlushDone - tOutputsDone).count();
    auto totalMs = std::chrono::duration_cast<std::chrono::microseconds>(tFlushDone - t0).count();
    sd_printf("DSP timing: segments=%lldus outputs=%lldus flush=%lldus total=%lldus (%d segs, %d slots) | graph=%lldus(%d segs/%d slots) sbs=%lldus(%d segs/%d slots)\n",
              segMs, outMs, flushMs, totalMs,
              static_cast<int>(segments_.size()), numSlots_,
              graphReplayUs, graphReplaySegs, graphReplaySlots,
              slotBySlotUs, slotBySlotSegs, slotBySlotSlots);
  }

  return Status::OK;
}

// ─── Segment execution: slot-by-slot ─────────────────────────────────────────

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    Status status;
    try {
      status = executeSlot(stepIdx, externalArrays, numExt, stream);
    } catch (const std::exception& e) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw exception: %s",
               stepIdx, slots_[stepIdx].opName.c_str(), e.what());
      sd_printf("NativeDynamicShapePlan: %s\n", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    } catch (...) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw unknown exception",
               stepIdx, slots_[stepIdx].opName.c_str());
      sd_printf("NativeDynamicShapePlan: %s\n", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    }
    if (status != Status::OK) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d",
               stepIdx, slots_[stepIdx].opName.c_str(), static_cast<int>(status));
      sd_printf("NativeDynamicShapePlan: %s\n", buf);

      // Emit failure context: input source slots/external indices and their current shapes.
      auto printShape = [&](const char* prefix, NDArray* arr) {
        if (arr == nullptr) {
          sd_printf("NativeDynamicShapePlan:   %s: null\n", prefix);
          return;
        }
        const LongType* si = arr->shapeInfo();
        int rank = shape::rank(si);
        sd_printf("NativeDynamicShapePlan:   %s: dtype=%d rank=%d shape=[",
                  prefix, static_cast<int>(arr->dataType()), rank);
        for (int d = 0; d < rank; d++) {
          sd_printf("%lld%s", static_cast<long long>(si[d + 1]), (d + 1 < rank ? ", " : ""));
        }
        sd_printf("]\n", "");
      };

      auto findProducerSlot = [&](int outputSlotIdx) -> int {
        for (int s = 0; s < numSlots_; s++) {
          const auto& prod = slots_[s];
          for (int o = 0; o < prod.numOutputs; o++) {
            if (prod.outputSlotIndices[o] == outputSlotIdx) return s;
          }
        }
        return -1;
      };

      auto& failedSlot = slots_[stepIdx];
      for (int i = 0; i < failedSlot.numInputs; i++) {
        int srcIdx = failedSlot.inputSourceIndices[i];
        if (srcIdx >= 0) {
          int prodSlot = findProducerSlot(srcIdx);
          const char* prodName = (prodSlot >= 0 ? slots_[prodSlot].opName.c_str() : "unknown");
          sd_printf("NativeDynamicShapePlan:   input[%d] from outputSlot[%d] producer slot %d (%s)\n",
                    i, srcIdx, prodSlot, prodName);
          printShape("input-shape", (srcIdx < totalOutputSlots_ ? outputSlots_[srcIdx] : nullptr));
        } else {
          int extIdx = -(srcIdx + 1);
          sd_printf("NativeDynamicShapePlan:   input[%d] from external[%d]\n", i, extIdx);
          printShape("input-shape", (extIdx < numExt ? externalArrays[extIdx] : nullptr));
        }
      }

      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(status));
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);

      // Clear CUDA error queue so non-sticky errors don't cascade to caller.
      // Sticky errors (e.g., error 700) persist regardless, but clearing here
      // prevents non-sticky errors from propagating as false positives.
#ifdef SD_CUDA
      cudaGetLastError();
#endif
      return status;
    }

    // Release dead slots per releaseAtStep
    int releaseCount = releaseAtStepCounts_[stepIdx];
    if (releaseCount > 0) {
      for (int r = 0; r < releaseCount; r++) {
        int slotIdx = releaseAtStep_[stepIdx][r];
        outputSlots_[slotIdx] = nullptr;
      }
    }

    // Mid-execution flush of evicted arrays
    if ((stepIdx % 100 == 99) || pendingCloseBytes_ > 256ULL * 1024 * 1024) {
      flushPendingClose(stream);
    }
  }

  // After the first complete slot-by-slot pass, view producer detection is done.
  // Subsequent executions skip the fastpath_out() comparison in executeSlot() Step 5.
  if (!viewProducerDetectionDone_) {
    viewProducerDetectionDone_ = true;
    int viewCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotIsViewProducer_[i]) viewCount++;
    }
    sd_printf("NativeDSP: view producer detection done: %d/%d output slots are view producers\n",
              viewCount, totalOutputSlots_);
  }

  seg.executionCount++;
  return Status::OK;
}

// ─── Segment execution: CUDA Graph capture/replay ────────────────────────────

#ifdef SD_CUDA

// cuBLAS workspace functions (ensureCublasWorkspace, setCublasWorkspaceForCapture,
// restoreCublasWorkspaceAfterCapture) are in NativeDynamicShapePlan_cublas.cu
// because cublas_v2.h includes cuda_fp16.h which conflicts with our float16.h
// when compiled by g++.

Status NativeDynamicShapePlan::executeSegmentWithGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Compute shape key for this segment's inputs
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  if (traceEnabled_) {
    bool hasGraph = (seg.cachedGraph != nullptr);
    bool shapeMatch = hasGraph && (seg.cachedShapeKey == segShapeKey);
    sd_printf("NativeDSP::executeSegmentWithGraph: seg[%d-%d] execCount=%d hasGraph=%d "
              "shapeMatch=%d captureFailed=%d\n",
              seg.startSlot, seg.endSlot, executeCount_,
              static_cast<int>(hasGraph), static_cast<int>(shapeMatch),
              static_cast<int>(seg.captureFailed));
  }

  auto needsHostMirror = [](NDArray* arr) -> bool {
    if (arr == nullptr) return false;
    auto dt = arr->dataType();
    return (dt == INT32 || dt == INT64 || dt == BOOL) && arr->lengthOf() > 0 && arr->lengthOf() <= 32;
  };

  auto mirrorHostAndDevice = [&](NDArray* src, NDArray* dst, size_t bytes) -> bool {
    if (src == nullptr || dst == nullptr || bytes == 0) return true;
    src->syncToHost();
    void* srcHost = src->buffer();
    void* dstHost = dst->buffer();
    if (srcHost == nullptr || dstHost == nullptr) return false;
    std::memcpy(dstHost, srcHost, bytes);
    dst->tickWriteHost();
    dst->syncToDevice();
    return true;
  };

  auto invalidateSegmentShapeState = [&](GraphSegment& segRef) {
    for (int stepIdx = segRef.startSlot; stepIdx <= segRef.endSlot; stepIdx++) {
      auto& slot = slots_[stepIdx];
      slot.shapeCacheValid = false;
      slot.cachedShapeKey = 0;
      slot.cachedOutputShapes.clear();
      slot.frozenContextReady = false;
      slot.frozenConstantSlot = false;
    }
  };

  auto clearGraphStreamError = [&](cudaStream_t cudaStrm) {
    // Clear sticky CUDA errors from failed replay/capture so post-segment
    // cudaGetLastError() checks don't surface false KERNEL_FAILUREs.
    cudaGetLastError();
    if (cudaStrm != nullptr) {
      cudaStreamSynchronize(cudaStrm);
      cudaGetLastError();
    }
  };

  // ── REPLAY: cached graph with matching shapes ──
  // CUDA graphs record exact GPU memory addresses during capture. External inputs
  // (position_ids, attention_mask) are recreated each decoder step with new addresses.
  // Instead of invalidating on address change, we use fixed-address "capture buffers":
  // copy new input data into them before each replay, keeping addresses stable.
  if (seg.cachedGraph && seg.cachedShapeKey == segShapeKey &&
      seg.cachedGraph->getState() == cuda::GraphState::INSTANTIATED) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Update capture buffers: copy current external input data into fixed-address buffers.
  // The graph references these stable addresses, so replay reads correct data.
  bool captureBuffersOk = true;
  for (auto& cb : seg.captureBuffers) {
    // Direct reference: graph uses external buffer directly — no copy needed
    if (cb.directReference) continue;

    NDArray* src = nullptr;
      if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
        src = externalArrays[cb.externalInputIndex];
      } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
        src = outputSlots_[cb.crossSegmentSlotIdx];
      }

      if (src == nullptr || cb.buffer == nullptr) {
        captureBuffersOk = false;
        break;
      }

      // Check shape compatibility — if shapes changed, need re-capture
      size_t srcBytes = src->lengthOf() * src->sizeOfT();
      if (srcBytes != cb.capturedSize) {
        captureBuffersOk = false;
        break;
      }

      // Ensure latest host-updated shape/data tensors are visible on device before
      // D2D copy (important for small INT64 shape arrays used by reshape ops).
      if (srcBytes > 0) {
        if (needsHostMirror(src)) {
          if (!mirrorHostAndDevice(src, cb.buffer, srcBytes)) {
            captureBuffersOk = false;
            break;
          }
        } else {
          src->syncToDevice();
          void* srcPtr = src->specialBuffer();
          void* dstPtr = cb.buffer->specialBuffer();
          if (srcPtr == nullptr || dstPtr == nullptr) {
            captureBuffersOk = false;
            break;
          }

          cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
          if (copyErr != cudaSuccess) {
            captureBuffersOk = false;
            sd_printf("NativeDynamicShapePlan: capture buffer replay copy failed for segment [%d-%d] "
                      "(ext=%d cross=%d): %d (%s)\n",
                      seg.startSlot, seg.endSlot,
                      cb.externalInputIndex, cb.crossSegmentSlotIdx,
                      static_cast<int>(copyErr), cudaGetErrorString(copyErr));
            // Clear sticky error so fallback path can continue cleanly.
            cudaGetLastError();
            break;
          }
        }
      }
    }

    // No sync needed before graph launch: capture buffer copies use cudaMemcpyAsync
    // on cudaStr, and cudaGraphLaunch also runs on cudaStr. Same-stream operations
    // are ordered by the CUDA runtime — the copies complete before graph launch begins.
    // (The frozen fast path at line ~719 already skips this sync successfully.)
    if (captureBuffersOk && seg.cachedGraph->launchAsync(cudaStr)) {
      // Restore outputSlots_ from slot cache — during replay, executeSlot() is not
      // called, but the graph writes to the same GPU buffers the cached arrays point to.
      for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
        NativeSlot& slot = slots_[stepIdx];
        for (int i = 0; i < slot.numOutputs; i++) {
          int slotIdx = slot.outputSlotIndices[i];
          if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
            outputSlots_[slotIdx] = slotArrayCache_[slotIdx];
          }
        }
      }
      totalGraphReplays_++;
      seg.executionCount++;
      return Status::OK;
    }

    if (!captureBuffersOk) {
      // Shape change detected — invalidate and re-capture with new shapes
      sd_printf("NativeDynamicShapePlan: capture buffer shape mismatch for segment [%d-%d], "
                "invalidating for re-capture\n", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      // Free old capture buffers
      for (auto& cb : seg.captureBuffers) {
        if (!cb.directReference) delete cb.buffer;
      }
      seg.captureBuffers.clear();
      seg.cachedGraph.reset();
      // Fall through to warmup + re-capture
    } else {
      // Launch failed — invalidate and fall through
      sd_printf("NativeDynamicShapePlan: graph replay failed for segment [%d-%d], "
                "falling back to slot-by-slot\n", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      seg.cachedGraph.reset();
      // Free capture workspace — kernel params from the old graph are invalid
      if (seg.captureWorkspacePtr != nullptr) {
        cudaFree(seg.captureWorkspacePtr);
        seg.captureWorkspacePtr = nullptr;
        seg.captureWorkspaceBytes = 0;
      }
    }
  }

  // ── WARM-UP: first execution or shape change populates slot cache ──
  // When shapes change (autoregressive seq_len grows, different batch size, etc.),
  // the slot cache has arrays from the OLD shapes. Capture requires shape inference
  // which may call syncToHost (e.g., gather reads indices). syncToHost is forbidden
  // during CUDA graph capture → error 901. So we must do a warmup pass WITHOUT capture
  // to populate the slot cache with the new shapes before attempting capture.
  bool shapeChanged = (seg.cachedShapeKey != segShapeKey);

  // Track consecutive shape changes to detect unstable segments.
  // Segments whose shapes change every step (e.g. KV-growing attention concat) will
  // never capture cleanly. After INSTABILITY_THRESHOLD consecutive shape changes,
  // mark the segment for splitting. maybeSplitUnstableSegments() will split it at
  // the midpoint after this execute() call; stable halves capture, unstable halves
  // split recursively until they reach MIN_SPLIT_SIZE (permanent slot-by-slot).
  if (seg.executionCount > 0 && shapeChanged) {
    seg.consecutiveShapeChanges++;
    if (seg.consecutiveShapeChanges >= GraphSegment::INSTABILITY_THRESHOLD) {
      int segSize = seg.endSlot - seg.startSlot + 1;
      if (segSize <= GraphSegment::MIN_SPLIT_SIZE) {
        // Too small to split further — mark permanently slot-by-slot
        seg.captureFailed = true;
      } else {
        seg.needsSplit = true;
      }
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  } else if (!shapeChanged) {
    seg.consecutiveShapeChanges = 0;
  }

  if (seg.executionCount == 0 || (shapeChanged && !seg.captureFailed)) {
    // Invalidate the old graph if shapes changed
    if (shapeChanged && seg.cachedGraph) {
      seg.cachedGraph.reset();
    }
    seg.cachedShapeKey = segShapeKey;

    // Note: AttentionWorkspace::clear() is NOT called here during warmup.
    // Clearing during warmup would free GPU memory from OTHER segments' async
    // kernels that are still running, causing KERNEL_FAILURE (status 50).
    // Instead, we clear ONCE at the beginning of execute() when no graphs are
    // cached — that path is safe because the Java side calls commit() between
    // steps, ensuring all previous GPU work is complete before the next execute().
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── CAPTURE: slot cache is warm, attempt to capture CUDA graph ──
  // Shape key changed → invalidate old graph (defensive, already handled above)
  if (seg.cachedGraph && seg.cachedShapeKey != segShapeKey) {
    seg.cachedGraph.reset();
  }

  // OOM retry cooldown: if this segment previously failed capture due to OOM,
  // wait until enough executions have passed before retrying.
  if (seg.captureOomRetries > 0 && seg.executionCount < seg.captureRetryAfterExec) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Value-dependent shape ops (reshape/slice/gather with INT/LONG params) read
  // small host-side tensors during shape inference. During CUDA graph capture,
  // host sync is intentionally suppressed to keep capture valid, so those reads
  // can observe stale host buffers from a previous execution.
  //
  // Run one slot-by-slot warmup in the CURRENT execution before capture to
  // refresh slot caches and host-side shape tensors with current inputs.
  // Capture then re-executes deterministically with the same inputs.
  bool hasValueDependentShapeOps = false;
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    if (slots_[stepIdx].outputShapeDependsOnInputValues) {
      hasValueDependentShapeOps = true;
      break;
    }
  }
  if (hasValueDependentShapeOps) {
    // Warmup executes the segment once and applies releaseAtStep_, which can
    // null cross-segment producer slots needed as inputs for the capture pass.
    // Snapshot and restore outputSlots_ so capture sees the same cross-segment
    // inputs this execution started with.
    std::vector<NDArray*> preWarmupOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);

    auto warmStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    if (warmStatus != Status::OK) {
      seg.captureFailed = true;
      return warmStatus;
    }

    std::memcpy(outputSlots_, preWarmupOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);

    // Warmup is internal to this capture attempt; don't double-count segment
    // executions (capture/replay path increments executionCount itself).
    if (seg.executionCount > 0) {
      seg.executionCount--;
    }

    // Warmup may refresh per-slot shape caches for this execution.
    // Recompute segment key from current inputs before attempting capture.
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    seg.cachedShapeKey = segShapeKey;
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Use CudaGraphScheduler for capture management
  auto& scheduler = cuda::CudaGraphScheduler::getInstance();

  int currentDevice = AffinityManager::currentDeviceId();
  if (!scheduler.deviceSupportsGraphs(currentDevice)) {
    // Device doesn't support graphs — permanent fallback
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Pool priming removed — caused 2x cudaStreamSynchronize overhead per capture
  // attempt with no measurable improvement in capture success rate.

  // ── PRE-CAPTURE MEMORY CHECK ──
  // During graph capture, cudaFreeAsync calls are recorded but NOT executed.
  // All intermediate allocations accumulate simultaneously. Estimate the total
  // capture memory from the slot cache (populated during warmup) and compare
  // to free GPU memory. If insufficient, skip capture to avoid OOM/GPU faults.
  // SKIP this check for OOM retries — we want retries to actually attempt capture
  // and rely on the exception handler for graceful fallback.
  bool isOomRetry = (seg.captureOomRetries > 0);
  if (!isOomRetry) {
    size_t estimatedCaptureBytes = 0;
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      for (int i = 0; i < slot.numOutputs; i++) {
        int slotIdx = slot.outputSlotIndices[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
          estimatedCaptureBytes += slotArrayCache_[slotIdx]->lengthOf() *
                                   slotArrayCache_[slotIdx]->sizeOfT();
        }
      }
    }

    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);

    // Use 2x safety factor (reduced from 4x). Output arrays are reused from
    // slotArrayCache_ during capture (no new allocation for those). The main
    // concern is workspace/temporary allocations on non-captured streams.
    // If the estimate is wrong, the exception handler catches it gracefully
    // and falls back to slot-by-slot with OOM retry.
    size_t requiredFree = estimatedCaptureBytes * 2;
    if (requiredFree > gpuFree) {
      sd_printf("NativeDynamicShapePlan: skipping graph capture for segment [%d-%d] (%d ops): "
                "estimated %zu MB (2x %zu MB) > free %zu MB (total %zu MB)\n",
                seg.startSlot, seg.endSlot, seg.endSlot - seg.startSlot + 1,
                requiredFree / (1024 * 1024),
                estimatedCaptureBytes / (1024 * 1024),
                gpuFree / (1024 * 1024),
                gpuTotal / (1024 * 1024));
      // Don't set captureFailed — shapes may change to smaller ones later
      // where capture would succeed
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }  // end pre-capture memory check

  auto handle = std::make_shared<cuda::CudaGraphHandle>(currentDevice);

  // Clear sticky CUDA errors and sync stream before capture
  cudaGetLastError();
  if (cudaStr != nullptr) {
    auto syncErr = cudaStreamSynchronize(cudaStr);
    if (syncErr != cudaSuccess) {
      cudaGetLastError();
      seg.captureFailed = true;
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Note: AttentionWorkspace::clear() is NOT called here before capture.
  // The workspace was already cleared at the start of execute() (when no graphs were
  // cached). The pre-warmup (if hasValueDependentShapeOps) already populated the
  // workspace with fresh buffers for the current shapes. Clearing here would:
  //   1. Corrupt workspace pointers in previously-captured segment graphs (same execute() call)
  //   2. Force unnecessary reallocation (the existing buffers are already correctly shaped)
  // The CUDA Pool Memory Model guarantees stable virtual addresses across replays,
  // so captured buffer pointers remain valid as long as the workspace is not cleared.

  // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
  // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
  // graph capture on a named stream, this cross-stream allocation breaks capture.
  // By providing an explicit workspace, cuBLAS uses our buffer instead.
  // 32MB covers workspace needs for most GEMM sizes on modern GPUs.
  static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32 MB
  ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
  setCublasWorkspaceForCapture(stream);

  // Reset MmulHelper cast cache indices so capture reuses pre-allocated HALF buffers
  // in the same order as the warmup execution (avoids capture workspace temporaries)
  MmulHelper::resetCastCacheIndices();

  // ── CAPTURE BUFFER CREATION ──
  // Allocate fixed-address GPU buffers for all external and cross-segment inputs
  // used by this segment. During capture, ops read from these stable addresses.
  // Before each replay, we copy fresh input data into these buffers via cudaMemcpyAsync.
  // This decouples graph replay from external input buffer lifetimes.

  // Free old capture buffers if any (from previous capture attempt)
  for (auto& cb : seg.captureBuffers) {
    delete cb.buffer;
  }
  seg.captureBuffers.clear();

  // Build set of output slot indices produced by this segment (intra-segment)
  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  // Track which external/cross-segment inputs we've already created buffers for
  std::unordered_map<int, int> extInputToCaptureIdx;    // extIdx -> captureBuffers index
  std::unordered_map<int, int> crossSlotToCaptureIdx;   // slotIdx -> captureBuffers index
  bool captureBufferInitFailed = false;

  for (int s = seg.startSlot; s <= seg.endSlot && !captureBufferInitFailed; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs && !captureBufferInitFailed; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External input
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr &&
            extInputToCaptureIdx.find(extIdx) == extInputToCaptureIdx.end()) {
          NDArray* src = externalArrays[extIdx];
          size_t srcBytes = src->lengthOf() * src->sizeOfT();

          // Check if this external input is a KV cache buffer with stable GPU address.
          // If so, use it directly (no capture buffer allocation, no per-step D2D copy).
          // The graph captures the KV buffer's actual GPU address, and since the buffer
          // is pre-allocated at max size, the address never changes.
          bool isKvCacheInput = false;
          if (kvCacheRetentionEnabled_) {
            for (int km = 0; km < kvCacheNumMappings_; km++) {
              if (kvCacheMappings_[km].pastInputExternalIdx == extIdx) {
                isKvCacheInput = true;
                break;
              }
            }
          }

          if (isKvCacheInput) {
            // Direct reference: graph captures the external buffer's GPU address.
            // KV scatter writes to this buffer, graph reads from it — zero D2D copy.
            src->syncToDevice();
            GraphSegment::CaptureBuffer cb;
            cb.buffer = src;  // NOT owned — don't delete
            cb.externalInputIndex = extIdx;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;
            cb.directReference = true;
            cb.initialCopyDone = true;
            cb.lastSourcePtr = src->specialBuffer();
            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.captureBuffers.size());
            seg.captureBuffers.push_back(std::move(cb));
          } else {
            // Allocate a capture buffer matching the source shape and type
            auto srcShapeVec = *src->getShapeAsVector();
            auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                       sd::LaunchContext::defaultContext());
            if (srcBytes > 0) {
              if (needsHostMirror(src)) {
                if (!mirrorHostAndDevice(src, capBuf, srcBytes)) {
                  sd_printf("NativeDynamicShapePlan: capture buffer init host mirror failed for segment [%d-%d] "
                            "(ext input %d)\n", seg.startSlot, seg.endSlot, extIdx);
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }
              } else {
                src->syncToDevice();
                void* srcPtr = src->specialBuffer();
                void* dstPtr = capBuf->specialBuffer();
                if (srcPtr == nullptr || dstPtr == nullptr) {
                  sd_printf("NativeDynamicShapePlan: capture buffer init got null ptr for segment [%d-%d] "
                            "(ext input %d)\n", seg.startSlot, seg.endSlot, extIdx);
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }

                cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                      srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
                if (copyErr != cudaSuccess) {
                  sd_printf("NativeDynamicShapePlan: capture buffer init copy failed for segment [%d-%d] "
                            "(ext input %d): %d (%s)\n",
                            seg.startSlot, seg.endSlot, extIdx,
                            static_cast<int>(copyErr), cudaGetErrorString(copyErr));
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }
              }
            }

            GraphSegment::CaptureBuffer cb;
            cb.buffer = capBuf;
            cb.externalInputIndex = extIdx;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;

            // Placeholders (attention_mask, position_ids) are modified in-place each step
            // — same GPU pointer, different data. Must always copy into capture buffer.
            auto srcType = static_cast<NativeSourceType>(slot.inputSourceTypes[i]);
            if (srcType == SOURCE_PLACEHOLDER) {
              cb.neverSkipCopy = true;
            }

            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.captureBuffers.size());
            seg.captureBuffers.push_back(std::move(cb));
          }
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        // Cross-segment input
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr &&
            crossSlotToCaptureIdx.find(srcIdx) == crossSlotToCaptureIdx.end()) {
          NDArray* src = outputSlots_[srcIdx];
          auto crossShapeVec = *src->getShapeAsVector();
          auto* capBuf = new NDArray(src->ordering(), crossShapeVec, src->dataType(),
                                     sd::LaunchContext::defaultContext());
          size_t srcBytes = src->lengthOf() * src->sizeOfT();
          if (srcBytes > 0) {
            if (needsHostMirror(src)) {
              if (!mirrorHostAndDevice(src, capBuf, srcBytes)) {
                sd_printf("NativeDynamicShapePlan: capture buffer init host mirror failed for segment [%d-%d] "
                          "(cross slot %d)\n", seg.startSlot, seg.endSlot, srcIdx);
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }
            } else {
              src->syncToDevice();
              void* srcPtr = src->specialBuffer();
              void* dstPtr = capBuf->specialBuffer();
              if (srcPtr == nullptr || dstPtr == nullptr) {
                sd_printf("NativeDynamicShapePlan: capture buffer init got null ptr for segment [%d-%d] "
                          "(cross slot %d)\n", seg.startSlot, seg.endSlot, srcIdx);
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }

              cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                    srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
              if (copyErr != cudaSuccess) {
                sd_printf("NativeDynamicShapePlan: capture buffer init copy failed for segment [%d-%d] "
                          "(cross slot %d): %d (%s)\n",
                          seg.startSlot, seg.endSlot, srcIdx,
                          static_cast<int>(copyErr), cudaGetErrorString(copyErr));
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }
            }
          }

          GraphSegment::CaptureBuffer cb;
          cb.buffer = capBuf;
          cb.externalInputIndex = -1;
          cb.crossSegmentSlotIdx = srcIdx;
          cb.capturedSize = srcBytes;

          crossSlotToCaptureIdx[srcIdx] = static_cast<int>(seg.captureBuffers.size());
          seg.captureBuffers.push_back(std::move(cb));
        }
      }
    }
  }

  if (captureBufferInitFailed) {
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();
    // Reset stream/error state so slot-by-slot fallback can run cleanly.
    cudaGetLastError();
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Wire external/cross-segment inputs to capture buffers for the capture phase.
  // Save original pointers so we can restore them after capture.
  std::vector<std::pair<int, NDArray*>> savedExternalInputs;  // (extIdx, originalPtr)
  std::vector<std::pair<int, NDArray*>> savedOutputSlots;     // (slotIdx, originalPtr)
  // Snapshot outputSlots_ BEFORE swapping in capture buffers.
  // This is used on capture failure fallback to recover the original cross-segment
  // inputs plus any non-swapped slots that may be nulled by releaseAtStep_.
  std::vector<NDArray*> preCapOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);
  // Save pendingClose_ size so we can discard entries added during capture on failure.
  // View-capable ops during capture push step-0 arrays into pendingClose_, but those
  // arrays are still referenced by preCapOutputSlots — deleting them causes dangling ptrs.
  size_t pendingClosePreCapSize = pendingClose_.size();

  // Disable frozen fast path during capture. Capture replaces external inputs with
  // capture buffers and upstream slots produce different output arrays (new views, etc.).
  // The frozen context has stale input/output pointers from the prior non-capture execution.
  // Using the full (non-frozen) path during capture is a one-time cost — all context
  // pointers are properly reconfigured with capture-time arrays.
  // Save and restore frozenContextReady after capture so replay uses frozen fast path.
  std::vector<bool> savedFrozenContextReady(seg.endSlot - seg.startSlot + 1);
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    savedFrozenContextReady[s - seg.startSlot] = slots_[s].frozenContextReady;
    slots_[s].frozenContextReady = false;
  }

  for (auto& [extIdx, cbIdx] : extInputToCaptureIdx) {
    savedExternalInputs.push_back({extIdx, externalArrays[extIdx]});
    externalArrays[extIdx] = seg.captureBuffers[cbIdx].buffer;
  }
  for (auto& [slotIdx, cbIdx] : crossSlotToCaptureIdx) {
    savedOutputSlots.push_back({slotIdx, outputSlots_[slotIdx]});
    outputSlots_[slotIdx] = seg.captureBuffers[cbIdx].buffer;
  }

  // Sync the copy operations before starting capture
  if (cudaStr != nullptr) {
    cudaStreamSynchronize(cudaStr);
  }

  // RELAXED mode allows CUDA operations on non-captured streams (e.g., memory allocations
  // on stream 0 for broadcast temporaries). Our tl_graphExecutionActive guards prevent
  // capture-breaking sync operations (cudaStreamSynchronize, cudaMemcpy) on the captured stream.
  //
  // Set tl_graphExecutionActive BEFORE beginCapture so all sync guards are active
  // during the entire capture phase. Reset on any exit path (success, failure, exception).
  const cudaStream_t prevCaptureStream = tl_graphCaptureStream;
  cudaStream_t resolvedCaptureStream = cudaStr;
  if (resolvedCaptureStream == nullptr) {
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      resolvedCaptureStream = *defaultStreamPtr;
    }
  }
  tl_graphCaptureStream = resolvedCaptureStream;

  // Allocate capture workspace BEFORE beginCapture (cudaMalloc must be outside capture).
  // This eliminates cudaMallocAsync/cudaFreeAsync graph nodes from PointersManager temporaries.
  // With 3781 slots and ~1939 temporary allocations, we need enough workspace to hold
  // all temporaries simultaneously. Default 512MB; configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB.
  static size_t CAPTURE_WORKSPACE_SIZE = []() -> size_t {
    const char* envVal = std::getenv("ND4J_DSP_CAPTURE_WORKSPACE_MB");
    size_t mb = 512;  // default
    if (envVal != nullptr) {
      int parsed = std::atoi(envVal);
      if (parsed > 0 && parsed <= 4096) {
        mb = static_cast<size_t>(parsed);
      }
    }
    return mb * 1024ULL * 1024ULL;
  }();
  sd_printf("NativeDSP: capture workspace check for segment [%d-%d]: ptr=%p bytes=%zu\n",
            seg.startSlot, seg.endSlot, seg.captureWorkspacePtr, seg.captureWorkspaceBytes);
  if (seg.captureWorkspacePtr == nullptr) {
    cudaError_t wsErr = cudaMalloc(&seg.captureWorkspacePtr, CAPTURE_WORKSPACE_SIZE);
    if (wsErr == cudaSuccess) {
      seg.captureWorkspaceBytes = CAPTURE_WORKSPACE_SIZE;
      sd_printf("NativeDSP: allocated %zuMB capture workspace for segment [%d-%d]\n",
                CAPTURE_WORKSPACE_SIZE / (1024*1024), seg.startSlot, seg.endSlot);
    } else {
      cudaGetLastError();
      seg.captureWorkspacePtr = nullptr;
      seg.captureWorkspaceBytes = 0;
      sd_printf("NativeDSP: WARNING - capture workspace allocation failed (%s), "
                "graph will contain cudaMallocAsync nodes\n", cudaGetErrorString(wsErr));
    }
  }
  // Set thread-local workspace for CudaMemoryPool to use during capture
  tl_captureWorkspace = seg.captureWorkspacePtr;
  tl_captureWorkspaceSize = seg.captureWorkspaceBytes;
  tl_captureWorkspaceOffset = 0;
  sd_printf("NativeDSP: tl_captureWorkspace=%p size=%zu for capture\n",
            tl_captureWorkspace, tl_captureWorkspaceSize);

  tl_graphExecutionActive = true;
  tl_capturedHostPtrs.clear();  // Reset pinned host ptr accumulator for this capture
  tl_captureReplicateCache.clear();  // Reset H2D content dedup cache

  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    sd_printf("NativeDynamicShapePlan: graph capture begin failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    tl_graphExecutionActive = false;
    tl_graphCaptureStream = prevCaptureStream;
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    restoreCublasWorkspaceAfterCapture(stream);
    clearGraphStreamError(cudaStr);
    seg.captureFailed = true;
    // Restore original pointers before fallback execution.
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    // Capture may have partially mutated slot shape caches before beginCapture failed.
    invalidateSegmentShapeState(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    // Restore frozen context state so slot-by-slot fallback uses frozen fast path
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
    }
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  bool captureOk = true;
  bool captureOomFailure = false;  // Track if failure was OOM (retryable)
  int lastCaptureSlot = seg.startSlot;  // Track for exception diagnostics
  lastCaptureAudit_.clear();

  try {
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
      lastCaptureSlot = stepIdx;
      // Check capture status before executing slot
      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          sd_printf("NativeDynamicShapePlan: CAPTURE BROKEN before slot %d (%s): "
                    "capErr=%d capStatus=%d\n",
                    stepIdx, slots_[stepIdx].opName.c_str(),
                    static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          break;
        }
      }

      // Track node count before this op for capture audit
      size_t nodesBefore = handle->getNumNodesDuringCapture(cudaStr);

      auto status = executeSlot(stepIdx, externalArrays, numExt, stream);
      if (status != Status::OK) {
        sd_printf("NativeDynamicShapePlan: op execution during capture failed at slot %d\n", stepIdx);
        captureOk = false;
        captureOomFailure = true;  // Op failure during capture is usually OOM
        break;
      }

      // Check capture status AFTER executing slot to detect which op broke it
      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          sd_printf("NativeDynamicShapePlan: CAPTURE INVALIDATED by slot %d (%s)! "
                    "capErr=%d capStatus=%d\n",
                    stepIdx, slots_[stepIdx].opName.c_str(),
                    static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          // Capture invalidation is permanent (op does host sync or cross-stream work)
          break;
        }
      }

      // Track node count after this op for capture audit
      size_t nodesAfter = handle->getNumNodesDuringCapture(cudaStr);
      {
        cuda::CaptureAuditEntry entry;
        entry.slotIndex = stepIdx;
        entry.opName = slots_[stepIdx].opName;
        entry.nodesBefore = nodesBefore;
        entry.nodesAfter = nodesAfter;
        entry.nodesContributed = (nodesAfter > nodesBefore) ? (nodesAfter - nodesBefore) : 0;
        lastCaptureAudit_.push_back(std::move(entry));
      }

      // Release dead slots during capture
      int releaseCount = releaseAtStepCounts_[stepIdx];
      if (releaseCount > 0) {
        for (int r = 0; r < releaseCount; r++) {
          int slotIdx = releaseAtStep_[stepIdx][r];
          outputSlots_[slotIdx] = nullptr;
        }
      }
    }
  } catch (const std::exception& e) {
    sd_printf("NativeDynamicShapePlan: exception during graph capture at slot %d (%s): %s\n",
              lastCaptureSlot, slots_[lastCaptureSlot].opName.c_str(), e.what());
    captureOk = false;
    // Detect OOM exceptions: "[DEVICE] allocation failed" from DataBuffer.cu
    std::string msg(e.what());
    if (msg.find("allocation failed") != std::string::npos) {
      captureOomFailure = true;
    }
  } catch (...) {
    sd_printf("NativeDynamicShapePlan: unknown exception during graph capture\n", "");
    captureOk = false;
    captureOomFailure = true;  // Unknown exceptions treated as OOM (retryable)
  }

  // Capture phase complete — reset the flag before any exit path
  size_t captureWorkspaceUsed = tl_captureWorkspaceOffset;  // Save before clearing
  tl_graphExecutionActive = false;
  tl_graphCaptureStream = prevCaptureStream;
  tl_captureWorkspace = nullptr;
  tl_captureWorkspaceSize = 0;
  tl_captureWorkspaceOffset = 0;
  restoreCublasWorkspaceAfterCapture(stream);

  // Restore original external/cross-segment input pointers.
  // During capture, we pointed these at capture buffers. Now restore them
  // so that slot-by-slot fallback (if capture failed) uses the real inputs.
  for (auto& [extIdx, origPtr] : savedExternalInputs) {
    externalArrays[extIdx] = origPtr;
  }
  for (auto& [slotIdx, origPtr] : savedOutputSlots) {
    outputSlots_[slotIdx] = origPtr;
  }

  if (!captureOk) {
    // Abort capture — stream is in an inconsistent state
    // endCapture will clean up by calling cudaStreamEndCapture
    handle->endCapture(cudaStr);

    // Free pinned host buffers accumulated during failed capture
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Clear any sticky CUDA errors left from the failed capture
    cudaGetLastError();

    // Synchronize the capture stream to ensure it returns to a clean state
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();  // Clear any sync error too
    }

    if (captureOomFailure && seg.captureOomRetries < GraphSegment::MAX_OOM_RETRIES) {
      // OOM failure — schedule retry instead of permanent disable.
      // Memory pressure may decrease as other segments get captured and replayed
      // (graph replay is more memory-efficient than slot-by-slot).
      seg.captureOomRetries++;
      seg.captureRetryAfterExec = seg.executionCount + GraphSegment::RETRY_INTERVAL;
      sd_printf("NativeDynamicShapePlan: graph capture OOM for segment [%d-%d], "
                "retry %d/%d scheduled after exec %d\n",
                seg.startSlot, seg.endSlot,
                seg.captureOomRetries, GraphSegment::MAX_OOM_RETRIES,
                seg.captureRetryAfterExec);
    } else {
      seg.captureFailed = true;
      sd_printf("NativeDynamicShapePlan: graph capture permanently failed for segment [%d-%d] "
                "(oom=%s, retries=%d)\n",
                seg.startSlot, seg.endSlot,
                captureOomFailure ? "true" : "false",
                seg.captureOomRetries);
    }

    // Discard pendingClose_ entries added during capture. These reference step-0
    // arrays that preCapOutputSlots still points to — deleting them would create
    // dangling pointers after outputSlots_ restore. Delete capture-phase arrays
    // that are NOT in preCapOutputSlots (they were allocated during capture).
    {
      // Delete capture-phase NDArrays that don't appear in preCapOutputSlots
      std::unordered_set<NDArray*> preCapSet(preCapOutputSlots.begin(), preCapOutputSlots.end());
      for (size_t pi = pendingClosePreCapSize; pi < pendingClose_.size(); pi++) {
        if (preCapSet.find(pendingClose_[pi]) == preCapSet.end()) {
          delete pendingClose_[pi];  // Capture-only array, safe to delete
        }
      }
      pendingClose_.resize(pendingClosePreCapSize);
    }

    // Clear slotArrayCache_ for this segment BEFORE freeing capture buffers.
    // During capture, view-capable ops (reshape/expand_dims/squeeze) create views
    // sharing DataBuffers from capture buffer inputs. Freeing capture buffers
    // invalidates those views. Clear the cache so slot-by-slot fallback
    // allocates fresh arrays instead of reusing dangling views.
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          slotArrayCache_[si] = nullptr;
        }
      }
    }

    // Free capture buffers on failure — they won't be needed
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();

    // Clear error reference so the capture failure doesn't propagate to the caller
    // when the slot-by-slot fallback succeeds.
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    // Restore frozen context state so slot-by-slot fallback uses frozen fast path
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
    }
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Helper lambda to clean up capture buffers on failure.
  // Must also clear slotArrayCache_ and rollback pendingClose_ because view-capable
  // ops during capture created views sharing DataBuffers with capture buffer inputs.
  auto cleanupCaptureBuffersOnFailure = [&seg, &preCapOutputSlots, &pendingClosePreCapSize, &savedFrozenContextReady, this]() {
    // Rollback pendingClose_ — same logic as main failure path
    std::unordered_set<NDArray*> preCapSet(preCapOutputSlots.begin(), preCapOutputSlots.end());
    for (size_t pi = pendingClosePreCapSize; pi < pendingClose_.size(); pi++) {
      if (preCapSet.find(pendingClose_[pi]) == preCapSet.end()) {
        delete pendingClose_[pi];
      }
    }
    pendingClose_.resize(pendingClosePreCapSize);
    // Clear slot array cache
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          slotArrayCache_[si] = nullptr;
        }
      }
    }
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();
    // Restore frozen context state so slot-by-slot fallback uses frozen fast path
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
    }
  };

  // End capture and instantiate
  if (!handle->endCapture(cudaStr)) {
    cudaGetLastError();
    sd_printf("NativeDynamicShapePlan: graph capture end failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    // Free pinned host buffers from failed capture
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    // Clear sticky CUDA errors and reset stream
    cudaGetLastError();
    cudaStreamSynchronize(cudaStr);
    cudaGetLastError();
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  if (!handle->instantiate()) {
    cudaGetLastError();
    sd_printf("NativeDynamicShapePlan: graph instantiate failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    // Free pinned host buffers from failed capture
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    clearGraphStreamError(cudaStr);
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Clear any sticky error that might have been left by updateStatistics or instantiate
  cudaGetLastError();

  // Log graph statistics after capture + instantiation
  {
    auto stats = handle->getStatistics();
    sd_printf("NativeDynamicShapePlan: graph captured for segment [%d-%d]: "
              "%zu nodes, %zu edges, %d kernels, %d memcpys, %d memsets, "
              "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty\n",
              seg.startSlot, seg.endSlot,
              handle->getNumNodes(), handle->getNumEdges(),
              stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
              stats.numMemAllocs, stats.numMemFrees,
              stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
    if (stats.numMemAllocs != stats.numMemFrees) {
      sd_printf("  WARNING: Unbalanced memory nodes: %d allocs vs %d frees. "
                "This WILL cause graph launch failure.\n",
                stats.numMemAllocs, stats.numMemFrees);
    }
  }

  // Launch the captured graph (actual execution)
  if (!handle->launchAsync(cudaStr)) {
    cudaGetLastError();
    sd_printf("NativeDynamicShapePlan: graph launch failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    // Free pinned host buffers from failed capture
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    clearGraphStreamError(cudaStr);
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Transfer pinned host buffers to the graph handle for lifetime management.
  // These persist for graph replay (H2D memcpy nodes reference them).
  for (auto* ptr : tl_capturedHostPtrs) {
    handle->addCapturedHostPtr(ptr);
  }
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();

  // Cache the graph for future replays
  seg.cachedGraph = handle;
  seg.cachedShapeKey = segShapeKey;
  seg.executionCount++;
  totalGraphReplays_++;

  // Populate outputSlots_ from slotArrayCache_ so downstream segments can read
  // this segment's outputs. The warmup+capture path fills slotArrayCache_ but
  // restores outputSlots_ to its pre-warmup state; without this step, any
  // subsequent slot-by-slot segment that reads a cross-segment output from this
  // captured segment will see a null pointer.  (Mirrors the replay path at the
  // top of executeSegmentWithCudaGraph.)
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    for (int i = 0; i < slot.numOutputs; i++) {
      int slotIdx = slot.outputSlotIndices[i];
      if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
        outputSlots_[slotIdx] = slotArrayCache_[slotIdx];
      }
    }
  }

  // Reset OOM retry state on success (capture may have succeeded on a retry)
  if (seg.captureOomRetries > 0) {
    sd_printf("NativeDynamicShapePlan: graph capture SUCCEEDED on OOM retry %d for segment [%d-%d]\n",
              seg.captureOomRetries, seg.startSlot, seg.endSlot);
    seg.captureOomRetries = 0;
    seg.captureRetryAfterExec = 0;
  }

  // Restore frozen context state after successful capture.
  // The frozen fast path was disabled during capture (stale pointers) but the captured
  // graph now has correct addresses. Subsequent replay executions use the frozen path.
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
  }

  if (executionTimingEnabled_) {
    auto stats = handle->getStatistics();
    double wsUtilPct = seg.captureWorkspaceBytes > 0
        ? (100.0 * captureWorkspaceUsed / seg.captureWorkspaceBytes) : 0.0;
    sd_printf("NativeDynamicShapePlan: captured CUDA graph for segment [%d-%d] "
              "(%zu nodes, %zu edges) [%d kern, %d memcpy, %d memset, %d alloc, %d free] "
              "workspace=%zuKB/%zuKB (%.1f%%)\n",
              seg.startSlot, seg.endSlot,
              handle->getNumNodes(), handle->getNumEdges(),
              stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
              stats.numMemAllocs, stats.numMemFrees,
              seg.captureWorkspacePtr ? (captureWorkspaceUsed / 1024) : 0,
              seg.captureWorkspaceBytes / 1024, wsUtilPct);

    // Print top-10 ops by node count for optimization targeting
    if (!lastCaptureAudit_.empty()) {
      printCaptureAudit();
    }
  }

  return Status::OK;
}

#endif  // SD_CUDA

Status NativeDynamicShapePlan::executeSlot(
    int stepIdx, NDArray** externalArrays, int numExt, void* stream) {
  NativeSlot& slot = slots_[stepIdx];

  // ── Fast path: identity ops ──────────────────────────────────────────────
  // Identity ops just pass input through. Skip shape inference, allocation,
  // op execution — just wire the output slot to point at the input.
  if (slot.isIdentityOp && slot.numInputs == 1 && slot.numOutputs >= 1) {
    int srcIdx = slot.inputSourceIndices[0];
    NDArray* input = nullptr;
    if (srcIdx >= 0) {
      input = outputSlots_[srcIdx];
    } else {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) input = externalArrays[extIdx];
    }
    if (input != nullptr) {
      for (int i = 0; i < slot.numOutputs; i++) {
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          outputSlots_[si] = input;
        }
      }
      return Status::OK;
    }
    // Fall through to normal path if input is null
  }

  // ── Frozen constant optimization ──────────────────────────────────────────
  // Some ops produce identical output every step with frozen shapes.
  // Skip both nullify and execution — output retains its value from warmup.
  // During graph CAPTURE, this means the op's kernel and memset are NOT recorded,
  // reducing graph node count. During REPLAY, the output buffer is untouched
  // by the graph (stable address, no graph node writes to it).
  // This check is BEFORE frozenContextReady so it fires during capture too
  // (capture disables frozenContextReady).
  if (slot.frozenConstantSlot) {
    return Status::OK;
  }

  // ── Fast path: frozen context ────────────────────────────────────────────
  // When shapes are frozen and this slot's context was already configured on a
  // prior execution, skip input gathering, shape inference, output allocation,
  // and context setup. Just execute the op and handle view producers.
  // outputSlots_ was pre-populated from slotArrayCache_ in execute(), so
  // downstream ops already have the right array pointers.
  if (slot.frozenContextReady) {

    // ── View-capable fast path (reshape/expand_dims/squeeze) ────────────
    // These ops are no-ops when output shares input 0's DataBuffer (set in Step 3).
    // Skip execution entirely — the output already points to input's data.
    // This eliminates copy kernels during CUDA graph capture/replay.
    if (slot.isViewCapableOp && slot.numInputs >= 1 && slot.numOutputs >= 1) {
      int si = slot.outputSlotIndices[0];
      if (si >= 0 && si < totalOutputSlots_) {
        // Get current input 0 (may have been updated by upstream slot this step)
        int srcIdx = slot.inputSourceIndices[0];
        NDArray* input0 = nullptr;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          input0 = outputSlots_[srcIdx];
        } else if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt) input0 = externalArrays[extIdx];
        }

        if (input0 != nullptr && input0->dataBuffer() != nullptr &&
            input0->ews() == 1 && input0->ordering() == 'c') {
          NDArray* currentOut = outputSlots_[si];
          if (currentOut != nullptr && currentOut->dataBuffer() == input0->dataBuffer()) {
            // Buffer matches — output already views input's data. Skip execution.
            return Status::OK;
          }
          // Buffer mismatch (external input changed) — recreate view
          if (slot.shapeCacheValid && !slot.cachedOutputShapes.empty()) {
            const LongType* outShapeInfo = slot.cachedOutputShapes[0];
            LongType outLen = shape::length(outShapeInfo);
            LongType inLen = input0->lengthOf();
            if (outLen > 0 && outLen <= inLen) {
              NDArray* newView = new NDArray(input0->dataBuffer(),
                                             const_cast<LongType*>(outShapeInfo));
              outputSlots_[si] = newView;
              slotIsViewProducer_[si] = true;
              auto& ctx2 = *contextPool_[stepIdx];
              ctx2.setOutputArray(0, newView);
              ctx2.setInputArray(0, input0);
              // Clean up old view
              NDArray* old = slotArrayCache_[si];
              if (old != nullptr && old != newView) {
                pendingClose_.push_back(old);
              }
              slotArrayCache_[si] = newView;
              return Status::OK;
            }
          }
        }
      }
      // Fall through to normal frozen execution if view not possible
    }

    auto& ctx = *contextPool_[stepIdx];

    // Refresh inputs that change each decode step:
    //   1. External (placeholder) inputs: always recreated each step (position_ids, etc.)
    //   2. Inputs from view-producer slots: their views point into the external inputs,
    //      so the view NDArray changes each step as the input is recreated.
    //
    // Execution is in ascending slot order, so view-producer slot X always runs before
    // slot Y that reads X's output. By the time Y runs, outputSlots_[si_X] holds the
    // fresh view created by X's fast-path execution this step.
    //
    // NOTE: During CUDA graph capture, frozenContextReady is temporarily disabled
    // (set to false before capture begins), so this code path is NOT reached during
    // capture. All slots go through the full (non-frozen) path during capture, which
    // properly configures context with capture-time arrays.
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External/placeholder input — always refresh
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
          ctx.setInputArray(i, externalArrays[extIdx]);
        }
      } else if (srcIdx < totalOutputSlots_ && slotIsViewProducer_[srcIdx]) {
        // Input from view-producer slot — refresh from outputSlots_ (already updated
        // this step by the view producer's own fast-path execution above us)
        if (outputSlots_[srcIdx] != nullptr) {
          ctx.setInputArray(i, outputSlots_[srcIdx]);
        }
      }
    }

    // Nullify output arrays before re-execution to prevent stale data accumulation.
    // EXCEPTION 1: Skip in-place fused ops — their output IS the input (same NDArray).
    // EXCEPTION 2: Skip view producer outputs — they're views into input data.
    if (!slot.inPlaceFused) {
      auto& ctxOuts = ctx.fastpath_out();
      for (int i = 0; i < static_cast<int>(ctxOuts.size()); i++) {
        if (ctxOuts[i] == nullptr) continue;
        int si = (i < slot.numOutputs) ? slot.outputSlotIndices[i] : -1;
        if (si >= 0 && si < totalOutputSlots_ && slotIsViewProducer_[si]) continue;
        ctxOuts[i]->nullify();
      }
    }

    // Execute the op with pre-configured context
    auto status = slot.op->execute(&ctx);

    // Update outputSlots_ and mark device-current.
    // This handles view producers (op returns a different array than pre-allocated),
    // in-place fused ops (output = input, not in slotArrayCache_), and normal ops.
    auto& ctxOuts = ctx.fastpath_out();
    for (int i = 0; i < slot.numOutputs && i < static_cast<int>(ctxOuts.size()); i++) {
      if (ctxOuts[i] != nullptr) {
        ctxOuts[i]->tickWriteDevice();
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          outputSlots_[si] = ctxOuts[i];
        }
      }
    }

    return status;
  }

  // ── Step 1: Gather inputs ────────────────────────────────────────────────
  // Use thread-local vector to avoid 4441 heap allocations per execute() call
  static thread_local std::vector<NDArray*> inputs;
  inputs.resize(slot.numInputs);
  for (int i = 0; i < slot.numInputs; i++) {
    int srcIdx = slot.inputSourceIndices[i];
    if (srcIdx >= 0) {
      // From a prior slot output
      inputs[i] = outputSlots_[srcIdx];
    } else {
      // From external inputs
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) {
        inputs[i] = externalArrays[extIdx];
      } else {
        inputs[i] = nullptr;
      }
    }

    if (inputs[i] == nullptr) {
      sd_printf("NativeDynamicShapePlan::executeSlot: null input %d for slot %d (%s)\n",
                i, stepIdx, slot.opName.c_str());
      return Status::BAD_INPUT;
    }
  }

  // ── Step 2: Shape inference ──────────────────────────────────────────────
  // When shapes are frozen and this slot's cache is populated from a prior execution,
  // skip shape key computation entirely. This saves ~5-10μs per slot × 4441 slots = ~22-44ms.
  LongType shapeKey = 0;
  bool cacheHit;
  if (shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid) {
    cacheHit = true;  // Trust cached shapes — frozen mode guarantees no shape changes
  } else {
    shapeKey = computeShapeKey(slot, inputs.data(), slot.numInputs);
    cacheHit = slot.shapeCacheValid && (slot.cachedShapeKey == shapeKey);
  }

  std::vector<const LongType*> outputShapes;
  if (cacheHit) {
    outputShapes = slot.cachedOutputShapes;
  } else {
    // Run shape inference
    auto& ctx = *contextPool_[stepIdx];

    // Set inputs on context for shape inference
    for (int i = 0; i < slot.numInputs; i++) {
      ctx.setInputArray(i, inputs[i]);
    }

    // Set arguments
    if (slot.numIArgs > 0) ctx.setIArguments(slot.iArgs, slot.numIArgs);
    if (slot.numTArgs > 0) ctx.setTArguments(slot.tArgs, slot.numTArgs);
    if (slot.numBArgs > 0) ctx.setBArguments(slot.bArgs, slot.numBArgs);
    if (slot.numDArgs > 0) ctx.setDArguments(slot.dArgs, slot.numDArgs);

    // Build input ShapeList for calculateOutputShape
    ShapeList inputShapes;
    for (int i = 0; i < slot.numInputs; i++) {
      if (inputs[i] != nullptr) {
        inputShapes.push_back(inputs[i]->shapeInfo());
      }
    }

    ShapeList* shapeList = nullptr;
    try {
      shapeList = slot.op->calculateOutputShape(&inputShapes, ctx);
    } catch (const std::exception& e) {
      sd_printf("NativeDynamicShapePlan: shape inference EXCEPTION at slot %d (%s): %s\n",
                stepIdx, slot.opName.c_str(), e.what());
      return Status::KERNEL_FAILURE;
    }
    if (shapeList == nullptr || shapeList->size() == 0) {
      sd_printf("NativeDynamicShapePlan: shape inference returned null for slot %d (%s)\n",
                stepIdx, slot.opName.c_str());
      return Status::KERNEL_FAILURE;
    }

    outputShapes.resize(shapeList->size());
    for (int i = 0; i < static_cast<int>(shapeList->size()); i++) {
      // Cache via ConstantShapeHelper for persistent shape pointers.
      // createFromExisting internally calls replicatePointer which can throw
      // if both pool allocation and pinned host fallback fail (e.g., sticky CUDA error).
      try {
        auto cached = ConstantShapeHelper::getInstance().createFromExisting(
            const_cast<LongType*>(shapeList->at(i)));
        outputShapes[i] = cached;
      } catch (const std::exception& e) {
        sd_printf("NativeDynamicShapePlan: createFromExisting EXCEPTION at slot %d (%s) output[%d]: %s\n",
                  stepIdx, slot.opName.c_str(), i, e.what());
        delete shapeList;
        return Status::KERNEL_FAILURE;
      }
    }

    // Update cache
    slot.cachedShapeKey = shapeKey;
    slot.cachedOutputShapes = outputShapes;
    slot.shapeCacheValid = true;

    delete shapeList;
  }

  // ── Step 3: Allocate/reuse outputs ───────────────────────────────────────
  int numActualOutputs = std::min(slot.numOutputs, static_cast<int>(outputShapes.size()));
  static thread_local std::vector<NDArray*> outputs;
  outputs.resize(numActualOutputs);

  // ── In-place fusion: reuse input buffer as output ──
  // When FusionPass marks a slot for in-place execution (e.g., element-wise chain),
  // the first output reuses the specified input buffer instead of allocating new memory.
  // This eliminates intermediate buffer allocations in fused chains.
  if (slot.inPlaceFused && slot.inPlaceFusedInputIdx >= 0 &&
      slot.inPlaceFusedInputIdx < slot.numInputs && numActualOutputs >= 1) {
    NDArray* inPlaceBuffer = inputs[slot.inPlaceFusedInputIdx];
    if (inPlaceBuffer != nullptr) {
      // Verify the in-place buffer has compatible shape with expected output
      const LongType* expectedShape = outputShapes[0];
      if (shape::equalsSoft(inPlaceBuffer->shapeInfo(), expectedShape) &&
          ArrayOptions::dataType(inPlaceBuffer->shapeInfo()) == ArrayOptions::dataType(expectedShape)) {
        outputs[0] = inPlaceBuffer;
        int slotIdx = slot.outputSlotIndices[0];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          outputSlots_[slotIdx] = inPlaceBuffer;
          // Don't cache in-place outputs — they share buffers with upstream ops
        }

        // Allocate remaining outputs (if any) normally
        for (int i = 1; i < numActualOutputs; i++) {
          int si = slot.outputSlotIndices[i];
          if (si < 0) {
            // Untracked output — try reuse from cache
            int cacheIdx = stepIdx * MAX_OUTPUTS_PER_SLOT + i;
            if (cacheIdx < untrackedOutputCacheSize_) {
              NDArray* cached = untrackedOutputCache_[cacheIdx];
              if (cached != nullptr && shape::equalsSoft(cached->shapeInfo(), outputShapes[i]) &&
                  ArrayOptions::dataType(cached->shapeInfo()) == ArrayOptions::dataType(outputShapes[i])) {
                outputs[i] = cached;
                continue;
              }
              delete cached;
              untrackedOutputCache_[cacheIdx] = nullptr;
            }
            outputs[i] = new NDArray(const_cast<LongType*>(outputShapes[i]), true);
            if (cacheIdx < untrackedOutputCacheSize_) {
              untrackedOutputCache_[cacheIdx] = outputs[i];
            }
            continue;
          }
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          int rank = shape::rank(shapeInfo);
          std::vector<LongType> shape(rank);
          for (int d = 0; d < rank; d++) shape[d] = shapeInfo[d + 1];
          outputs[i] = new NDArray(order, shape, dt);
          outputSlots_[si] = outputs[i];
          slotArrayCache_[si] = outputs[i];
        }

        goto step4_execute;  // Skip normal allocation path
      }
      // Shape mismatch — fall through to normal allocation
    }
  }

  // ── View-capable ops: share input 0's DataBuffer for output 0 ──────────
  // For reshape/expand_dims/squeeze, create output that wraps input 0's buffer.
  // The op sees x->dataBuffer() == z->dataBuffer() → returns OK → no copy kernel.
  // Safety: only when input is C-contiguous (ews==1), so standard output strides are correct.
  if (slot.isViewCapableOp && slot.numInputs >= 1 && numActualOutputs >= 1) {
    NDArray* input0 = inputs[0];
    if (input0 != nullptr && input0->dataBuffer() != nullptr &&
        input0->ews() == 1 && input0->ordering() == 'c') {
      const LongType* outShapeInfo = outputShapes[0];
      LongType outLen = shape::length(outShapeInfo);
      LongType inLen = input0->lengthOf();
      if (outLen > 0 && outLen <= inLen) {
        int slotIdx = slot.outputSlotIndices[0];
        NDArray* view = new NDArray(input0->dataBuffer(),
                                     const_cast<LongType*>(outShapeInfo));
        outputs[0] = view;
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          NDArray* old = slotArrayCache_[slotIdx];
          if (old != nullptr && old != view) {
            pendingClose_.push_back(old);
          }
          outputSlots_[slotIdx] = view;
          slotArrayCache_[slotIdx] = view;
          slotIsViewProducer_[slotIdx] = true;
        }

        // Allocate remaining outputs normally (rare for these ops)
        for (int i = 1; i < numActualOutputs; i++) {
          int si = slot.outputSlotIndices[i];
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          int rank = shape::rank(shapeInfo);
          std::vector<LongType> shape(rank);
          for (int d = 0; d < rank; d++) shape[d] = shapeInfo[d + 1];
          outputs[i] = new NDArray(order, shape, dt);
          if (si >= 0 && si < totalOutputSlots_) {
            outputSlots_[si] = outputs[i];
            slotArrayCache_[si] = outputs[i];
          }
        }
        goto step4_execute;
      }
    }
  }

  for (int i = 0; i < numActualOutputs; i++) {
    int slotIdx = slot.outputSlotIndices[i];
    if (slotIdx < 0) {
      // Untracked output — try reuse from cache (critical for CUDA graph capture
      // where cudaMallocAsync on the captured stream is deferred and may fail).
      int cacheIdx = stepIdx * MAX_OUTPUTS_PER_SLOT + i;
      if (cacheIdx < untrackedOutputCacheSize_) {
        NDArray* cached = untrackedOutputCache_[cacheIdx];
        if (cached != nullptr) {
          const LongType* cachedShape = cached->shapeInfo();
          if (shape::equalsSoft(cachedShape, outputShapes[i]) &&
              ArrayOptions::dataType(cachedShape) == ArrayOptions::dataType(outputShapes[i])) {
            outputs[i] = cached;
            continue;
          }
          // Shape mismatch — evict old cached array
          delete cached;
          untrackedOutputCache_[cacheIdx] = nullptr;
        }
      }
      // Allocate new and cache
      outputs[i] = new NDArray(const_cast<LongType*>(outputShapes[i]), true);
      if (cacheIdx < untrackedOutputCacheSize_) {
        untrackedOutputCache_[cacheIdx] = outputs[i];
      }
      continue;
    }

    const LongType* shapeInfo = outputShapes[i];
    auto dt = ArrayOptions::dataType(shapeInfo);
    auto order = shape::order(shapeInfo);
    int rank = shape::rank(shapeInfo);

    // Try to reuse cached array from prior execution
    NDArray* cached = slotArrayCache_[slotIdx];
    if (cached != nullptr) {
      const LongType* cachedShape = cached->shapeInfo();
      if (shape::equalsSoft(cachedShape, shapeInfo) &&
          ArrayOptions::dataType(cachedShape) == dt) {
        // Shape matches — reuse cached buffer.
        // ALWAYS nullify reused arrays. Some ops don't fully overwrite their output,
        // causing stale data accumulation across decode steps.
        cached->nullify();
        outputs[i] = cached;
        outputSlots_[slotIdx] = cached;
        continue;
      } else {
        // Shape doesn't match — evict cached array to pending close
        pendingCloseBytes_ += cached->lengthOf() * cached->sizeOfT();
        pendingClose_.push_back(cached);
        slotArrayCache_[slotIdx] = nullptr;
      }
    }

    // Allocate new output
    std::vector<LongType> shape(rank);
    for (int d = 0; d < rank; d++) {
      shape[d] = shapeInfo[d + 1];
    }
    
    // Check if this slot has a max-allocation size configured
    auto maxIt = outputSlotMaxSizes_.find(slotIdx);
    if (maxIt != outputSlotMaxSizes_.end() && maxIt->second > 0) {
      // Max-allocation mode: check if we already allocated at max size
      if (maxAllocatedSlots_.find(slotIdx) == maxAllocatedSlots_.end()) {
        // First time: allocate at max size
        LongType maxElements = maxIt->second;
        std::vector<LongType> maxShape = shape;
        
        // Calculate which dimension to scale based on shape rank and maxKvCacheLen_
        // For KV cache [batch, numHeads, seqLen, headDim], we scale seqLen (dim 2)
        if (rank == 4 && maxKvCacheLen_ > 0 && shape[2] > 0 && shape[2] < maxKvCacheLen_) {
          // This looks like a KV cache shape - scale seq dimension
          maxShape[2] = maxKvCacheLen_;
        } else if (rank == 4 && maxKvCacheLen_ > 0 && shape[1] > 0 && shape[1] < maxKvCacheLen_) {
          // Alternative KV cache format [batch, seqLen, numHeads, headDim] - scale dim 1
          maxShape[1] = maxKvCacheLen_;
        } else {
          // Default: scale last dimension to reach max elements
          LongType currentElements = 1;
          for (int d = 0; d < rank; d++) currentElements *= shape[d];
          if (currentElements > 0 && maxElements > currentElements) {
            LongType scale = maxElements / currentElements;
            if (scale > 1) {
              maxShape[rank - 1] *= scale;
            }
          }
        }
        
        sd_printf("NativeDynamicShapePlan: max-allocating slot %d, current shape=[%lld,%lld,%lld,%lld], max shape=[%lld,%lld,%lld,%lld]\n",
                  slotIdx, shape[0], rank>1?shape[1]:0, rank>2?shape[2]:0, rank>3?shape[3]:0,
                  maxShape[0], maxShape.size()>1?maxShape[1]:0, maxShape.size()>2?maxShape[2]:0, maxShape.size()>3?maxShape[3]:0);
        
        NDArray* maxOut = nullptr;
        try {
          maxOut = new NDArray(order, maxShape, dt);
          maxOut->nullify();  // Zero the entire buffer
        } catch (const std::exception& e) {
          sd_printf("NativeDynamicShapePlan: max-allocation FAILED at slot %d (%s): %s\n",
                    stepIdx, slot.opName.c_str(), e.what());
          // Fall back to regular allocation
          maxOut = new NDArray(order, shape, dt);
          if (slot.needsZeroedOutput) maxOut->nullify();
        }
        
        outputs[i] = maxOut;
        outputSlots_[slotIdx] = maxOut;
        slotArrayCache_[slotIdx] = maxOut;
        maxAllocatedSlots_.insert(slotIdx);
        continue;
      }
      // Already max-allocated: reuse the cached buffer (it's at max size)
      NDArray* cached = slotArrayCache_[slotIdx];
      if (cached != nullptr) {
        // The cached buffer is at max size
        outputs[i] = cached;
        outputSlots_[slotIdx] = cached;
        continue;
      }
    }
    
    NDArray* out = nullptr;
    try {
      out = new NDArray(order, shape, dt);
      if (slot.needsZeroedOutput) {
        out->nullify();
      }
    } catch (const std::exception& e) {
      sd_printf("NativeDynamicShapePlan: output ALLOC EXCEPTION at slot %d (%s) output[%d]: %s\n",
                stepIdx, slot.opName.c_str(), i, e.what());
      return Status::KERNEL_FAILURE;
    }

    outputs[i] = out;
    outputSlots_[slotIdx] = out;
    slotArrayCache_[slotIdx] = out;
  }

  step4_execute:

  // ── Step 4: Configure context and execute ────────────────────────────────
  auto& ctx = *contextPool_[stepIdx];

  // Set inputs
  for (int i = 0; i < slot.numInputs; i++) {
    ctx.setInputArray(i, inputs[i]);
  }

  // Set outputs
  for (int i = 0; i < numActualOutputs; i++) {
    ctx.setOutputArray(i, outputs[i]);
  }

  // Set arguments
  if (slot.numIArgs > 0) ctx.setIArguments(slot.iArgs, slot.numIArgs);
  if (slot.numTArgs > 0) ctx.setTArguments(slot.tArgs, slot.numTArgs);
  if (slot.numBArgs > 0) ctx.setBArguments(slot.bArgs, slot.numBArgs);
  if (slot.numDArgs > 0) ctx.setDArguments(slot.dArgs, slot.numDArgs);

  // Skip redundant shape inference inside op->execute() -> prepareOutputs().
  // We've already computed and validated shapes in Step 2 and allocated matching outputs.
  // During CUDA graph capture, calculateOutputShape() can trigger ConstantShapeHelper
  // operations that use synchronous CUDA APIs (cudaMemcpy), breaking the capture.
  ctx.setShapeFunctionOverride(true);

  // Execute
  auto status = slot.op->execute(&ctx);

  // Mark all outputs as device-current after execution.
  // Legacy transform ops (LegacyTransformFloatOp, LegacyTransformSameOp,
  // LegacyTransformStrictOp) call prepareSpecialUse before GPU kernel execution
  // but never call registerSpecialUse after. Without this, the output's device
  // write counter is not incremented, and the next op's prepareSpecialUse sees
  // stale device data and syncs HOST→DEVICE, overwriting correct GPU values with zeros.
  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      outputs[i]->tickWriteDevice();
    }
  }

  // ── Step 5: View producer handling ────────────────────────────────────────
  // Some ops (Reshape, Transpose, etc.) ignore their pre-allocated output and
  // return a VIEW of one of their inputs.  We must detect this and update
  // outputSlots_ to point to the actual view, not the pre-allocated buffer.
  //
  // On the first execution we discover which slots are view producers.
  // On ALL subsequent executions we must STILL update outputSlots_ for known
  // view producers, because the view changes each execution (different input).
  {
    auto& ctxOutputs = ctx.fastpath_out();
    for (int i = 0; i < numActualOutputs && i < static_cast<int>(ctxOutputs.size()); i++) {
      int si = slot.outputSlotIndices[i];
      if (si < 0) continue;

      if (!viewProducerDetectionDone_) {
        // First pass: detect view producers
        if (ctxOutputs[i] != outputs[i]) {
          slotIsViewProducer_[si] = true;
          outputSlots_[si] = ctxOutputs[i];
        }
      } else if (slotIsViewProducer_[si]) {
        // Subsequent passes: always update outputSlots_ for known view producers
        // because the view points to a NEW input buffer each execution
        outputSlots_[si] = ctxOutputs[i];
      }
    }
  }

  // Mark slot as frozen-context-ready for subsequent shapes-frozen executions.
  // The context is now fully configured with inputs, outputs, and arguments.
  // In-place fused and view-producer slots also get the fast path — their
  // context outputs point to the correct arrays after this execution.
  if (shapesFrozen_ && executeCount_ > 0 && status == Status::OK) {
    slot.frozenContextReady = true;
  }

  return status;
}

// ─── Shape key computation ──────────────────────────────────────────────────

LongType NativeDynamicShapePlan::computeShapeKey(
    NativeSlot& slot, NDArray** inputs, int numInputs) {
  // FNV-1a style hash
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  // Mix op identity
  mix(slot.opHash);

  // Mix input shapes and dtypes
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i] == nullptr) continue;
    const LongType* si = inputs[i]->shapeInfo();
    int rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
    }
    mix(static_cast<LongType>(inputs[i]->dataType()));
  }

  // Also mix literal values for tiny integer/bool inputs.
  // These arrays are commonly shape/control tensors; their shape often stays
  // constant while values change across decode steps (e.g., KV length growth).
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i] == nullptr) continue;
    auto dt = inputs[i]->dataType();
    auto len = inputs[i]->lengthOf();
    if ((dt == INT32 || dt == INT64 || dt == BOOL) && len > 0 && len <= 32) {
      inputs[i]->syncToHost();
      for (LongType j = 0; j < len; j++) {
        if (dt == BOOL) {
          mix(static_cast<LongType>(inputs[i]->e<bool>(j)));
        } else {
          mix(inputs[i]->e<LongType>(j));
        }
      }
    }
  }

  return key;
}

// ─── Segment shape key computation ──────────────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentShapeKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  // Hash the shapes of all inputs referenced by slots in this segment.
  // This includes BOTH external inputs AND cross-segment inputs (outputs from
  // prior segments). Cross-segment inputs are critical because data-dependent
  // ops like Where produce different output shapes on each call — if we only
  // hash external inputs, we'd miss shape changes from prior segments and
  // incorrectly skip warmup before capture.
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  // Mix shape and dtype only — NOT small-int values.
  //
  // With strict capturability (all outputShapeDependsOnInputValues ops are non-capturable),
  // capturable segments never contain ops whose output SHAPE depends on input VALUES.
  // Value changes in small INT/BOOL tensors (e.g., position_ids changing each step) do NOT
  // affect the CUDA graph's validity — the capture buffer mechanism handles these via
  // device-to-device copy before each replay. Mixing values here would cause the shape key
  // to change every step for any segment consuming position_ids or similar tensors, triggering
  // spurious instability detection → binary splits → captureFailed for stable segments.
  //
  // Stability is correctly tracked by the capture buffer byte-size check (line ~1049):
  // if an external/cross-segment input grows (e.g., KV cache), its byte count changes →
  // captureBuffersOk=false → re-capture. If it stays the same size but changes values,
  // the D2D copy in the replay path updates the capture buffer with fresh values → correct.
  auto mixArraySignature = [&](NDArray* arr) {
    if (arr == nullptr) return;

    const LongType* si = arr->shapeInfo();
    int rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
    }
    // Mix total byte length so that shape changes (not just rank changes) are detected.
    // E.g., [1,4,N,64] → [1,4,N+1,64] has the same rank and strides but different length.
    mix(static_cast<LongType>(arr->lengthOf()));
    mix(static_cast<LongType>(arr->dataType()));
  };

  // Mix segment identity
  mix(seg.startSlot);
  mix(seg.endSlot);

  // Build set of output slot indices produced by this segment, so we can
  // distinguish cross-segment inputs (from prior segments) from intra-segment
  // inputs (produced within this segment). Only cross-segment inputs need
  // hashing since intra-segment shapes are determined by this segment's execution.
  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  // Mix shapes of external and cross-segment inputs used by this segment
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External input
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mixArraySignature(externalInputs[extIdx]);
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        // Cross-segment input — produced by a prior segment, available in outputSlots_
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mixArraySignature(outputSlots_[srcIdx]);
        }
      }
    }
  }

  return key;
}

// ─── Segment input address key computation ──────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentInputAddrKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  // Hash the GPU buffer addresses (specialBuffer) of ALL inputs referenced by
  // this segment. CUDA graphs record exact memory addresses during capture.
  // If any input buffer is reallocated between executions (e.g., position_ids
  // recreated each decoder step), the captured graph would read from stale/freed
  // addresses → CUDA error 700. Compare this key before replay to detect changes.
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  // Build set of output slot indices produced by this segment
  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External input — hash its GPU buffer address
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mix(reinterpret_cast<LongType>(externalInputs[extIdx]->specialBuffer()));
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        // Cross-segment input — hash its GPU buffer address
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mix(reinterpret_cast<LongType>(outputSlots_[srcIdx]->specialBuffer()));
        }
      }
      // Intra-segment inputs use cached buffers (same addresses) — skip
    }
  }

  return key;
}

// ─── Statistics ─────────────────────────────────────────────────────────────

int NativeDynamicShapePlan::getNumCapturedGraphSegments() const {
  int count = 0;
#ifdef SD_CUDA
  for (const auto& seg : segments_) {
    if (seg.cachedGraph) count++;
  }
#endif
  return count;
}

int NativeDynamicShapePlan::getTotalGraphReplays() const {
  return totalGraphReplays_;
}

// ─── Memory management ─────────────────────────────────────────────────────

void NativeDynamicShapePlan::flushPendingClose(void* stream) {
  // Delete evicted NDArrays (shape mismatch during slot cache reuse).
  // NDArray destructor handles GPU memory deallocation via DataBuffer::deleteSpecial()
  // which calls cudaFreeAsync through CudaMemoryPool.
  for (auto* arr : pendingClose_) {
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
    // each step via kvScatter even though the GPU pointer stays the same
    for (auto& seg : segments_) {
      for (auto& cb : seg.captureBuffers) {
        for (int i = 0; i < numMappings; i++) {
          if (cb.externalInputIndex == kvCacheMappings_[i].pastInputExternalIdx) {
            cb.neverSkipCopy = true;
            break;
          }
        }
      }
    }

    sd_printf("NativeDynamicShapePlan: KV cache retention configured: %d mappings, maxLen=%d, initialPos=%d\n",
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

#ifdef SD_CUDA
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
#endif // SD_CUDA

void NativeDynamicShapePlan::scatterKvEntries(NDArray** externalInputs, int numExt, void* stream) {
  if (!kvCacheRetentionEnabled_ || kvCacheNumMappings_ == 0) return;

  // Scatter present KV outputs into external input (static) buffers using
  // the kvScatter CUDA kernel. No operator() calls, no heap allocations.
  // The external buffers are then copied into capture buffers by the next step's
  // frozen fast path (neverSkipCopy=true for KV inputs ensures the copy happens).
  // Use the execution stream (if provided) so scatter runs on the same stream as
  // the graph — avoids cross-stream synchronization overhead.
  auto* lc = LaunchContext::defaultContext();
  cudaStream_t* savedStream = nullptr;
#ifdef SD_CUDA
  if (stream != nullptr) {
    savedStream = lc->getCudaStream();
    lc->setCudaStream(static_cast<cudaStream_t*>(stream));
  }
#endif

  int scattered = 0, skipped = 0;
  for (int m = 0; m < kvCacheNumMappings_; m++) {
    KvCacheMapping& mapping = kvCacheMappings_[m];

    int presentSlotIdx = mapping.presentOutputSlotIdx;
    if (presentSlotIdx < 0 || presentSlotIdx >= totalOutputSlots_) { skipped++; continue; }
    NDArray* presentKv = slotArrayCache_[presentSlotIdx];
    if (presentKv == nullptr) { skipped++; continue; }

    int extIdx = mapping.pastInputExternalIdx;
    if (extIdx < 0 || extIdx >= numExt) { skipped++; continue; }
    NDArray* staticBuf = externalInputs[extIdx];
    if (staticBuf == nullptr) { skipped++; continue; }

    if (presentKv->rankOf() != 4 || staticBuf->rankOf() != 4) { skipped++; continue; }

#ifdef SD_CUDA
    // Direct CUDA kernel — no heap allocations, no operator()
    ops::helpers::kvScatter(presentKv, staticBuf, kvCachePosition_, lc);
#else
    // CPU fallback: operator() + assign()
    int seqDim = mapping.seqDim;
    int rank = presentKv->rankOf();
    LongType lastPos = presentKv->sizeAt(seqDim) - 1;
    std::vector<LongType> srcIdx(rank * 2), dstIdx(rank * 2);
    for (int d = 0; d < rank; d++) {
      if (d == seqDim) {
        srcIdx[d*2] = lastPos; srcIdx[d*2+1] = lastPos + 1;
        dstIdx[d*2] = kvCachePosition_; dstIdx[d*2+1] = kvCachePosition_ + 1;
      } else {
        srcIdx[d*2] = 0; srcIdx[d*2+1] = 0;
        dstIdx[d*2] = 0; dstIdx[d*2+1] = 0;
      }
    }
    NDArray* srcSlice = (*presentKv)(srcIdx, true);
    NDArray* dstSlice = (*staticBuf)(dstIdx, true);
    dstSlice->assign(srcSlice);
    delete srcSlice;
    delete dstSlice;
#endif
    scattered++;
  }

#ifdef SD_CUDA
  // Restore original stream on the default context
  if (savedStream != nullptr) {
    lc->setCudaStream(savedStream);
  }
#endif

  if (traceEnabled_ && (skipped > 0 || executeCount_ <= 3)) {
    sd_printf("KV scatter: %d scattered, %d skipped, pos=%d\n", scattered, skipped, kvCachePosition_);
  }
}

// ─── Graph segmentation for CUDA Graphs ─────────────────────────────────────

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
    if (slot.isDataDependent) return false;
    // Value-dep-shape ops must always run slot-by-slot, regardless of input source.
    // Their output shapes depend on runtime VALUES (not just shapes), so the segment
    // shape key (which hashes input shapes) can't detect when their output shapes change.
    // A CUDA graph replay with stale output shapes produces wrong results.
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
      segments_.push_back(current);

      // Start new segment
      current = GraphSegment();
      current.startSlot = i;
      current.isCapturable = thisCapturable;
    }
  }

  // Finalize last segment
  current.endSlot = numSlots_ - 1;
  segments_.push_back(current);

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
  sd_printf("NativeDynamicShapePlan: %d segments (%d capturable: %d static, %d dynamic; covering %d/%d slots)\n",
            (int)segments_.size(), capturableCount,
            staticCapturableCount, dynamicCapturableCount,
            totalCapturable, numSlots_);
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
#ifdef SD_CUDA
    seg.cachedGraph.reset();
    seg.captureBuffers.clear();
    if (seg.captureWorkspacePtr != nullptr) {
      cudaFree(seg.captureWorkspacePtr);
      seg.captureWorkspacePtr = nullptr;
      seg.captureWorkspaceBytes = 0;
    }
#endif
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
  static constexpr int MAX_FROZEN_SEGMENT_SIZE = INT_MAX;

  auto isSlotCapturableFrozen = [this](int idx) -> bool {
    return !slots_[idx].isDataDependent;
  };

  GraphSegment current;
  current.startSlot = 0;
  current.isCapturable = isSlotCapturableFrozen(0);

  for (int i = 1; i < numSlots_; i++) {
    bool thisCapturable = isSlotCapturableFrozen(i);
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);
    int currentSize = i - current.startSlot;
    bool sizeLimit = (current.isCapturable && currentSize >= MAX_FROZEN_SEGMENT_SIZE);

    if (thisCapturable != current.isCapturable || deviceChange || sizeLimit) {
      current.endSlot = i - 1;
      segments_.push_back(current);
      current = GraphSegment();
      current.startSlot = i;
      current.isCapturable = thisCapturable;
    }
  }
  current.endSlot = numSlots_ - 1;
  segments_.push_back(current);

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
        sd_printf("NativeDSP::rebuildSegments: slot %d op='%s' is data-dependent\n",
                  i, slots_[i].opName.c_str());
      }
    }
  }
  if (dataDepCount > 10) {
    sd_printf("NativeDSP::rebuildSegments: ... and %d more data-dependent slots\n",
              dataDepCount - 10);
  }
  sd_printf("NativeDSP::rebuildSegmentsForFrozenShapes: %d -> %d segments (%d/%d slots capturable, %d data-dep)\n",
            oldSegCount, (int)segments_.size(), capturableSlots, numSlots_, dataDepCount);
}

// ─── Adaptive segment splitting ───────────────────────────────────────────────
//
// When a capturable segment's shape key changes for INSTABILITY_THRESHOLD
// consecutive executions, it contains mixed stable+unstable ops (e.g., KV-growing
// attention mixed with stable FFN/projection ops). We binary-split the segment
// at its midpoint. One half contains the unstable ops (keeps splitting until
// small enough to be permanently slot-by-slot); the other half is stable and
// captures cleanly.
//
// Convergence: O(log2(seg_size) * INSTABILITY_THRESHOLD) decode steps.
// For a 150-op segment with THRESHOLD=2: ~14 warmup steps before convergence.
// All transformer layers have the same structure so they converge in parallel,
// not sequentially.
//
// Split semantics:
//   - Sub-segments start fresh (executionCount=0, consecutiveShapeChanges=0)
//   - Stable sub-segments capture on their 2nd execution
//   - Sub-segments that remain unstable and reach MIN_SPLIT_SIZE → captureFailed

void NativeDynamicShapePlan::maybeSplitUnstableSegments() {
  // Quick check: any segment needing a split?
  bool anySplit = false;
  for (auto& seg : segments_) {
    if (seg.needsSplit) { anySplit = true; break; }
  }
  if (!anySplit) return;

  std::vector<GraphSegment> result;
  result.reserve(segments_.size() + 4);  // some extra for splits

  for (auto& seg : segments_) {
    if (!seg.needsSplit) {
      result.push_back(std::move(seg));
      continue;
    }

    int segSize = seg.endSlot - seg.startSlot + 1;
    if (segSize <= GraphSegment::MIN_SPLIT_SIZE) {
      // Too small to split — make permanently slot-by-slot
      seg.needsSplit = false;
      seg.captureFailed = true;
      seg.consecutiveShapeChanges = 0;
      result.push_back(std::move(seg));
      continue;
    }

    // Binary midpoint split: bisect the unstable segment.
    // One half contains the growing-input ops (stays unstable → keeps splitting),
    // the other half is stable (captures cleanly on its next execution).
    // Convergence: O(log2(N) * INSTABILITY_THRESHOLD) decode steps.
    // All 36 transformer layers have the same structure so they converge in parallel
    // (not sequentially), giving ~log2(seg_size) * 2 total warmup steps total.
    {
      int mid = seg.startSlot + segSize / 2;

      auto makeSubSeg = [&](int start, int end) {
        if (start > end) return;
        GraphSegment sub;
        sub.startSlot = start;
        sub.endSlot = end;
        sub.isCapturable = seg.isCapturable;
        sub.executionCount = 0;
        sub.consecutiveShapeChanges = 0;
        sub.needsSplit = false;
#ifdef SD_CUDA
        sub.cachedShapeKey = 0;
#endif
        // Invalidate slot shape caches so sub-segments re-warm correctly
        for (int s = start; s <= end; s++) {
          slots_[s].shapeCacheValid = false;
          slots_[s].cachedShapeKey = 0;
          slots_[s].cachedOutputShapes.clear();
          slots_[s].frozenContextReady = false;
          slots_[s].frozenConstantSlot = false;
        }
        result.push_back(std::move(sub));
      };

      makeSubSeg(seg.startSlot, mid - 1);
      makeSubSeg(mid, seg.endSlot);

      sd_printf("NativeDynamicShapePlan: binary-splitting unstable segment [%d-%d] (%d ops) "
                "at midpoint %d into 2 sub-segments\n",
                seg.startSlot, seg.endSlot, segSize, mid);
    }
  }

  segments_ = std::move(result);
}

// ─── CPU Graph backend integration ──────────────────────────────────────────
GraphBackend* NativeDynamicShapePlan::getCpuGraphBackend() {
  if (cpuGraphBackendChecked_) return cpuGraphBackend_;
  cpuGraphBackendChecked_ = true;

#if HAVE_ONEDNN
  auto& onednn = OneDnnGraphBackend::getInstance();
  if (onednn.isAvailable()) {
    cpuGraphBackend_ = &onednn;
    sd_printf("NativeDynamicShapePlan: using oneDNN Graph backend\n", "");
    return cpuGraphBackend_;
  }
#endif

#if HAVE_ARMCOMPUTE
  auto& acl = AclGraphBackend::getInstance();
  if (acl.isAvailable()) {
    cpuGraphBackend_ = &acl;
    sd_printf("NativeDynamicShapePlan: using ARM ACL backend\n", "");
    return cpuGraphBackend_;
  }
#endif

  cpuGraphBackend_ = nullptr;
  return nullptr;
}

Status NativeDynamicShapePlan::executeSegmentWithCpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  auto* backend = getCpuGraphBackend();
  if (backend == nullptr) return Status::KERNEL_FAILURE;

  // If compilation previously failed validation, never try again
  if (seg.captureFailed) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Check if this segment can be compiled by the backend
  if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
    return Status::KERNEL_FAILURE;  // Caller will fall back to slot-by-slot
  }

  // Compute shape key for cache lookup
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // Compile (or use cache) for this shape
  if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                               outputSlots_, totalOutputSlots_, segShapeKey)) {
    return Status::KERNEL_FAILURE;
  }

  // First execution: validate compilation coverage, then run slot-by-slot
  // to populate outputSlots_ for the backend's tensor wiring.
  if (seg.executionCount == 0) {
    // Validate that all ops were compiled — if any were skipped,
    // the compiled graph will produce stale outputs on replay.
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    bool allCompiled = true;
    for (const auto& entry : audit) {
      if (!entry.wasCompiled) {
        allCompiled = false;
        sd_printf("GRAPH VALIDATION: slot %d (%s) was NOT compiled by %s backend: %s\n",
                  entry.slotIndex, entry.opName.c_str(), backend->name(), entry.reason.c_str());
      }
    }
    if (!allCompiled) {
      sd_printf("GRAPH VALIDATION FAILURE: segment [%d-%d] has ops not covered by %s backend. "
                "Falling back to slot-by-slot to prevent stale outputs.\n",
                seg.startSlot, seg.endSlot, backend->name());
      seg.captureFailed = true;  // Mark as failed — never try again
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }

    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Execute via backend
  seg.shapeKey = segShapeKey;
  auto status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                         outputSlots_, totalOutputSlots_, stream);

  if (status == Status::OK) {
    seg.executionCount++;
    totalGraphReplays_++;
  }

  return status;
}

// ─── GPU Graph backend integration (Triton) ─────────────────────────────────

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

#if HAVE_TRITON
  auto& triton = TritonGraphBackend::getInstance();
  if (triton.isAvailable()) {
    gpuGraphBackend_ = &triton;
    sd_printf("NativeDynamicShapePlan: using Triton GPU compiler backend\n", "");
    return gpuGraphBackend_;
  }
#endif

  gpuGraphBackend_ = nullptr;
  return nullptr;
}

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) return Status::KERNEL_FAILURE;

  // If compilation previously failed validation, never try again
  if (seg.captureFailed) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the Triton backend
  if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // Compute shape key for cache lookup
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // Compile (or use cache) for this shape
  if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                               outputSlots_, totalOutputSlots_, segShapeKey)) {
    return Status::KERNEL_FAILURE;
  }

  // First execution: validate compilation coverage, then run slot-by-slot
  if (seg.executionCount == 0) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    bool allCompiled = true;
    for (const auto& entry : audit) {
      if (!entry.wasCompiled) {
        allCompiled = false;
        sd_printf("TRITON VALIDATION: slot %d (%s) was NOT compiled: %s\n",
                  entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (!allCompiled) {
      sd_printf("TRITON VALIDATION FAILURE: segment [%d-%d] has ops not covered by Triton. "
                "Falling back to CUDA Graphs.\n",
                seg.startSlot, seg.endSlot);
      // Don't set captureFailed — let CUDA Graphs try next
      return Status::KERNEL_FAILURE;
    }

    // Warm-up: run slot-by-slot to populate outputSlots_
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Execute via Triton backend
  seg.shapeKey = segShapeKey;
  auto status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                         outputSlots_, totalOutputSlots_, stream);

  if (status == Status::OK) {
    seg.executionCount++;
    totalGraphReplays_++;
  }

  return status;
}

// ─── fromFlatGraph (delegates to NativePlanCompiler) ─────────────────────────

NativeDynamicShapePlan* NativeDynamicShapePlan::fromFlatGraph(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {
  return NativePlanCompiler::compile(graph, variables, requestedOutputs);
}

// ─── CUDA Graph capture audit and validation ────────────────────────────────

#ifdef SD_CUDA

std::vector<cuda::CaptureAuditEntry> NativeDynamicShapePlan::getHostOnlyOps() const {
  std::vector<cuda::CaptureAuditEntry> result;
  for (const auto& entry : lastCaptureAudit_) {
    if (entry.isHostOnly()) {
      result.push_back(entry);
    }
  }
  return result;
}

void NativeDynamicShapePlan::printCaptureAudit() const {
  if (lastCaptureAudit_.empty()) {
    sd_print("NativeDynamicShapePlan: No capture audit data (no capture has occurred)\n");
    return;
  }

  sd_print("╔══════════════════════════════════════════════════════════════════════════╗\n");
  sd_print("║           CUDA GRAPH CAPTURE AUDIT (per-op node count)                 ║\n");
  sd_print("╠══════════════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Total ops in segment: %zu\n", lastCaptureAudit_.size());
  sd_print("╠══════════════════════════════════════════════════════════════════════════╣\n");

  int hostOnlyCount = 0;
  size_t totalNodes = 0;

  for (const auto& entry : lastCaptureAudit_) {
    totalNodes += entry.nodesContributed;
    if (entry.isHostOnly()) {
      hostOnlyCount++;
    }
  }

  // Print top-10 ops by node contribution (highest first)
  sd_print("║ TOP-10 OPS BY NODE COUNT:\n");
  std::vector<size_t> indices(lastCaptureAudit_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
    return lastCaptureAudit_[a].nodesContributed > lastCaptureAudit_[b].nodesContributed;
  });
  int topN = std::min(static_cast<int>(indices.size()), 10);
  for (int i = 0; i < topN; i++) {
    const auto& entry = lastCaptureAudit_[indices[i]];
    sd_printf("║  #%2d [slot %3d] %-25s  nodes: %3zu%s\n",
              i + 1, entry.slotIndex, entry.opName.c_str(), entry.nodesContributed,
              entry.isHostOnly() ? "  *** HOST-ONLY ***" : "");
  }

  sd_print("╠══════════════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Total CUDA graph nodes: %zu from %zu ops\n",
            totalNodes, lastCaptureAudit_.size());
  sd_printf("║ Host-only ops: %d, Node-contributing ops: %zu\n",
            hostOnlyCount, lastCaptureAudit_.size() - hostOnlyCount);
  if (hostOnlyCount > 0) {
    sd_printf("║ *** WARNING: %d HOST-ONLY ops detected! ***\n", hostOnlyCount);
    sd_print("║ Host-only ops do work during capture but NOT during replay.\n");
    sd_print("║ Their outputs will be STALE on the 2nd+ graph execution.\n");
  } else {
    sd_print("║ All ops contributed CUDA graph nodes. Graph is complete.\n");
  }
  sd_print("╚══════════════════════════════════════════════════════════════════════════╝\n");
}

bool NativeDynamicShapePlan::validateCapturedGraph(int segmentIndex) const {
  if (lastCaptureAudit_.empty()) return true;  // No audit data = no validation

  bool allOpsHaveNodes = true;

  for (const auto& entry : lastCaptureAudit_) {
    if (entry.isHostOnly()) {
      allOpsHaveNodes = false;
      sd_printf("CUDA GRAPH VALIDATION FAILURE: slot %d (%s) contributed 0 CUDA graph nodes. "
                "This op does host-only work that will NOT be replayed.\n",
                entry.slotIndex, entry.opName.c_str());
    }
  }

  return allOpsHaveNodes;
}

#endif  // SD_CUDA

// ─── CPU Graph compilation audit and validation ─────────────────────────────

void NativeDynamicShapePlan::printCompilationAudit() const {
  if (lastCompilationAudit_.empty()) {
    sd_print("NativeDynamicShapePlan: No compilation audit data\n");
    return;
  }

  const char* backendName = cpuGraphBackend_ ? cpuGraphBackend_->name() : "unknown";

  sd_print("╔══════════════════════════════════════════════════════════════════╗\n");
  sd_printf("║        CPU GRAPH COMPILATION AUDIT (%s backend)\n", backendName);
  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Total ops in segment: %zu\n", lastCompilationAudit_.size());
  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");

  int skippedCount = 0;
  int compiledCount = 0;

  for (const auto& entry : lastCompilationAudit_) {
    if (entry.wasCompiled) {
      compiledCount++;
      if (entry.reason.empty()) {
        sd_printf("║  [slot %3d] %-30s  COMPILED\n",
                  entry.slotIndex, entry.opName.c_str());
      } else {
        sd_printf("║  [slot %3d] %-30s  COMPILED (%s)\n",
                  entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    } else {
      skippedCount++;
      sd_printf("║  [slot %3d] %-30s  *** SKIPPED *** (%s)\n",
                entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
    }
  }

  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Compiled: %d, Skipped: %d out of %zu total ops\n",
            compiledCount, skippedCount, lastCompilationAudit_.size());
  if (skippedCount > 0) {
    sd_printf("║ *** WARNING: %d ops were SKIPPED by %s backend! ***\n",
              skippedCount, backendName);
    sd_print("║ Skipped ops execute during warm-up but NOT during graph replay.\n");
    sd_print("║ Their outputs will be STALE on the 2nd+ graph execution.\n");
    sd_print("║ Segment will fall back to slot-by-slot execution.\n");
  } else {
    sd_print("║ All ops compiled successfully. Graph is complete.\n");
  }
  sd_print("╚══════════════════════════════════════════════════════════════════╝\n");
}

bool NativeDynamicShapePlan::validateCompiledCpuGraph(int segmentIndex) const {
  if (lastCompilationAudit_.empty()) return true;  // No audit data = no validation

  bool allOpsCompiled = true;

  for (const auto& entry : lastCompilationAudit_) {
    if (!entry.wasCompiled) {
      allOpsCompiled = false;
      const char* backendName = cpuGraphBackend_ ? cpuGraphBackend_->name() : "unknown";
      sd_printf("CPU GRAPH VALIDATION FAILURE: slot %d (%s) was NOT compiled by %s backend: %s\n",
                entry.slotIndex, entry.opName.c_str(), backendName, entry.reason.c_str());
    }
  }

  return allOpsCompiled;
}

}  // namespace graph
}  // namespace sd
