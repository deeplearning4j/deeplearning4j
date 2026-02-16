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
#include <helpers/ConstantShapeHelper.h>
#include <helpers/helper_hash.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <system/Environment.h>
#ifdef SD_CUDA
#include <memory/cuda/CudaMemoryPool.h>
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
      contextPool_(nullptr), viewProducerDetectionDone_(false),
      pendingCloseBytes_(0), cudaGraphsEnabled_(false), totalGraphReplays_(0),
      minCaptureSegmentSize_(10), maxCaptureSegmentSize_(50),
      shapesFrozen_(false), executeCount_(0), executionTimingEnabled_(false),
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false),
      gpuGraphBackend_(nullptr), gpuGraphBackendChecked_(false),
      untrackedOutputCache_(nullptr), untrackedOutputCacheSize_(0),
      kvCacheRetentionEnabled_(false), kvCachePosition_(0), kvCacheMaxLen_(0),
      kvCacheNumMappings_(0), kvCacheMappings_(nullptr)
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
  // Free capture buffers from all segments
  for (auto& seg : segments_) {
    for (auto& cb : seg.captureBuffers) {
      delete cb.buffer;
    }
    seg.captureBuffers.clear();
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
    slot.isIdentityOp = (slot.opName == "identity");

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

  // Step 0: Clear stale CUDA errors and error references from prior execution.
  // Async GPU errors from the previous execute() call may not surface until the
  // next CUDA API call, causing false failures. Clear them proactively.
  // Also clear the custom error reference so failed graph captures from prior
  // executions don't cause false positives at stream sync time.
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
#ifdef SD_CUDA
  cudaGetLastError();

  // Pre-execution flush: free arrays evicted during the previous call's warmup.
  // Without this, 2-3 calls' worth of evicted arrays accumulate before the next
  // mid-execution flush (every 100 steps), causing OOM on shape buffer allocation.
  flushPendingClose(stream);

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
    if (segment.isCapturable && !segment.captureFailed &&
        (cudaGraphsEnabled_ || getGpuGraphBackend() != nullptr)) {
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
          if (status != Status::OK) return status;
          // Check if this segment actually replayed a graph (cachedGraph exists and didn't fail)
          segUsedGraph = (segment.cachedGraph != nullptr && !segment.captureFailed);
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
        snprintf(buf, sizeof(buf), "CUDA error after segment [%d-%d]: %d (%s)",
                 segment.startSlot, segment.endSlot,
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

  // Step 3.5: KV cache retention — scatter new entries into static input buffers
  if (kvCacheRetentionEnabled_) {
    scatterKvEntries(externalInputs, numExternalInputs, stream);
  }

  auto tOutputsDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Step 4: Final flush
  flushPendingClose(stream);

  auto tFlushDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Track execution count for shapes-frozen optimization
  if (shapesFrozen_) executeCount_++;

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

      // Async copy data into the capture buffer
      cudaMemcpyAsync(cb.buffer->specialBuffer(), src->specialBuffer(),
                      srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
    }

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
      // Free old capture buffers
      for (auto& cb : seg.captureBuffers) {
        delete cb.buffer;
      }
      seg.captureBuffers.clear();
      seg.cachedGraph.reset();
      // Fall through to warmup + re-capture
    } else {
      // Launch failed — invalidate and fall through
      sd_printf("NativeDynamicShapePlan: graph replay failed for segment [%d-%d], "
                "falling back to slot-by-slot\n", seg.startSlot, seg.endSlot);
      seg.cachedGraph.reset();
    }
  }

  // ── WARM-UP: first execution or shape change populates slot cache ──
  // When shapes change (autoregressive seq_len grows, different batch size, etc.),
  // the slot cache has arrays from the OLD shapes. Capture requires shape inference
  // which may call syncToHost (e.g., gather reads indices). syncToHost is forbidden
  // during CUDA graph capture → error 901. So we must do a warmup pass WITHOUT capture
  // to populate the slot cache with the new shapes before attempting capture.
  bool shapeChanged = (seg.cachedShapeKey != segShapeKey);
  if (seg.executionCount == 0 || (shapeChanged && !seg.captureFailed)) {
    // Invalidate the old graph if shapes changed
    if (shapeChanged && seg.cachedGraph) {
      seg.cachedGraph.reset();
    }
    seg.cachedShapeKey = segShapeKey;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── CAPTURE: slot cache is warm, attempt to capture CUDA graph ──
  // Shape key changed → invalidate old graph (defensive, already handled above)
  if (seg.cachedGraph && seg.cachedShapeKey != segShapeKey) {
    seg.cachedGraph.reset();
  }

  int segSize = seg.endSlot - seg.startSlot + 1;
  // Don't bother capturing tiny segments (overhead > benefit)
  if (segSize < minCaptureSegmentSize_) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // OOM retry cooldown: if this segment previously failed capture due to OOM,
  // wait until enough executions have passed before retrying.
  if (seg.captureOomRetries > 0 && seg.executionCount < seg.captureRetryAfterExec) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
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
                seg.startSlot, seg.endSlot, segSize,
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

  // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
  // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
  // graph capture on a named stream, this cross-stream allocation breaks capture.
  // By providing an explicit workspace, cuBLAS uses our buffer instead.
  // 32MB covers workspace needs for most GEMM sizes on modern GPUs.
  static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32 MB
  ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
  setCublasWorkspaceForCapture(stream);

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

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External input
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr &&
            extInputToCaptureIdx.find(extIdx) == extInputToCaptureIdx.end()) {
          NDArray* src = externalArrays[extIdx];
          // Allocate a capture buffer matching the source shape and type
          auto srcShapeVec = *src->getShapeAsVector();
          auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                     sd::LaunchContext::defaultContext());
          // Copy initial data
          size_t srcBytes = src->lengthOf() * src->sizeOfT();
          cudaMemcpyAsync(capBuf->specialBuffer(), src->specialBuffer(),
                          srcBytes, cudaMemcpyDeviceToDevice, cudaStr);

          GraphSegment::CaptureBuffer cb;
          cb.buffer = capBuf;
          cb.externalInputIndex = extIdx;
          cb.crossSegmentSlotIdx = -1;
          cb.capturedSize = srcBytes;

          extInputToCaptureIdx[extIdx] = static_cast<int>(seg.captureBuffers.size());
          seg.captureBuffers.push_back(std::move(cb));
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
          cudaMemcpyAsync(capBuf->specialBuffer(), src->specialBuffer(),
                          srcBytes, cudaMemcpyDeviceToDevice, cudaStr);

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

  // Wire external/cross-segment inputs to capture buffers for the capture phase.
  // Save original pointers so we can restore them after capture.
  std::vector<std::pair<int, NDArray*>> savedExternalInputs;  // (extIdx, originalPtr)
  std::vector<std::pair<int, NDArray*>> savedOutputSlots;     // (slotIdx, originalPtr)

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
  tl_graphExecutionActive = true;
  tl_capturedHostPtrs.clear();  // Reset pinned host ptr accumulator for this capture

  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    sd_printf("NativeDynamicShapePlan: graph capture begin failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    tl_graphExecutionActive = false;
    restoreCublasWorkspaceAfterCapture(stream);
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  bool captureOk = true;
  bool captureOomFailure = false;  // Track if failure was OOM (retryable)

  // Save outputSlots_ before capture loop. If capture fails mid-loop and we fall
  // back to slot-by-slot, the capture loop's release processing (releaseAtStep_)
  // may have already nulled cross-segment input slots that the fallback needs.
  // E.g., slot 549 (Where output from segment 1) is released at step 549 during
  // capture, but the fallback re-starts at step 549 and needs to read it again.
  std::vector<NDArray*> preCapOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);

  try {
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
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
    sd_printf("NativeDynamicShapePlan: exception during graph capture: %s\n", e.what());
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
  tl_graphExecutionActive = false;
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

    // Free capture buffers on failure — they won't be needed
    for (auto& cb : seg.captureBuffers) {
      delete cb.buffer;
    }
    seg.captureBuffers.clear();

    // Clear error reference so the capture failure doesn't propagate to the caller
    // when the slot-by-slot fallback succeeds.
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Helper lambda to clean up capture buffers on failure
  auto cleanupCaptureBuffersOnFailure = [&seg]() {
    for (auto& cb : seg.captureBuffers) {
      delete cb.buffer;
    }
    seg.captureBuffers.clear();
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
    // Clear sticky CUDA errors and reset stream
    cudaGetLastError();
    cudaStreamSynchronize(cudaStr);
    cudaGetLastError();
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
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
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Clear any sticky error that might have been left by updateStatistics or instantiate
  cudaGetLastError();

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
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    // Restore outputSlots_ — capture loop releases may have cleared cross-segment inputs
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Transfer pinned host buffers to the graph handle for lifetime management.
  // These persist for graph replay (H2D memcpy nodes reference them).
  for (auto* ptr : tl_capturedHostPtrs) {
    handle->addCapturedHostPtr(ptr);
  }
  tl_capturedHostPtrs.clear();

  // Cache the graph for future replays
  seg.cachedGraph = handle;
  seg.cachedShapeKey = segShapeKey;
  seg.executionCount++;
  totalGraphReplays_++;

  // Reset OOM retry state on success (capture may have succeeded on a retry)
  if (seg.captureOomRetries > 0) {
    sd_printf("NativeDynamicShapePlan: graph capture SUCCEEDED on OOM retry %d for segment [%d-%d]\n",
              seg.captureOomRetries, seg.startSlot, seg.endSlot);
    seg.captureOomRetries = 0;
    seg.captureRetryAfterExec = 0;
  }

  sd_printf("NativeDynamicShapePlan: captured CUDA graph for segment [%d-%d] "
            "(%zu nodes, %zu edges)\n",
            seg.startSlot, seg.endSlot,
            handle->getNumNodes(), handle->getNumEdges());

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

  // ── Fast path: frozen context ────────────────────────────────────────────
  // When shapes are frozen and this slot's context was already configured on a
  // prior execution, skip input gathering, shape inference, output allocation,
  // and context setup. Just execute the op and handle view producers.
  // outputSlots_ was pre-populated from slotArrayCache_ in execute(), so
  // downstream ops already have the right array pointers.
  if (slot.frozenContextReady) {
    auto& ctx = *contextPool_[stepIdx];

    // For ops that need zeroed output, nullify the cached array
    if (slot.needsZeroedOutput) {
      auto& ctxOuts = ctx.fastpath_out();
      for (int i = 0; i < static_cast<int>(ctxOuts.size()); i++) {
        if (ctxOuts[i] != nullptr) ctxOuts[i]->nullify();
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
        // When shapes are frozen, skip nullify for ops that fully write their output
        // (needsZeroedOutput=false). This avoids ~4441 cudaMemsetAsync calls per step.
        if (slot.needsZeroedOutput || !shapesFrozen_) {
          cached->nullify();
        }
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

  // For data-dependent ops, also mix input values (expensive)
  if (slot.outputShapeDependsOnInputValues) {
    for (int i = 0; i < numInputs; i++) {
      if (inputs[i] == nullptr) continue;
      auto dt = inputs[i]->dataType();
      if (dt == INT32 || dt == INT64) {
        // Sync to host and read values
        inputs[i]->syncToHost();
        auto len = inputs[i]->lengthOf();
        if (len <= 16) {  // Only for small arrays (shape params)
          for (LongType j = 0; j < len; j++) {
            mix(inputs[i]->e<LongType>(j));
          }
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
          const LongType* si = externalInputs[extIdx]->shapeInfo();
          int rank = shape::rank(si);
          mix(rank);
          for (int d = 0; d < rank; d++) {
            mix(si[d + 1]);
          }
          mix(static_cast<LongType>(externalInputs[extIdx]->dataType()));
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        // Cross-segment input — produced by a prior segment, available in outputSlots_
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          const LongType* si = outputSlots_[srcIdx]->shapeInfo();
          int rank = shape::rank(si);
          mix(rank);
          for (int d = 0; d < rank; d++) {
            mix(si[d + 1]);
          }
          mix(static_cast<LongType>(outputSlots_[srcIdx]->dataType()));
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
      kvCacheMappings_[i].presentOutputSlotIdx = mappings[i * 3];
      kvCacheMappings_[i].pastInputExternalIdx = mappings[i * 3 + 1];
      kvCacheMappings_[i].seqDim = mappings[i * 3 + 2];
    }
    kvCacheRetentionEnabled_ = true;
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

void NativeDynamicShapePlan::scatterKvEntries(NDArray** externalInputs, int numExt, void* stream) {
  if (!kvCacheRetentionEnabled_ || kvCacheNumMappings_ == 0) return;

#ifdef SD_CUDA
  // Sync the execution stream before reading op outputs.
  // Model ops ran on this stream; without sync, assign() on the default stream
  // may read stale/incomplete data from presentKv outputs.
  if (stream != nullptr) {
    cudaStream_t execStream = *reinterpret_cast<cudaStream_t*>(stream);
    cudaStreamSynchronize(execStream);
  }
#endif

  int scattered = 0, skipped = 0;
  for (int m = 0; m < kvCacheNumMappings_; m++) {
    KvCacheMapping& mapping = kvCacheMappings_[m];

    // Get the present KV output directly from the absolute output slot index
    int presentSlotIdx = mapping.presentOutputSlotIdx;
    if (presentSlotIdx < 0 || presentSlotIdx >= totalOutputSlots_) { skipped++; continue; }
    NDArray* presentKv = outputSlots_[presentSlotIdx];
    if (presentKv == nullptr) { skipped++; continue; }

    // Get the static past KV input buffer
    int extIdx = mapping.pastInputExternalIdx;
    if (extIdx < 0 || extIdx >= numExt) { skipped++; continue; }
    NDArray* staticBuf = externalInputs[extIdx];
    if (staticBuf == nullptr) { skipped++; continue; }

    int seqDim = mapping.seqDim;
    int rank = presentKv->rankOf();
    if (seqDim < 0 || seqDim >= rank) { skipped++; continue; }

    // presentKv shape: [B, H, maxKvLen+1, D] — new entry is at the last position
    LongType lastPos = presentKv->sizeAt(seqDim) - 1;

    // Build subarray indices using operator() convention:
    // {dim0Start, dim0End, dim1Start, dim1End, ...}
    // When dimStart == dimEnd, it means the whole range for that dimension.
    // For a point index, use {pos, pos+1} (half-open interval).
    std::vector<LongType> srcIdx(rank * 2);
    std::vector<LongType> dstIdx(rank * 2);
    for (int d = 0; d < rank; d++) {
      if (d == seqDim) {
        srcIdx[d * 2] = lastPos;
        srcIdx[d * 2 + 1] = lastPos + 1;
        dstIdx[d * 2] = kvCachePosition_;
        dstIdx[d * 2 + 1] = kvCachePosition_ + 1;
      } else {
        // Whole range: dimStart == dimEnd signals "all"
        srcIdx[d * 2] = 0;
        srcIdx[d * 2 + 1] = 0;
        dstIdx[d * 2] = 0;
        dstIdx[d * 2 + 1] = 0;
      }
    }

    // operator() returns a view (no copy); assign() does strided GPU copy
    // Stream was synced above so presentKv data is ready to read
    NDArray* srcSlice = (*presentKv)(srcIdx, true);  // keepUnities=true for shape compat
    NDArray* dstSlice = (*staticBuf)(dstIdx, true);
    dstSlice->assign(srcSlice);
    delete srcSlice;
    delete dstSlice;
    scattered++;
  }
  if (skipped > 0) {
    sd_printf("KV scatter: %d scattered, %d skipped, pos=%d\n", scattered, skipped, kvCachePosition_);
  }
}

// ─── Graph segmentation for CUDA Graphs ─────────────────────────────────────

void NativeDynamicShapePlan::buildSegments() {
  if (numSlots_ == 0) return;

  GraphSegment current;
  current.startSlot = 0;
  current.isCapturable = !slots_[0].isDataDependent;

  for (int i = 1; i < numSlots_; i++) {
    bool thisDataDependent = slots_[i].isDataDependent;
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);

    // Break segment when data-dependency status changes OR device changes.
    bool capturabilityChanged = (thisDataDependent == current.isCapturable);

    // Also break capturable segments that exceed maxCaptureSegmentSize_.
    // During CUDA graph capture, cudaFreeAsync calls are recorded but NOT
    // executed — all intermediate allocations accumulate simultaneously.
    // Splitting into smaller segments limits peak capture memory.
    int segSize = i - current.startSlot;
    bool sizeExceeded = (maxCaptureSegmentSize_ > 0 && current.isCapturable &&
                         segSize >= maxCaptureSegmentSize_);

    if (capturabilityChanged || deviceChange || sizeExceeded) {
      // End current segment
      current.endSlot = i - 1;
      segments_.push_back(current);

      // Start new segment — capturable if the new slot is not data-dependent
      current = GraphSegment();
      current.startSlot = i;
      current.isCapturable = !thisDataDependent;
    }
  }

  // Finalize last segment
  current.endSlot = numSlots_ - 1;
  segments_.push_back(current);

  // Log segment structure
  int capturableCount = 0, totalCapturable = 0;
  for (auto& seg : segments_) {
    if (seg.isCapturable) {
      capturableCount++;
      totalCapturable += (seg.endSlot - seg.startSlot + 1);
    }
  }
  sd_printf("NativeDynamicShapePlan: %d segments (%d capturable covering %d/%d slots, max segment size %d)\n",
            (int)segments_.size(), capturableCount, totalCapturable, numSlots_, maxCaptureSegmentSize_);
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
    sd_print("NativeDynamicShapePlan: No capture audit data (debug/verbose mode was off during capture)\n");
    return;
  }

  sd_print("╔══════════════════════════════════════════════════════════════════╗\n");
  sd_print("║           CUDA GRAPH CAPTURE AUDIT (per-op node count)         ║\n");
  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Total ops in segment: %zu\n", lastCaptureAudit_.size());
  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");

  int hostOnlyCount = 0;
  size_t totalNodes = 0;

  for (const auto& entry : lastCaptureAudit_) {
    totalNodes += entry.nodesContributed;
    if (entry.isHostOnly()) {
      hostOnlyCount++;
      sd_printf("║  [slot %3d] %-30s  nodes: %3zu  *** HOST-ONLY ***\n",
                entry.slotIndex, entry.opName.c_str(), entry.nodesContributed);
    } else {
      sd_printf("║  [slot %3d] %-30s  nodes: %3zu\n",
                entry.slotIndex, entry.opName.c_str(), entry.nodesContributed);
    }
  }

  sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
  sd_printf("║ Total CUDA graph nodes: %zu from %zu ops\n",
            totalNodes, lastCaptureAudit_.size());
  if (hostOnlyCount > 0) {
    sd_printf("║ *** WARNING: %d HOST-ONLY ops detected! ***\n", hostOnlyCount);
    sd_print("║ Host-only ops do work during capture but NOT during replay.\n");
    sd_print("║ Their outputs will be STALE on the 2nd+ graph execution.\n");
    sd_print("║ These ops must be excluded from CUDA graph segments or\n");
    sd_print("║ must be rewritten to use CUDA kernels for all their work.\n");
  } else {
    sd_print("║ All ops contributed CUDA graph nodes. Graph is complete.\n");
  }
  sd_print("╚══════════════════════════════════════════════════════════════════╝\n");
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
