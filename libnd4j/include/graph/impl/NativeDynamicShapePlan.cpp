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
#include <cstring>

// Include CPU graph backends conditionally
#include <config.h>
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
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
      targetDeviceId(other.targetDeviceId),
      legacyOpType(other.legacyOpType),
      legacyOpNum(other.legacyOpNum),
      cachedShapeKey(other.cachedShapeKey),
      cachedOutputShapes(std::move(other.cachedOutputShapes)),
      shapeCacheValid(other.shapeCacheValid) {
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
    targetDeviceId = other.targetDeviceId;
    legacyOpType = other.legacyOpType;
    legacyOpNum = other.legacyOpNum;
    cachedShapeKey = other.cachedShapeKey;
    cachedOutputShapes = std::move(other.cachedOutputShapes);
    shapeCacheValid = other.shapeCacheValid;

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
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false) {}

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

  // Pre-allocate context pool
  plan->contextPool_ = new Context*[plan->numSlots_];
  for (int i = 0; i < plan->numSlots_; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // Build graph segments for CUDA Graphs
  plan->buildSegments();

  // Detect fusion candidates (logged for diagnostics; not yet applied)
  if (plan->numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(plan->slots_, plan->numSlots_);
    if (!fusions.empty()) {
      sd_printf("NativeDynamicShapePlan: detected %d fusion candidates\n",
                static_cast<int>(fusions.size()));
      for (auto& f : fusions) {
        sd_printf("  fusion: slots %d-%d, type=%d, chain=%d\n",
                  f.startSlot, f.endSlot, static_cast<int>(f.type), f.chainLength);
      }
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

  // Step 1: Clear output slots
  std::memset(outputSlots_, 0, sizeof(NDArray*) * totalOutputSlots_);

  // Step 2: Execute segments
  for (auto& segment : segments_) {
    // Set graph execution flag for ALL graph backends (CUDA Graphs, oneDNN Graph, ACL).
    // This tells DataBuffer::syncToPrimary to skip D2H transfers during graph execution,
    // preventing stream conflicts (CUDA) and unnecessary data movement (CPU graphs).
    bool useGraph = false;

#ifdef SD_CUDA
    if (cudaGraphsEnabled_ && segment.isCapturable && !segment.captureFailed) {
      useGraph = true;
    }
#else
    if (segment.isCapturable && getCpuGraphBackend() != nullptr) {
      useGraph = true;
    }
#endif

    if (useGraph) {
      tl_graphExecutionActive = true;
    }

#ifdef SD_CUDA
    if (useGraph) {
      auto status = executeSegmentWithGraph(segment, externalInputs, numExternalInputs, stream);
      tl_graphExecutionActive = false;
      if (status != Status::OK) return status;
    } else {
      auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
    }
#else
    if (useGraph) {
      auto status = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
      tl_graphExecutionActive = false;
      if (status != Status::OK) {
        // Fall back to slot-by-slot if CPU graph execution fails
        status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
        if (status != Status::OK) return status;
      }
    } else {
      auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) return status;
    }
#endif
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

  // Step 4: Final flush
  flushPendingClose(stream);

  return Status::OK;
}

// ─── Segment execution: slot-by-slot ─────────────────────────────────────────

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    auto status = executeSlot(stepIdx, externalArrays, numExt, stream);
    if (status != Status::OK) {
      sd_printf("NativeDynamicShapePlan: slot %d (%s) failed with status %d\n",
                stepIdx, slots_[stepIdx].opName.c_str(), static_cast<int>(status));
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

Status NativeDynamicShapePlan::executeSegmentWithGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Compute shape key for this segment's inputs
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // ── REPLAY: cached graph with matching shapes ──
  if (seg.cachedGraph && seg.cachedShapeKey == segShapeKey &&
      seg.cachedGraph->getState() == cuda::GraphState::INSTANTIATED) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;

    if (seg.cachedGraph->launchAsync(cudaStr)) {
      totalGraphReplays_++;
      seg.executionCount++;
      return Status::OK;
    }

    // Launch failed — invalidate and fall through to re-capture or slot-by-slot
    sd_printf("NativeDynamicShapePlan: graph replay failed for segment [%d-%d], "
              "falling back to slot-by-slot\n", seg.startSlot, seg.endSlot);
    seg.cachedGraph.reset();
  }

  // ── WARM-UP: first execution populates slot cache (no capture) ──
  if (seg.executionCount == 0) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── CAPTURE: slot cache is warm, attempt to capture CUDA graph ──
  // Shape key changed → invalidate old graph
  if (seg.cachedGraph && seg.cachedShapeKey != segShapeKey) {
    seg.cachedGraph.reset();
  }

  int segSize = seg.endSlot - seg.startSlot + 1;
  // Don't bother capturing tiny segments (overhead > benefit)
  if (segSize < 10) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Use CudaGraphScheduler for capture management
  auto& scheduler = cuda::CudaGraphScheduler::getInstance();

  if (!scheduler.deviceSupportsGraphs(0)) {
    // Device doesn't support graphs — permanent fallback
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  auto handle = std::make_shared<cuda::CudaGraphHandle>();

  // Begin capture in RELAXED mode (allows some host ops during capture)
  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    sd_printf("NativeDynamicShapePlan: graph capture begin failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // tl_graphExecutionActive is set by the segment dispatch in execute().
  // DataBuffer::syncToPrimary checks it and skips D2H transfers during capture,
  // preventing illegal stream dependencies between the capture stream and stream 0.

  // Execute all ops in the segment (recorded into graph, not actually executed)
  // Wrapped in try-catch to ensure capture is properly aborted on exception
  bool captureOk = true;
  try {
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
      auto status = executeSlot(stepIdx, externalArrays, numExt, stream);
      if (status != Status::OK) {
        sd_printf("NativeDynamicShapePlan: op execution during capture failed at slot %d\n", stepIdx);
        captureOk = false;
        break;
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
  } catch (...) {
    sd_printf("NativeDynamicShapePlan: unknown exception during graph capture\n", "");
    captureOk = false;
  }

  if (!captureOk) {
    // Abort capture — stream is in an inconsistent state
    // endCapture will clean up by calling cudaStreamEndCapture
    handle->endCapture(cudaStr);

    // Clear any sticky CUDA errors left from the failed capture
    cudaGetLastError();

    // Synchronize the capture stream to ensure it returns to a clean state
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();  // Clear any sync error too
    }

    seg.captureFailed = true;
    sd_printf("NativeDynamicShapePlan: graph capture aborted for segment [%d-%d], stream state recovered\n",
              seg.startSlot, seg.endSlot);
    // Re-execute normally (capture didn't actually execute anything)
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // End capture and instantiate
  if (!handle->endCapture(cudaStr)) {
    sd_printf("NativeDynamicShapePlan: graph capture end failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    // Clear sticky CUDA errors and reset stream
    cudaGetLastError();
    cudaStreamSynchronize(cudaStr);
    cudaGetLastError();
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  if (!handle->instantiate()) {
    sd_printf("NativeDynamicShapePlan: graph instantiate failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Launch the captured graph (actual execution)
  if (!handle->launchAsync(cudaStr)) {
    sd_printf("NativeDynamicShapePlan: graph launch failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Cache the graph for future replays
  seg.cachedGraph = handle;
  seg.cachedShapeKey = segShapeKey;
  seg.executionCount++;
  totalGraphReplays_++;

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

  // ── Step 1: Gather inputs ────────────────────────────────────────────────
  std::vector<NDArray*> inputs(slot.numInputs);
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
  LongType shapeKey = computeShapeKey(slot, inputs.data(), slot.numInputs);
  bool cacheHit = slot.shapeCacheValid && (slot.cachedShapeKey == shapeKey);

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

    auto shapeList = slot.op->calculateOutputShape(&inputShapes, ctx);
    if (shapeList == nullptr || shapeList->size() == 0) {
      sd_printf("NativeDynamicShapePlan: shape inference failed for slot %d (%s)\n",
                stepIdx, slot.opName.c_str());
      return Status::KERNEL_FAILURE;
    }

    outputShapes.resize(shapeList->size());
    for (int i = 0; i < static_cast<int>(shapeList->size()); i++) {
      // Cache via ConstantShapeHelper for persistent shape pointers
      auto cached = ConstantShapeHelper::getInstance().createFromExisting(
          const_cast<LongType*>(shapeList->at(i)));
      outputShapes[i] = cached;
    }

    // Update cache
    slot.cachedShapeKey = shapeKey;
    slot.cachedOutputShapes = outputShapes;
    slot.shapeCacheValid = true;

    delete shapeList;
  }

  // ── Step 3: Allocate/reuse outputs ───────────────────────────────────────
  int numActualOutputs = std::min(slot.numOutputs, static_cast<int>(outputShapes.size()));
  std::vector<NDArray*> outputs(numActualOutputs);

  for (int i = 0; i < numActualOutputs; i++) {
    int slotIdx = slot.outputSlotIndices[i];
    if (slotIdx < 0) {
      // Untracked output — allocate temporary
      outputs[i] = new NDArray(const_cast<LongType*>(outputShapes[i]), true);
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
        // Always nullify reused buffers: some ops marked as "fully writing" may not
        // fully overwrite the output (e.g., reduction edge cases, broadcasting patterns),
        // causing stale data accumulation across execute() calls.
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
    auto* out = new NDArray(order, shape, dt);
    if (slot.needsZeroedOutput) {
      out->nullify();
    }

    outputs[i] = out;
    outputSlots_[slotIdx] = out;
    slotArrayCache_[slotIdx] = out;
  }

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
  // Hash the shapes of all external inputs referenced by slots in this segment.
  // This determines whether the cached CUDA graph is still valid.
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  // Mix segment identity
  mix(seg.startSlot);
  mix(seg.endSlot);

  // Mix shapes of external inputs used by this segment
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
      }
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
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].cachedShapeKey = 0;
    slots_[i].cachedOutputShapes.clear();
    slots_[i].shapeCacheValid = false;
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
    // Each new segment's capturability depends only on whether the first op
    // in the segment is data-dependent — device boundaries start fresh segments
    // that are capturable if their ops are not data-dependent.
    bool capturabilityChanged = (thisDataDependent == current.isCapturable);
    if (capturabilityChanged || deviceChange) {
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

  // First execution: must run slot-by-slot to populate outputSlots_ for the backend
  // The backend needs outputSlots_ to have valid shape info for tensor wiring.
  if (seg.executionCount == 0) {
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

// ─── fromFlatGraph (delegates to NativePlanCompiler) ─────────────────────────

NativeDynamicShapePlan* NativeDynamicShapePlan::fromFlatGraph(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {
  return NativePlanCompiler::compile(graph, variables, requestedOutputs);
}

}  // namespace graph
}  // namespace sd
