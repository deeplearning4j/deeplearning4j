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

/**
 * NativeDynamicShapePlan — Slot Execution
 *
 * Contains per-op slot execution (executeSlot), shape key computation
 * (computeShapeKey), and frozen constant detection (detectFrozenConstants).
 * All methods are platform-neutral (no CUDA dependencies).
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <graph/FusionPass.h>
#include <ops/OpTraitTable.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/OpRegistrator.h>

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif

// ── DSP_SLOT_WRITE: guarded slot write macro ────────────────────────────────
// Wraps all `outputSlots_[si] = value` assignments. In POINTERS_STABLE+ phase
// with executeCount > 2, logs a DSP_DIAG FALLBACK warning if the NEW value has
// a different DataBuffer than the OLD value. This detects post-freeze buffer
// replacement that would corrupt captured CUDA graphs.
// Diagnostic only — logs and continues (not a hard error).
// DSP_SLOT_WRITE: ALL output slot writes go through writeOutputSlot().
// Phase validation, stale write detection, ownership tracking, and diagnostics
// are centralized in one method on NativeDynamicShapePlan. No scattered guards.
#define DSP_SLOT_WRITE(si, value, tag) \
  writeOutputSlot((si), (value), (tag))

// DSP_NEW_ARRAY: plain allocation. Registration as plan-owned happens in
// DSP_SLOT_WRITE when the array is assigned to an output slot.
#define DSP_NEW_ARRAY(...) new NDArray(__VA_ARGS__)

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_set>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

// Verify helpers now in DspVerifyUtils.h (dspLogSlotOutput, dspDumpSlotValues, etc.)

/**
 * Backfill cachedOutputShapes from the actual output arrays in outputSlots_.
 * Identity ops, view ops, and frozen constant paths may skip shape inference,
 * leaving cachedOutputShapes empty. Cross-segment Triton compilation needs
 * these shapes, so we backfill from the live output arrays after execution.
 */
static void backfillCachedOutputShapes(NativeSlot& slot, NDArray** outputSlots, int totalOutputSlots) {
  if (!slot.cachedOutputShapes.empty()) return;  // already populated
  if (slot.numOutputs <= 0) return;

  for (int o = 0; o < slot.numOutputs; o++) {
    int si = slot.outputSlotIndices[o];
    if (si >= 0 && si < totalOutputSlots && outputSlots[si] != nullptr) {
      auto* outArr = outputSlots[si];
      if (outArr->shapeInfo() != nullptr) {
        auto cached = ConstantShapeHelper::getInstance().createFromExisting(
            const_cast<LongType*>(outArr->shapeInfo()));
        slot.cachedOutputShapes.push_back(cached);
      }
    }
  }
  if (!slot.cachedOutputShapes.empty() && slot.state_ < NativeSlot::SlotState::SHAPE_CACHED) {
    slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;
  }
}

namespace {

static int findProducingStepForOutputSlot(const NativeSlot* slots, int numSlots, int outputSlotIdx) {
  if (slots == nullptr || outputSlotIdx < 0) return -1;
  for (int s = 0; s < numSlots; s++) {
    const auto& slot = slots[s];
    for (int o = 0; o < slot.numOutputs; o++) {
      if (slot.outputSlotIndices[o] == outputSlotIdx) {
        return s;
      }
    }
  }
  return -1;
}

static NDArray* resolveInputSourceArray(int srcIdx,
                                        NDArray** outputSlots,
                                        int totalOutputSlots,
                                        NDArray** externalArrays,
                                        int numExt) {
  if (srcIdx < 0) {
    const int extIdx = -(srcIdx + 1);
    return (extIdx >= 0 && extIdx < numExt) ? externalArrays[extIdx] : nullptr;
  }
  return (srcIdx >= 0 && srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
}

static bool isSmallIntegralControlArray(const NDArray* arr) {
  if (arr == nullptr) return false;
  const auto dt = arr->dataType();
  if (dt != INT32 && dt != INT64 && dt != BOOL) return false;
  const auto len = arr->lengthOf();
  return len > 0 && len <= 32;
}

static SD_INLINE bool shouldPreserveWarmupOutputsDuringCapture() {
  // Capture records GPU work against the warmup-established state; it must not
  // mutate the current step's live output buffers. If we nullify reused outputs
  // during capture, the zeroes stay resident because the recorded kernels/H2D
  // nodes do not execute until replay on a later step.
  return tl_graphExecutionActive;
}

static void traceSmallControlSlotIO(const char* stage,
                                    int stepIdx,
                                    const NativeSlot& slot,
                                    const NDArray* input0,
                                    const NDArray* output0,
                                    int executeCount,
                                    PlanPhase planPhase) {
  if (!DSP_DIAG_ENABLED(SHAPE)) return;
  if (!slot.outputShapeDependsOnInputValues) return;
  if (!isSmallIntegralControlArray(input0) && !isSmallIntegralControlArray(output0)) return;

  const bool sharesBuffer =
      input0 != nullptr && output0 != nullptr &&
      input0->dataBuffer() != nullptr &&
      input0->dataBuffer() == output0->dataBuffer();

  DSP_DIAG(SHAPE,
           "CONTROL_SLOT_IO: stage=%s slot=%d (%s) state=%d exec=%d phase=%d "
           "input0=%s inputShape=%s output0=%s outputShape=%s sharesBuffer=%d",
           stage, stepIdx, slot.opName.c_str(), static_cast<int>(slot.state_),
           executeCount, static_cast<int>(planPhase),
           input0 != nullptr ? dspDumpHostDeviceValues(const_cast<NDArray*>(input0), 16).c_str() : "null",
           input0 != nullptr ? dspShapeStr(const_cast<NDArray*>(input0)).c_str() : "null",
           output0 != nullptr ? dspDumpHostDeviceValues(const_cast<NDArray*>(output0), 16).c_str() : "null",
           output0 != nullptr ? dspShapeStr(const_cast<NDArray*>(output0)).c_str() : "null",
           sharesBuffer ? 1 : 0);
}

static void traceSmallControlSlotTensors(const char* stage,
                                         int stepIdx,
                                         const NativeSlot& slot,
                                         NDArray** inputs,
                                         int numInputs,
                                         NDArray** outputs,
                                         int numOutputs,
                                         int executeCount,
                                         PlanPhase planPhase) {
  if (!DSP_DIAG_ENABLED(SHAPE)) return;
  if (!slot.outputShapeDependsOnInputValues) return;

  bool anySmallTensor = false;
  std::string inputStr;
  for (int i = 0; i < numInputs; i++) {
    NDArray* arr = (inputs != nullptr && i < numInputs) ? inputs[i] : nullptr;
    if (!isSmallIntegralControlArray(arr)) continue;
    anySmallTensor = true;
    if (!inputStr.empty()) inputStr += " | ";
    inputStr += "input[";
    inputStr += std::to_string(i);
    inputStr += "]=";
    inputStr += dspDumpHostDeviceValues(arr, 16);
    inputStr += " shape=";
    inputStr += dspShapeStr(arr);
  }

  std::string outputStr;
  for (int i = 0; i < numOutputs; i++) {
    NDArray* arr = (outputs != nullptr && i < numOutputs) ? outputs[i] : nullptr;
    if (!isSmallIntegralControlArray(arr)) continue;
    anySmallTensor = true;
    if (!outputStr.empty()) outputStr += " | ";
    outputStr += "output[";
    outputStr += std::to_string(i);
    outputStr += "]=";
    outputStr += dspDumpHostDeviceValues(arr, 16);
    outputStr += " shape=";
    outputStr += dspShapeStr(arr);
  }

  if (!anySmallTensor) return;

  DSP_DIAG(SHAPE,
           "CONTROL_SLOT_TENSORS: stage=%s slot=%d (%s) state=%d exec=%d phase=%d inputs=[%s] outputs=[%s]",
           stage, stepIdx, slot.opName.c_str(), static_cast<int>(slot.state_),
           executeCount, static_cast<int>(planPhase),
           inputStr.empty() ? "none" : inputStr.c_str(),
           outputStr.empty() ? "none" : outputStr.c_str());
}

static void reconcileExecutedOutputActuality(const char* stage,
                                             int stepIdx,
                                             const NativeSlot& slot,
                                             NDArray* output) {
  if (output == nullptr) return;
  auto* db = output->dataBuffer();
  if (db == nullptr || db->isClosed()) return;

#ifdef SD_CUDA
  const bool primaryActual = db->isPrimaryActual();
  const bool specialActual = db->isSpecialActual();
  const bool needsDeviceVisibleControl =
      slot.isDataDependent || slot.outputShapeDependsOnInputValues ||
      isSmallIntegralControlArray(output);

  if (primaryActual && !specialActual) {
    if (needsDeviceVisibleControl) {
      output->syncToDevice();
      DSP_DIAG(SHAPE,
               "CONTROL_OUTPUT_SYNC: stage=%s slot=%d (%s) "
               "synced host-current output to device after native execution",
               stage, stepIdx, slot.opName.c_str());
    }
    return;
  }

  if (!primaryActual && !specialActual) {
    db->writeSpecial();
    DSP_DIAG(EXECUTE,
             "OUTPUT_ACTUALITY_REPAIR: stage=%s slot=%d (%s) "
             "marked device current after untracked native write",
             stage, stepIdx, slot.opName.c_str());
  }
#else
  (void)stage;
  (void)stepIdx;
  (void)slot;
#endif
}

static bool slotHasOnlyPlanInternalControlInputs(const NativeSlot& slot,
                                                 NDArray** outputSlots,
                                                 int totalOutputSlots,
                                                 NDArray** externalArrays,
                                                 int numExt) {
  bool sawControlInput = false;
  for (int i = 0; i < slot.numInputs; i++) {
    NDArray* arr = resolveInputSourceArray(slot.inputSourceIndices[i], outputSlots,
                                           totalOutputSlots, externalArrays, numExt);
    if (!isSmallIntegralControlArray(arr)) continue;
    sawControlInput = true;
    if (slot.inputSourceIndices[i] < 0) {
      return false;
    }
  }
  return sawControlInput;
}

static bool hasPlanInternalValueShapeAncestor(int stepIdx,
                                              const NativeSlot* slots,
                                              int numSlots,
                                              NDArray** outputSlots,
                                              int totalOutputSlots,
                                              NDArray** externalArrays,
                                              int numExt,
                                              std::unordered_set<int>& visitedSteps) {
  if (slots == nullptr || stepIdx < 0 || stepIdx >= numSlots) return false;
  if (!visitedSteps.insert(stepIdx).second) return false;

  const auto& slot = slots[stepIdx];
  if (slot.outputShapeDependsOnInputValues &&
      slotHasOnlyPlanInternalControlInputs(slot, outputSlots, totalOutputSlots,
                                           externalArrays, numExt)) {
    return true;
  }

  for (int i = 0; i < slot.numInputs; i++) {
    const int srcIdx = slot.inputSourceIndices[i];
    if (srcIdx < 0) continue;
    const int producerStep = findProducingStepForOutputSlot(slots, numSlots, srcIdx);
    if (producerStep >= 0 &&
        hasPlanInternalValueShapeAncestor(producerStep, slots, numSlots,
                                          outputSlots, totalOutputSlots,
                                          externalArrays, numExt, visitedSteps)) {
      return true;
    }
  }

  return false;
}

static bool allowFrozenShapeRestabilization(int stepIdx,
                                            const NativeSlot& slot,
                                            PlanPhase planPhase,
                                            const NativeSlot* slots,
                                            int numSlots,
                                            NDArray** outputSlots,
                                            int totalOutputSlots,
                                            NDArray** externalArrays,
                                            int numExt) {
  if (planPhase >= PlanPhase::POINTERS_STABLE) return false;

  if (slot.outputShapeDependsOnInputValues &&
      slotHasOnlyPlanInternalControlInputs(slot, outputSlots, totalOutputSlots,
                                           externalArrays, numExt)) {
    return true;
  }

  std::unordered_set<int> visitedSteps;
  for (int i = 0; i < slot.numInputs; i++) {
    const int srcIdx = slot.inputSourceIndices[i];
    if (srcIdx < 0) continue;
    const int producerStep = findProducingStepForOutputSlot(slots, numSlots, srcIdx);
    if (producerStep >= 0 &&
        hasPlanInternalValueShapeAncestor(producerStep, slots, numSlots,
                                          outputSlots, totalOutputSlots,
                                          externalArrays, numExt, visitedSteps)) {
      return true;
    }
  }

  return false;
}

static void appendUpstreamControlTrace(std::string& out,
                                       int srcIdx,
                                       int depth,
                                       const NativeSlot* slots,
                                       int numSlots,
                                       NDArray** outputSlots,
                                       int totalOutputSlots,
                                       NDArray** externalArrays,
                                       int numExt,
                                       const std::vector<std::string>& externalInputNames,
                                       std::vector<int>& seenSlotSteps) {
  if (depth < 0) return;

  if (srcIdx < 0) {
    const int extIdx = -(srcIdx + 1);
    out += "ext[";
    out += std::to_string(extIdx);
    out += "]";
    if (extIdx >= 0 && extIdx < numExt) {
      if (extIdx < static_cast<int>(externalInputNames.size()) &&
          !externalInputNames[extIdx].empty()) {
        out += "=\"";
        out += externalInputNames[extIdx];
        out += "\"";
      }
      NDArray* ext = externalArrays != nullptr ? externalArrays[extIdx] : nullptr;
      out += " ";
      out += ext != nullptr ? dspDumpHostDeviceValues(ext, 8) : "null";
      if (ext != nullptr) {
        out += " shape=";
        out += dspShapeStr(ext);
      }
    } else {
      out += " OUT_OF_RANGE";
    }
    return;
  }

  const int producerStep = findProducingStepForOutputSlot(slots, numSlots, srcIdx);
  out += "slot[";
  out += std::to_string(srcIdx);
  out += "]";
  if (producerStep >= 0) {
    out += " step=";
    out += std::to_string(producerStep);
    out += " op=";
    out += slots[producerStep].opName;
  }

  NDArray* arr = (outputSlots != nullptr && srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
  out += " ";
  out += arr != nullptr ? dspDumpHostDeviceValues(arr, 8) : "null";
  if (arr != nullptr) {
    out += " shape=";
    out += dspShapeStr(arr);
  }

  if (producerStep < 0 || depth == 0) return;
  if (std::find(seenSlotSteps.begin(), seenSlotSteps.end(), producerStep) != seenSlotSteps.end()) {
    out += " <cycle>";
    return;
  }

  seenSlotSteps.push_back(producerStep);
  const auto& producer = slots[producerStep];
  if (producer.numInputs > 0) {
    out += " <= {";
    for (int i = 0; i < producer.numInputs; i++) {
      if (i > 0) out += "; ";
      appendUpstreamControlTrace(out, producer.inputSourceIndices[i], depth - 1,
                                 slots, numSlots, outputSlots, totalOutputSlots,
                                 externalArrays, numExt, externalInputNames,
                                 seenSlotSteps);
    }
    out += "}";
  }
  seenSlotSteps.pop_back();
}

/**
 * Build shape info for a permute view: takes the input's shape info and permutes
 * both dimensions AND strides according to the permutation vector.
 * Unlike calculateOutputShape (which sets contiguous strides), this preserves
 * the input's actual stride pattern so the view reads data correctly.
 *
 * Returns a ConstantShapeHelper-managed pointer (not owned by caller).
 */
static const LongType* buildPermutedViewShapeInfo(const NDArray* input, const NativeSlot& slot) {
  int rank = shape::rank(input->shapeInfo());
  if (rank <= 0) return nullptr;

  const LongType* inShape = shape::shapeOf(const_cast<LongType*>(input->shapeInfo()));
  const LongType* inStrides = shape::stride(const_cast<LongType*>(input->shapeInfo()));

  // Build effective permutation, adapting if rank > numIArgs
  // (e.g., expand_dims added leading size-1 dims: rank-5 input with rank-4 permutation)
  std::vector<int> permVec;
  if (slot.numIArgs >= rank) {
    for (int i = 0; i < rank; i++) permVec.push_back(static_cast<int>(slot.iArgs[i]));
  } else if (slot.numIArgs > 0) {
    int extraDims = rank - slot.numIArgs;
    int leadingOnes = 0;
    for (int i = 0; i < rank && leadingOnes < extraDims; i++) {
      if (inShape[i] == 1) leadingOnes++;
      else break;
    }
    if (leadingOnes >= extraDims) {
      for (int i = 0; i < extraDims; i++) permVec.push_back(i);
      for (int i = 0; i < slot.numIArgs; i++) permVec.push_back(static_cast<int>(slot.iArgs[i]) + extraDims);
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }

  // Build permuted shape and strides from input
  std::vector<LongType> permShape(rank);
  std::vector<LongType> permStrides(rank);

  for (int i = 0; i < rank; i++) {
    int srcDim = permVec[i];
    if (srcDim < 0 || srcDim >= rank) return nullptr;  // invalid permutation
    permShape[i] = inShape[srcDim];
    permStrides[i] = inStrides[srcDim];
  }

  // Build shape info buffer: [rank, shape..., strides..., 0, ews, order+flags]
  auto shapeInfoLen = shape::shapeInfoLength(rank);
  // Allocate — createFromExisting takes ownership
  LongType* shapeInfoBuf = new LongType[shapeInfoLen];
  shapeInfoBuf[0] = rank;
  for (int i = 0; i < rank; i++) {
    shapeInfoBuf[1 + i] = permShape[i];
    shapeInfoBuf[1 + rank + i] = permStrides[i];
  }
  // extras (contains data type) — copy from input
  shapeInfoBuf[2 * rank + 1] = input->shapeInfo()[2 * rank + 1];
  // ews = 0 (view, not contiguous)
  shapeInfoBuf[2 * rank + 2] = 0;
  // Copy order from input
  shapeInfoBuf[2 * rank + 3] = input->shapeInfo()[2 * rank + 3];

  return ConstantShapeHelper::getInstance().createFromExisting(shapeInfoBuf);
}

std::string normalizeOpName_slotexec(const std::string& opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}
/**
 * Compute the byte-element offset for a strided_slice view into the input buffer.
 * iArgs layout: [beginMask, ellipsisMask, endMask, newAxisMask, shrinkAxisMask]
 * Inputs beyond input0 are begin(1), end(2), strides(3) tensors.
 *
 * Returns the element offset, or -1 if a view cannot be created (e.g., non-unit
 * strides that require non-contiguous access, or newAxisMask is set).
 */
static LongType computeStridedSliceViewOffset(const NativeSlot& slot,
                                               NDArray* input0,
                                               NDArray** inputs,
                                               int numInputs) {
  if (slot.numIArgs < 5 || numInputs < 4) return -1;

  int beginMask = static_cast<int>(slot.iArgs[0]);
  // int ellipsisMask = static_cast<int>(slot.iArgs[1]); // not needed for offset
  // int endMask = static_cast<int>(slot.iArgs[2]);       // not needed for offset
  int newAxisMask = static_cast<int>(slot.iArgs[3]);
  int shrinkAxisMask = static_cast<int>(slot.iArgs[4]);

  // Can't create a simple offset view when newAxisMask adds dimensions
  if (newAxisMask != 0) return -1;

  NDArray* beginArr = inputs[1];
  NDArray* stridesArr = (numInputs > 3) ? inputs[3] : nullptr;
  if (beginArr == nullptr) return -1;

  int inputRank = input0->rankOf();
  int sliceDims = static_cast<int>(beginArr->lengthOf());

  // Check strides: only unit strides allow a contiguous view
  if (stridesArr != nullptr) {
    for (int i = 0; i < stridesArr->lengthOf(); i++) {
      if (stridesArr->e<LongType>(i) != 1) return -1;
    }
  }

  LongType offset = 0;
  for (int i = 0; i < inputRank; i++) {
    LongType dimSize = input0->sizeAt(i);
    LongType effectiveBegin = 0;

    if (i < sliceDims) {
      if ((beginMask & (1 << i)) != 0) {
        effectiveBegin = 0;
      } else {
        effectiveBegin = beginArr->e<LongType>(i);
        if (effectiveBegin < 0) effectiveBegin += dimSize;
        if (effectiveBegin < 0) effectiveBegin = 0;
        if (effectiveBegin > dimSize) effectiveBegin = dimSize;
      }
      // For shrink axis, begin is the selected index
      if ((shrinkAxisMask & (1 << i)) != 0) {
        // effectiveBegin is already the index
      }
    }

    offset += effectiveBegin * input0->strideAt(i);
  }

  return offset;
}

/**
 * Result codes for tryCreateViewForSlot().
 */
enum ViewCreateResult {
  VIEW_NOT_POSSIBLE = 0,      // Input not contiguous or length mismatch — fall through to normal alloc
  VIEW_CREATED = 1,           // View successfully created and returned
  VIEW_STRIDED_SLICE_FAIL = -1, // strided_slice offset computation failed (non-unit strides/newAxisMask)
  VIEW_STALE_EMPTY_SHAPE = 2  // outLen == 0 but inLen > 0 — stale shape, needs re-inference
};

/**
 * Try to create a zero-copy view for a view-capable op (reshape, expand_dims,
 * squeeze, strided_slice, permute).
 *
 * On success (VIEW_CREATED), sets *outView to the newly created NDArray view
 * and *outViewOffset to the byte-element offset used.
 *
 * On failure, *outView is nullptr and the result code indicates why:
 *   VIEW_NOT_POSSIBLE       — input not C-contiguous, or output length > input length
 *   VIEW_STRIDED_SLICE_FAIL — strided_slice offset computation failed
 *   VIEW_STALE_EMPTY_SHAPE  — cached output shape is empty but input is non-empty
 *
 * The caller is responsible for installing the view into outputSlots_,
 * slotArrayCache_, slotIsViewProducer_, and context — this helper only
 * creates the NDArray wrapper.
 *
 * @param slot           The NativeSlot for this op
 * @param input0         The primary input array (must not be null)
 * @param outShapeInfo   The shape info to use for the output view
 * @param allInputs      All resolved inputs (needed for strided_slice offset)
 * @param numInputs      Number of entries in allInputs
 * @param outView        [out] The created view, or nullptr
 * @param outViewOffset  [out] The element offset into input0's buffer
 * @return ViewCreateResult code
 */
static ViewCreateResult tryCreateViewForSlot(
    int stepIdx,
    const NativeSlot& slot,
    NDArray* input0,
    const LongType* outShapeInfo,
    NDArray** allInputs,
    int numInputs,
    NDArray** outView,
    LongType* outViewOffset) {

  *outView = nullptr;
  *outViewOffset = 0;

  const LongType outLen = shape::length(outShapeInfo);
  const LongType inLen = input0 != nullptr ? input0->lengthOf() : 0;
  const bool traceSmallControl =
      input0 != nullptr && isSmallIntegralControlArray(input0) && outLen > 0 && outLen <= 32;

  // Check C-contiguity of input
  if (input0->dataBuffer() == nullptr ||
      input0->ordering() != 'c' ||
      !shape::strideDescendingCAscendingF(const_cast<LongType*>(input0->shapeInfo()))) {
    if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
      DSP_DIAG(SHAPE,
               "VIEW_TRACE_FAIL: slot %d (%s) input not contiguous enough for zero-copy "
               "view: order=%c strideDescending=%d inShape=%s outShape=%s inLen=%lld outLen=%lld",
               stepIdx, slot.opName.c_str(), input0->ordering(),
               shape::strideDescendingCAscendingF(const_cast<LongType*>(input0->shapeInfo())) ? 1 : 0,
               ShapeUtils::shapeAsString(input0).c_str(),
               ShapeUtils::shapeAsString(outShapeInfo).c_str(),
               static_cast<long long>(inLen), static_cast<long long>(outLen));
    }
    return VIEW_NOT_POSSIBLE;
  }

  // For permute: use permuted strides from input, not contiguous strides
  const LongType* effectiveShapeInfo = outShapeInfo;
  bool isPermute = (normalizeOpName_slotexec(slot.opName) == "permute");
  if (isPermute) {
    const LongType* permSI = buildPermutedViewShapeInfo(input0, slot);
    if (permSI != nullptr) effectiveShapeInfo = permSI;
  }

  // Compute view offset for strided_slice
  LongType viewOffset = 0;
  if (slot.opName == "strided_slice") {
    viewOffset = computeStridedSliceViewOffset(slot, input0, allInputs, numInputs);
    if (viewOffset < 0) {
      return VIEW_STRIDED_SLICE_FAIL;
    }
  }
  *outViewOffset = viewOffset;

  LongType absoluteOffset = input0->offset() + viewOffset;
  LongType sourceBufferElems =
      input0->dataBuffer() != nullptr ? input0->dataBuffer()->getNumElements() : 0;
  bool allowSubsetView =
      slot.op != nullptr && slot.op->getOpDescriptor() != nullptr &&
      slot.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SLICE);
  bool elementCountCompatible =
      allowSubsetView ? (outLen > 0 && outLen <= inLen) : (outLen > 0 && outLen == inLen);
  bool fitsBackingBuffer =
      absoluteOffset >= 0 && absoluteOffset <= sourceBufferElems &&
      outLen <= (sourceBufferElems - absoluteOffset);

  if (elementCountCompatible && fitsBackingBuffer) {
    *outView = DSP_NEW_ARRAY(input0->dataBuffer(),
                           const_cast<LongType*>(effectiveShapeInfo),
                           LaunchContext::defaultContext(),
                           absoluteOffset);
    *outViewOffset = absoluteOffset;
    if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
      DSP_DIAG(SHAPE,
               "VIEW_TRACE_OK: slot %d (%s) zero-copy view created "
               "inShape=%s outShape=%s inLen=%lld outLen=%lld offset=%lld",
               stepIdx, slot.opName.c_str(),
               ShapeUtils::shapeAsString(input0).c_str(),
               ShapeUtils::shapeAsString(effectiveShapeInfo).c_str(),
               static_cast<long long>(inLen), static_cast<long long>(outLen),
               static_cast<long long>(absoluteOffset));
    }
    return VIEW_CREATED;
  } else if (outLen == 0 && inLen > 0) {
    if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
      DSP_DIAG(SHAPE,
               "VIEW_TRACE_FAIL: slot %d (%s) stale empty cached shape "
               "inShape=%s outShape=%s inLen=%lld outLen=%lld",
               stepIdx, slot.opName.c_str(),
               ShapeUtils::shapeAsString(input0).c_str(),
               ShapeUtils::shapeAsString(effectiveShapeInfo).c_str(),
               static_cast<long long>(inLen), static_cast<long long>(outLen));
    }
    return VIEW_STALE_EMPTY_SHAPE;
  }

  if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
    DSP_DIAG(SHAPE,
             "VIEW_TRACE_FAIL: slot %d (%s) element/buffer mismatch "
             "inShape=%s outShape=%s inLen=%lld outLen=%lld allowSubset=%d absoluteOffset=%lld "
             "sourceBufferElems=%lld fitsBackingBuffer=%d",
             stepIdx, slot.opName.c_str(),
             ShapeUtils::shapeAsString(input0).c_str(),
             ShapeUtils::shapeAsString(effectiveShapeInfo).c_str(),
             static_cast<long long>(inLen), static_cast<long long>(outLen),
             allowSubsetView ? 1 : 0,
             static_cast<long long>(absoluteOffset),
             static_cast<long long>(sourceBufferElems),
             fitsBackingBuffer ? 1 : 0);
  }

  return VIEW_NOT_POSSIBLE;
}

}  // namespace

// ─── Frozen constant detection ──────────────────────────────────────────────
// After the warmup execution (executeCount_ just went from 0 to 1), identify
// slots whose output never changes between decode steps. These slots are
// skipped entirely during subsequent executions (including graph capture),
// removing their kernels, memsets, and memcpys from the captured graph.

void NativeDynamicShapePlan::detectFrozenConstants() {
  if (!shapesFrozen_ || executeCount_ != 1 || frozenConstantDetectionDone_) return;
  frozenConstantDetectionDone_ = true;

  // Ops whose output depends ONLY on input shapes, not input values.
  // When shapes are frozen, these produce identical output every step.
  // Classification now comes from OpDescriptor traits (OP_TRAIT_SHAPE_ONLY_OUTPUT).

  std::vector<bool> dependsOnExternal(totalOutputSlots_, false);
  std::vector<bool> isValueIndependentSlot(numSlots_, false);

  // Propagate external dependency through the graph (topological order).
  // Value-independent ops do NOT propagate dependency — their outputs
  // are constant when shapes are frozen.
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];

    // Check if this op is value-independent via trait
    bool isShapeOnly = (sl.op && sl.op->getOpDescriptor() &&
                        sl.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT));
    if (isShapeOnly) {
      isValueIndependentSlot[s] = true;
      continue;
    }

    bool anyInputDependsOnExternal = false;
    for (int i = 0; i < sl.numInputs; i++) {
      int srcIdx = sl.inputSourceIndices[i];
      if (srcIdx < 0) {
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
      sl.state_ = NativeSlot::SlotState::FROZEN_CONSTANT;
      frozenConstCount++;
      if (isValueIndependentSlot[s]) valueIndepCount++;
    }
  }
  // Collect frozen output slot indices for quick lookup
  std::unordered_set<int> frozenOutputSlots;
  for (int s = 0; s < numSlots_; s++) {
    if (slots_[s].frozenConstantSlot()) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        frozenOutputSlots.insert(slots_[s].outputSlotIndices[o]);
      }
    }
  }

  // Un-freeze any frozen constant whose output DataBuffer is shared (via view aliasing)
  // with a non-frozen slot's output. View-producing ops (reshape, expand_dims, etc.)
  // create outputs that share their input's DataBuffer. If the frozen constant's output
  // is a view of a non-frozen slot's buffer, the non-frozen slot can overwrite the buffer
  // on subsequent steps, corrupting the frozen constant's data.
  // Example: shape_of → concat → reshape(view) chain where reshape shares concat's buffer,
  // and a downstream non-frozen op writes to the same buffer region.
  int viewAliasUnfrozen = 0;
  {
    // Build a set of DataBuffer pointers owned by non-frozen output slots
    std::unordered_set<const void*> nonFrozenBuffers;
    for (int si = 0; si < totalOutputSlots_; si++) {
      if (!frozenOutputSlots.count(si) && outputSlots_[si] != nullptr
          && outputSlots_[si]->dataBuffer() != nullptr) {
        nonFrozenBuffers.insert(
            static_cast<const void*>(outputSlots_[si]->dataBuffer()));
      }
    }
    // Check each frozen constant's outputs for buffer aliasing
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      if (!sl.frozenConstantSlot()) continue;
      bool aliased = false;
      for (int o = 0; o < sl.numOutputs; o++) {
        int si = sl.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr
            && outputSlots_[si]->dataBuffer() != nullptr) {
          const void* bufPtr = static_cast<const void*>(
              outputSlots_[si]->dataBuffer());
          if (nonFrozenBuffers.count(bufPtr)) {
            aliased = true;
            break;
          }
        }
      }
      if (aliased) {
        sl.state_ = NativeSlot::SlotState::SHAPE_CACHED;
        // Remove from frozenOutputSlots
        for (int o = 0; o < sl.numOutputs; o++) {
          frozenOutputSlots.erase(sl.outputSlotIndices[o]);
        }
        viewAliasUnfrozen++;
        frozenConstCount--;
      }
    }
  }

  // Disable in-place fusion for any op that would overwrite a frozen output buffer.
  // In-place fusion writes the op's output directly into its input buffer.
  // If that input comes from a frozen constant slot, the frozen value gets corrupted.
  // Note: in-place fusion disabling for frozen slots is now done earlier in
  // rebuildSegmentsForFrozenShapes() (before the warmup), so the warmup doesn't
  // corrupt cached frozen values. The code here is a safety net for any case
  // where rebuildSegments wasn't called before frozen detection.
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

  DSP_DIAG(SHAPE, "frozen constant detection: %d/%d slots are frozen constants (%d value-independent, %d in-place disabled, %d view-alias unfrozen)",
            frozenConstCount, numSlots_, valueIndepCount, disabledInPlace, viewAliasUnfrozen);

  // Log which output slots are frozen constant vs external-dependent for debugging
  // NaN issues during frozen steady-state execution.
  if (DSP_DIAG_ENABLED(SHAPE)) {
    int extDepCount = 0;
    for (int si = 0; si < totalOutputSlots_; si++) {
      if (dependsOnExternal[si]) extDepCount++;
    }
    fprintf(stdout, "[DSP_DIAG] [FROZEN_CONST] %d/%d output slots depend on external input\n",
            extDepCount, totalOutputSlots_);
    // Log specific slots around the known NaN chain for debugging
    for (int checkSlot : {299, 1214, 1215}) {
      if (checkSlot < totalOutputSlots_) {
        bool depExt = dependsOnExternal[checkSlot];
        // Find which step produces this output slot
        int producingStep = -1;
        bool isFrozenConst = false, isShapeStatic = false;
        for (int s = 0; s < numSlots_; s++) {
          for (int o = 0; o < slots_[s].numOutputs; o++) {
            if (slots_[s].outputSlotIndices[o] == checkSlot) {
              producingStep = s;
              isFrozenConst = slots_[s].frozenConstantSlot();
              isShapeStatic = slots_[s].shapeStatic;
              goto foundStep;
            }
          }
        }
        foundStep:;
        fprintf(stdout, "[DSP_DIAG] [FROZEN_CONST] outSlot=%d dependsOnExternal=%d "
                "producingStep=%d frozenConst=%d shapeStatic=%d opName=%s\n",
                checkSlot, depExt ? 1 : 0, producingStep,
                isFrozenConst ? 1 : 0, isShapeStatic ? 1 : 0,
                producingStep >= 0 ? slots_[producingStep].opName.c_str() : "?");
      }
    }
    fflush(stdout);
  }
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

  // Mix input shapes, dtypes, and empty flag
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i] == nullptr) continue;
    const LongType* si = inputs[i]->shapeInfo();
    LongType rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
    }
    mix(static_cast<LongType>(inputs[i]->dataType()));
    // Include ARRAY_EMPTY flag — an input transitioning from empty to
    // non-empty with the same dimensions must produce a different key.
    mix(static_cast<LongType>(inputs[i]->isEmpty() ? 1 : 0));
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

// ─── Per-slot execution ─────────────────────────────────────────────────────

Status NativeDynamicShapePlan::executeSlot(
    int stepIdx, NDArray** externalArrays, int numExt, void* stream) {
  NativeSlot& slot = slots_[stepIdx];

  // DIAGNOSTIC: trace entry for controller slots
  bool traceEntry = DSP_DIAG_ENABLED(SHAPE) &&
      (stepIdx == 349 || stepIdx == 350 || stepIdx == 357 || stepIdx == 358);
  if (traceEntry) {
    DSP_DIAG_SLOT(SHAPE, stepIdx,
        "EXEC_SLOT_ENTRY: slot %d (%s) state=%d frozenContextReady=%d shapeValid=%d "
        "isViewCapable=%d isDataDep=%d outputShapeDep=%d execCount=%d shapesFrozen=%d",
        stepIdx, slot.opName.c_str(),
        (int)slot.state_,
        slot.frozenContextReady() ? 1 : 0,
        slot.shapeCacheValid() ? 1 : 0,
        slot.isViewCapableOp ? 1 : 0,
        slot.isDataDependent ? 1 : 0,
        slot.outputShapeDependsOnInputValues ? 1 : 0,
        executeCount_,
        shapesFrozen_ ? 1 : 0);
  }

  auto discardCachedSlotArray = [&](int slotIdx, NDArray* cached, const char* tag) {
    if (cached == nullptr) return;

    // In Phase 2, slotArrayCache_ aliases outputSlots_. Clearing the live slot
    // before deleting avoids leaving a dangling NDArray wrapper behind when a
    // shape transition forces inline replacement.
    if (slotArrayCache_ == outputSlots_ &&
        slotIdx >= 0 && slotIdx < totalOutputSlots_ &&
        outputSlots_[slotIdx] == cached) {
      outputSlots_[slotIdx] = nullptr;
      if (slotOwnership_ != nullptr) {
        slotOwnership_[slotIdx].reset();
      }
    }

    if (!tl_graphExecutionActive && !isSlotArrayShared(cached, slotIdx)) {
      planOwnedArrays_.erase(cached);
      delete cached;
      DSP_DIAG(MEMORY, "discardCachedSlotArray: slot=%d tag=%s deleted=%p",
               slotIdx, tag, (void*)cached);
    } else {
      DSP_DIAG(MEMORY, "discardCachedSlotArray: slot=%d tag=%s preserved=%p sharedOrCapturing=%d",
               slotIdx, tag, (void*)cached,
               (tl_graphExecutionActive || isSlotArrayShared(cached, slotIdx)) ? 1 : 0);
    }

    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      slotArrayCache_[slotIdx] = nullptr;
    }
  };

  auto validateReusableSlotArray = [&](int slotIdx, NDArray* cached,
                                       const char* tag) -> NDArray* {
    if (cached == nullptr) return nullptr;

    auto* db = cached->dataBuffer();
    bool invalid = (db == nullptr) || db->isClosed() || !db->isValid();
#ifdef SD_CUDA
    if (!invalid && db != nullptr && db->special() == nullptr && !cached->isEmpty()) {
      invalid = true;
    }
#else
    if (!invalid && db != nullptr && db->primary() == nullptr && !cached->isEmpty()) {
      invalid = true;
    }
#endif

    if (!invalid) return cached;

    DSP_DIAG_SLOT(MEMORY, stepIdx,
        "STALE_CACHED_OUTPUT: slot %d (%s) tag=%s arr=%p db=%p valid=%d closed=%d len=%lld",
        slotIdx, slot.opName.c_str(), tag, (void*)cached, (void*)db,
        db != nullptr && db->isValid() ? 1 : 0,
        db != nullptr && db->isClosed() ? 1 : 0,
        (long long)cached->lengthOf());
    discardCachedSlotArray(slotIdx, cached, tag);
    return nullptr;
  };

  // ── Fast path: identity ops ──────────────────────────────────────────────
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
          DSP_SLOT_WRITE(si, input, "identity-alias");  // expected: alias, not new allocation
        }
      }
#ifdef SD_CUDA
      if (Environment::getInstance().tritonVerifyKernels()) {
        dspLogSlotOutput(stepIdx, slot.opName.c_str(), "IDENTITY",
                            outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
      }
#endif
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
      return Status::OK;
    }
  }

  // ── Frozen constant optimization ──────────────────────────────────────────
  // Only skip when shapesFrozen_ && executeCount_ > 0, because that's when
  // outputSlots_ is populated from slotArrayCache_ (line 921-922).  When
  // shapesFrozen_=false, outputSlots_ is zeroed — skipping a slot would
  // leave a NULL entry and downstream slots would get NULL inputs.
  if (slot.frozenConstantSlot() && shapesFrozen_ && executeCount_ > 0) {
    // Verify all outputs are populated before skipping. If outputSlots_[si]
    // is null (first execution), fall through to execute the slot so
    // downstream consumers get a valid array.
    bool allOutputsPopulated = true;
    for (int o = 0; o < slot.numOutputs; o++) {
      int si = slot.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
        allOutputsPopulated = false;
        break;
      }
    }
    if (allOutputsPopulated) {
#ifdef SD_CUDA
      if (Environment::getInstance().tritonVerifyKernels()) {
        DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s [SKIPPED:frozen-const]", stepIdx, slot.opName.c_str());
      }
#endif
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
      return Status::OK;
    }
    // Fall through to execute — output slot is null, must re-execute to populate it
    DSP_DIAG(EXECUTE, "SLOT_EXEC step=%d op=%s frozen-const but output null, re-executing",
             stepIdx, slot.opName.c_str());
  }

  // ── Fused chain tail skip ─────────────────────────────────────────────────
  if (slot.isFusedChainTail) {
#ifdef SD_CUDA
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s [SKIPPED:fused-tail]", stepIdx, slot.opName.c_str());
    }
#endif
    return Status::OK;
  }

  // ── Fused chain head dispatch ─────────────────────────────────────────────
  if (slot.isFusedChainHead && slot.fusedChainLength >= 2) {
    // 1. Gather primary input (head slot's first input)
    NDArray* primaryInput = nullptr;
    int primarySrcIdx = slot.inputSourceIndices[0];
    if (primarySrcIdx >= 0) {
      primaryInput = outputSlots_[primarySrcIdx];
      // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore needed
    } else {
      int extIdx = -(primarySrcIdx + 1);
      if (extIdx < numExt) primaryInput = externalArrays[extIdx];
    }
    if (primaryInput == nullptr) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx, "NULL fused head input for slot %d (%s), primarySrcIdx=%d",
                stepIdx, slot.opName.c_str(), primarySrcIdx);
      return Status::BAD_INPUT;
    }

    bool headIsBinary = (slot.fusedChainSecondaryInputSources[0] != INT32_MIN);
    if (headIsBinary && slot.numInputs == 2) {
      int secSrc = slot.fusedChainSecondaryInputSources[0];
      for (int k = 0; k < slot.numInputs; k++) {
        if (slot.inputSourceIndices[k] != secSrc) {
          int chainSrcIdx = slot.inputSourceIndices[k];
          if (chainSrcIdx >= 0) {
            primaryInput = outputSlots_[chainSrcIdx];
          } else {
            int extIdx = -(chainSrcIdx + 1);
            if (extIdx < numExt) primaryInput = externalArrays[extIdx];
          }
          break;
        }
      }
    }

    // 2. Gather secondary inputs
    sd::ops::helpers::FusedElemOp fusedOps[8];
    NDArray* secondaryInputs[8] = {};

    for (int ci = 0; ci < slot.fusedChainLength; ci++) {
      fusedOps[ci] = static_cast<sd::ops::helpers::FusedElemOp>(slot.fusedChainOpCodes[ci]);

      int secSrc = slot.fusedChainSecondaryInputSources[ci];
      if (secSrc != INT32_MIN) {
        if (secSrc >= 0) {
          secondaryInputs[ci] = outputSlots_[secSrc];
          // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore needed
        } else {
          int extIdx = -(secSrc + 1);
          if (extIdx < numExt) secondaryInputs[ci] = externalArrays[extIdx];
        }
      }
      if (ci == 0 && secondaryInputs[ci] == nullptr &&
          sd::ops::helpers::isBinaryFusedOp(fusedOps[ci]) &&
          slot.numInputs == 2 &&
          slot.inputSourceIndices[0] == slot.inputSourceIndices[1]) {
        secondaryInputs[ci] = primaryInput;
      }
    }

    // 3. Determine output shape considering broadcasting
    const LongType* outputShapeInfo = primaryInput->shapeInfo();
    bool needsBroadcast = false;
    for (int ci = 0; ci < slot.fusedChainLength; ci++) {
      if (secondaryInputs[ci] != nullptr && sd::ops::helpers::isBinaryFusedOp(fusedOps[ci])) {
        const LongType* secShape = secondaryInputs[ci]->shapeInfo();
        if (!shape::equalsSoft(outputShapeInfo, secShape)) {
          needsBroadcast = true;
          break;
        }
      }
    }

    if (needsBroadcast) {
      slot.isFusedChainHead = false;
      for (int ci = 0; ci < slot.fusedChainLength; ci++) {
        int chainSlotIdx = slot.fusedChainSlots[ci];
        if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_) {
          slots_[chainSlotIdx].isFusedChainTail = false;
        }
      }
      slot.fusedChainLength = 0;
      goto normalExecution;
    }

    // 4. Allocate/reuse output for the LAST chain slot
    int lastSlotIdx = slot.fusedChainSlots[slot.fusedChainLength - 1];
    int lastOutputSlotIdx = slots_[lastSlotIdx].outputSlotIndices[0];

    NDArray* output = nullptr;
    if (lastOutputSlotIdx >= 0 && lastOutputSlotIdx < totalOutputSlots_) {
      output = slotArrayCache_[lastOutputSlotIdx];
        if (output != nullptr) {
          if (!shape::equalsSoft(output->shapeInfo(), outputShapeInfo)) {
            // Same plan = same shapes. Shape mismatch is a bug — use a different plan.
            DSP_DIAG(EXECUTE, "SHAPE MISMATCH at fused chain slot %d (cached vs expected) — "
                     "same plan should never see different shapes", lastSlotIdx);
          discardCachedSlotArray(lastOutputSlotIdx, output, "fused-shape-mismatch");
          output = nullptr;
        }
      }
      if (output == nullptr) {
        output = DSP_NEW_ARRAY(const_cast<LongType*>(outputShapeInfo), true, LaunchContext::defaultContext());
        slotArrayCache_[lastOutputSlotIdx] = output;
      }
      if (!isBatchZeroActive() && !shouldPreserveWarmupOutputsDuringCapture()) {
        if (isBatchZeroRegistering() && DSP_BUF(output) != nullptr) {
          registerBatchZeroBuffer(DSP_BUF(output),
                                  output->dataBuffer()->getLenInBytes(),
                                  lastOutputSlotIdx);
        }
        output->nullify();
      }
    } else {
      DSP_DIAG(EXECUTE, "invalid output slot index for fused chain tail slot %d", lastSlotIdx);
      return Status::BAD_INPUT;
    }

    // 5. Call fused kernel
    LaunchContext* lc = LaunchContext::defaultContext();
    sd::ops::helpers::fusedElementwiseChain(
        primaryInput, output, fusedOps, slot.fusedChainLength,
        secondaryInputs, nullptr, nullptr, lc);

    output->tickWriteDevice();

    // 6. Register result at all chain slots' output indices
    DSP_SLOT_WRITE(lastOutputSlotIdx, output, "fused-chain-head");
    for (int ci = 0; ci < slot.fusedChainLength - 1; ci++) {
      int chainSlotIdx = slot.fusedChainSlots[ci];
      int chainOutputSlotIdx = slots_[chainSlotIdx].outputSlotIndices[0];
      if (chainOutputSlotIdx >= 0 && chainOutputSlotIdx < totalOutputSlots_) {
        DSP_SLOT_WRITE(chainOutputSlotIdx, output, "fused-chain-member");
      }
    }

#ifdef SD_CUDA
    if (Environment::getInstance().tritonVerifyKernels()) {
      int outIndices[1] = { lastOutputSlotIdx };
      dspLogSlotOutput(stepIdx, slot.opName.c_str(), "FUSED_HEAD",
                          outputSlots_, outIndices, 1, totalOutputSlots_);
    }
#endif

    // Backfill shapes for head AND all tail slots in the fused chain
    backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
    for (int ci = 0; ci < slot.fusedChainLength; ci++) {
      int chainSlotIdx = slot.fusedChainSlots[ci];
      if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_) {
        backfillCachedOutputShapes(slots_[chainSlotIdx], outputSlots_, totalOutputSlots_);
      }
    }

    return Status::OK;
  }

  // ── Fast path: frozen context ────────────────────────────────────────────
  if (slot.frozenContextReady()) {

    // ── View-capable fast path (reshape/expand_dims/squeeze/strided_slice) ──
    if (slot.isViewCapableOp && slot.numInputs >= 1 && slot.numOutputs >= 1) {
      int si = slot.outputSlotIndices[0];
      if (si >= 0 && si < totalOutputSlots_) {
        // Resolve input0 from slot source indices
        int srcIdx = slot.inputSourceIndices[0];
        NDArray* input0 = nullptr;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          input0 = outputSlots_[srcIdx];
        } else if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt) input0 = externalArrays[extIdx];
        }

        if (input0 != nullptr && slot.shapeCacheValid() && !slot.cachedOutputShapes.empty()) {
          // DIAGNOSTIC: trace frozen view installation for controller slots
          bool traceThis = DSP_DIAG_ENABLED(SHAPE) &&
              (stepIdx == 349 || stepIdx == 350 || stepIdx == 357 || stepIdx == 358);
          if (traceThis) {
            auto* db = input0->dataBuffer();
            DSP_DIAG_SLOT(SHAPE, stepIdx,
                "FROZEN-VIEW-TRACE: slot %d (%s) input0 len=%lld dtype=%d srcIdx=%d "
                "pAct=%d sAct=%d cachedShape=%s values=%s",
                stepIdx, slot.opName.c_str(),
                (long long)input0->lengthOf(), (int)input0->dataType(), srcIdx,
                db ? db->isPrimaryActual() : -1,
                db ? db->isSpecialActual() : -1,
                ShapeUtils::shapeAsString(slot.cachedOutputShapes[0]).c_str(),
                dspDumpHostDeviceValues(input0, 16).c_str());
          }
          // Gather all inputs for strided_slice offset computation
          static thread_local std::vector<NDArray*> ssInputs;
          ssInputs.resize(slot.numInputs);
          ssInputs[0] = input0;
          for (int ii = 1; ii < slot.numInputs; ii++) {
            int iiSrc = slot.inputSourceIndices[ii];
            if (iiSrc >= 0 && iiSrc < totalOutputSlots_) {
              ssInputs[ii] = outputSlots_[iiSrc];
            } else if (iiSrc < 0) {
              int iiExt = -(iiSrc + 1);
              ssInputs[ii] = (iiExt < numExt) ? externalArrays[iiExt] : nullptr;
            }
          }

          NDArray* newView = nullptr;
          LongType viewOffset = 0;
          ViewCreateResult vcr = tryCreateViewForSlot(
              stepIdx, slot, input0, slot.cachedOutputShapes[0],
              ssInputs.data(), slot.numInputs,
              &newView, &viewOffset);

          if (vcr == VIEW_STRIDED_SLICE_FAIL) {
            // Can't create view (non-unit strides or newAxisMask) — fall through to normal execution
            if (slot.state_ >= NativeSlot::SlotState::FROZEN)
              slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;
            goto normalExecution;
          } else if (vcr == VIEW_STALE_EMPTY_SHAPE) {
            if (executeCount_ > 1) {
              sd_printf("DSP BUG: Frozen output empty post-warmup at slot %d (%s) — persistence bug. executeCount=%d\n",
                        stepIdx, slot.opName.c_str(), executeCount_);
            }
            // Stale empty shape from Step 0 — input is now non-empty.
            // Invalidate frozen state and fall through to normal path
            // for shape re-inference with actual input shapes.
            DSP_DIAG_SLOT(SHAPE, stepIdx,
                "view-capable slot %d (%s): frozen shape empty but input "
                "non-empty (len=%lld) — re-inferring via normal path (warmup)",
                stepIdx, slot.opName.c_str(), input0->lengthOf());
            slot.state_ = NativeSlot::SlotState::WARMUP;
            goto normalExecution;
          } else if (vcr == VIEW_CREATED) {
            // Check if existing view already matches — skip re-creation
            NDArray* currentOut = outputSlots_[si];
            if (currentOut != nullptr && currentOut->dataBuffer() == input0->dataBuffer()
                && currentOut->offset() == viewOffset) {
              NDArray* tracedInputs[1] = {input0};
              NDArray* tracedOutputs[1] = {currentOut};
              traceSmallControlSlotIO("frozen-view-reuse", stepIdx, slot,
                                      input0, currentOut,
                                      executeCount_, planPhase_);
              traceSmallControlSlotTensors("frozen-view-reuse", stepIdx, slot,
                                           tracedInputs, 1, tracedOutputs, 1,
                                           executeCount_, planPhase_);
              delete newView;  // Discard redundant view
              backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
              return Status::OK;
            }
            // Install the new view
            DSP_SLOT_WRITE(si, newView, "view-install");
            slotIsViewProducer_[si] = true;
            auto& ctx2 = *contextPool_[stepIdx];
            ctx2.setOutputArray(0, newView);
            ctx2.setInputArray(0, input0);
            // In Phase 2, slotArrayCache_ == outputSlots_. Deleting 'old'
            // would delete the slot's output array, leaving a dangling pointer.
            NDArray* old = slotArrayCache_[si];
            if (old != nullptr && old != newView
                && slotArrayCache_ != outputSlots_) {
              if (!isSlotArrayShared(old, si)) {
                delete old;  // View wrapper only — no GPU memory freed
              }
            }
            slotArrayCache_[si] = newView;
            NDArray* tracedInputs[1] = {input0};
            NDArray* tracedOutputs[1] = {newView};
            traceSmallControlSlotIO("frozen-view-install", stepIdx, slot,
                                    input0, newView,
                                    executeCount_, planPhase_);
            traceSmallControlSlotTensors("frozen-view-install", stepIdx, slot,
                                         tracedInputs, 1, tracedOutputs, 1,
                                         executeCount_, planPhase_);
            backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
            return Status::OK;
          }
          // VIEW_NOT_POSSIBLE — fall through to frozen fallback / normal execution below
        }
      }
    }

    // Fallback: view-capable op in frozen path but view fast path didn't handle it.
    // If input is non-empty but cached outputs are empty, re-infer via normal path.
    if (slot.isViewCapableOp && slot.numInputs >= 1) {
      int srcIdx0 = slot.inputSourceIndices[0];
      NDArray* inp0 = nullptr;
      if (srcIdx0 >= 0 && srcIdx0 < totalOutputSlots_) inp0 = outputSlots_[srcIdx0];
      else if (srcIdx0 < 0) {
        int extIdx = -(srcIdx0 + 1);
        if (extIdx < numExt) inp0 = externalArrays[extIdx];
      }
      if (inp0 != nullptr && !inp0->isEmpty() && inp0->lengthOf() > 0) {
        // Check if all frozen outputs are empty
        bool allEmpty = true;
        for (int oi = 0; oi < slot.numOutputs; oi++) {
          int si = slot.outputSlotIndices[oi];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr &&
              !outputSlots_[si]->isEmpty()) {
            allEmpty = false;
            break;
          }
        }
        if (allEmpty) {
          if (executeCount_ > 1) {
            sd_printf("DSP BUG: Frozen output empty post-warmup at slot %d (%s) — persistence bug. executeCount=%d\n",
                      stepIdx, slot.opName.c_str(), executeCount_);
          }
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "view-capable slot %d (%s): frozen outputs empty but input non-empty "
              "(len=%lld) — re-inferring via normal path (fallback, warmup)",
              stepIdx, slot.opName.c_str(), inp0->lengthOf());
          slot.state_ = NativeSlot::SlotState::WARMUP;
          goto normalExecution;
        }
      }
    }

    auto& ctx = *contextPool_[stepIdx];

    // Refresh ALL inputs from current outputSlots_[].
    // Gap ops (shape_of, gather, matmul) re-execute each step and write new
    // arrays to outputSlots_[]. Without refreshing ALL slot-sourced inputs,
    // this frozen op's context retains stale pointers from warmup.
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
          ctx.setInputArray(i, externalArrays[extIdx]);
        }
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        if (outputSlots_[srcIdx] != nullptr) {
          ctx.setInputArray(i, outputSlots_[srcIdx]);
        }
      }
    }

    // Nullify output arrays before re-execution.
    // Skip outputs that share a data buffer with any input — those are views,
    // and zeroing them would corrupt the input data.
    if (!slot.inPlaceFused) {
      auto& ctxOuts = ctx.fastpath_out();
      auto& ctxIns = ctx.fastpath_in();
      for (int i = 0; i < static_cast<int>(ctxOuts.size()); i++) {
        if (ctxOuts[i] == nullptr) continue;
        int si = (i < slot.numOutputs) ? slot.outputSlotIndices[i] : -1;
        if (si >= 0 && si < totalOutputSlots_ && slotIsViewProducer_[si]) continue;

        // Check if this output shares a buffer with any input — if so it's a view
        auto* outBuf = ctxOuts[i]->dataBuffer();
        bool isView = false;
        if (outBuf != nullptr) {
          for (int j = 0; j < static_cast<int>(ctxIns.size()); j++) {
            if (ctxIns[j] != nullptr && ctxIns[j]->dataBuffer() == outBuf) {
              isView = true;
              break;
            }
          }
        }
        if (isView) continue;

        if (!isBatchZeroActive() && !shouldPreserveWarmupOutputsDuringCapture()) {
          if (isBatchZeroRegistering() && DSP_BUF(ctxOuts[i]) != nullptr) {
            registerBatchZeroBuffer(DSP_BUF(ctxOuts[i]),
                                    ctxOuts[i]->dataBuffer()->getLenInBytes(),
                                    si);
          }
          ctxOuts[i]->nullify();
        }
      }
    }

    // Log attention op inputs for frozen tracking / lineage debugging
    if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
      const char* opName = slot.op->getOpName()->c_str();
      if (strcmp(opName, "onnx_multi_head_attention") == 0 || strcmp(opName, "dot_product_attention_v2") == 0) {
        int nIn = (int)ctx.fastpath_in().size();
        DSP_DIAG(EXECUTE, "ATTN_OP_INPUTS: %s step=%d slot=%d numInputs=%d planInputs=%d",
                 opName, executeCount_, stepIdx, nIn, slot.numInputs);
        if (nIn > 6) {
          NDArray* cachePos = ctx.fastpath_in()[6];
          DSP_DIAG(EXECUTE, "ATTN_OP_CACHE_POS: input[6]=%p type=%d val=%lld",
                   cachePos, cachePos ? (int)cachePos->dataType() : -1,
                   (cachePos && cachePos->lengthOf() > 0) ? cachePos->e<sd::LongType>(0) : -999);
        }
      }
    }

    auto status = slot.op->execute(&ctx);

    auto& ctxOuts = ctx.fastpath_out();
    for (int i = 0; i < slot.numOutputs && i < static_cast<int>(ctxOuts.size()); i++) {
      if (ctxOuts[i] != nullptr) {
        reconcileExecutedOutputActuality("frozen-op-exec", stepIdx, slot, ctxOuts[i]);
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          // In Phase 2, slotArrayCache_ == outputSlots_. Deleting oldCached
          // would delete the slot's output array, leaving a dangling pointer.
          NDArray* oldCached = slotArrayCache_[si];
          if (oldCached != nullptr && oldCached != ctxOuts[i] && !tl_graphExecutionActive
              && slotArrayCache_ != outputSlots_) {
            if (!isSlotArrayShared(oldCached, si)) {
              delete oldCached;  // Replace stale cached array inline
              // During graph capture, don't delete — it's the saved warmup array
            }
          }
          DSP_SLOT_WRITE(si, ctxOuts[i], "frozen-op-exec");
          slotArrayCache_[si] = ctxOuts[i];
        }
      }
    }

#ifdef SD_CUDA
    if (Environment::getInstance().tritonVerifyKernels()) {
      dspLogSlotOutput(stepIdx, slot.opName.c_str(), "OP_EXEC(frozen)",
                          outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
    }
#endif

    if (status == Status::OK) {
      NDArray* tracedInput0 = slot.numInputs > 0 && !ctx.fastpath_in().empty()
                                  ? ctx.fastpath_in()[0]
                                  : nullptr;
      NDArray* tracedOutput0 = slot.numOutputs > 0 && !ctxOuts.empty()
                                   ? ctxOuts[0]
                                   : nullptr;
      traceSmallControlSlotIO("frozen-op-exec", stepIdx, slot,
                              tracedInput0, tracedOutput0,
                              executeCount_, planPhase_);
      traceSmallControlSlotTensors("frozen-op-exec", stepIdx, slot,
                                   ctx.fastpath_in().data(),
                                   static_cast<int>(ctx.fastpath_in().size()),
                                   ctxOuts.data(),
                                   static_cast<int>(ctxOuts.size()),
                                   executeCount_, planPhase_);
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
    }
    return status;
  }

  normalExecution:
  // ── Step 1: Gather inputs ────────────────────────────────────────────────
  static thread_local std::vector<NDArray*> inputs;
  inputs.resize(slot.numInputs);
  for (int i = 0; i < slot.numInputs; i++) {
    int srcIdx = slot.inputSourceIndices[i];
    if (srcIdx >= 0) {
      inputs[i] = outputSlots_[srcIdx];
      // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore needed
    } else {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) {
        inputs[i] = externalArrays[extIdx];
      } else {
        inputs[i] = nullptr;
      }
    }

    if (inputs[i] == nullptr) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx, "NULL input for slot %d (%s) input %d, srcIdx=%d",
                stepIdx, slot.opName.c_str(), i, slot.inputSourceIndices[i]);
      return Status::BAD_INPUT;
    }

    // Validate shapeInfo is present — an NDArray with nullptr _shapeInfo is
    // uninitialized or destroyed. Using it would crash inside op->execute().
    // This catches use-after-free and stale cache entries early.
    {
      const LongType* si = nullptr;
      try { si = inputs[i]->shapeInfo(); } catch (...) { si = nullptr; }
      if (si == nullptr) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
            "NULL shapeInfo for slot %d (%s) input %d, srcIdx=%d "
            "ptr=%p db=%p exec=%d",
            stepIdx, slot.opName.c_str(), i, slot.inputSourceIndices[i],
            (void*)inputs[i], (void*)inputs[i]->dataBuffer(), executeCount_);
        // Invalidate the cache entry so future executions don't hit the same bad array
        int badSrcIdx = slot.inputSourceIndices[i];
        if (badSrcIdx >= 0 && badSrcIdx < totalOutputSlots_) {
          DSP_SLOT_WRITE(badSrcIdx, nullptr, "null-bad-input");
          slotArrayCache_[badSrcIdx] = nullptr;
        }
        return Status::BAD_INPUT;
      }
    }

    // Validate GPU buffer is alive — detect freed DataBuffers that would cause
    // CUDA illegal memory access (error 700) downstream.
    auto* db = inputs[i]->dataBuffer();
    if (db != nullptr && db->isClosed()) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
          "CLOSED DataBuffer for slot %d (%s) input %d, srcIdx=%d "
          "specialBuf=%p isConst=%d exec=%d",
          stepIdx, slot.opName.c_str(), i, slot.inputSourceIndices[i],
          db->special(), db->isConstant ? 1 : 0, executeCount_);
      return Status::BAD_INPUT;
    }
#if defined(SD_CUDA)
    // On CUDA, a null special (device) buffer indicates a freed or unsynced DataBuffer
    // that would cause illegal memory access (error 700). On CPU, special() is always null.
    if (db != nullptr && db->special() == nullptr && !inputs[i]->isEmpty()) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
          "NULL GPU buffer for slot %d (%s) input %d, srcIdx=%d "
          "len=%lld isClosed=%d isConst=%d exec=%d fromCache=%d",
          stepIdx, slot.opName.c_str(), i, slot.inputSourceIndices[i],
          (long long)inputs[i]->lengthOf(), db->isClosed() ? 1 : 0,
          db->isConstant ? 1 : 0, executeCount_,
          (outputSlots_[slot.inputSourceIndices[i]] == nullptr) ? 1 : 0);
      return Status::BAD_INPUT;
    }
#else
    // On CPU, validate the primary (host) buffer instead.
    if (db != nullptr && db->primary() == nullptr && !inputs[i]->isEmpty()) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
          "NULL host buffer for slot %d (%s) input %d, srcIdx=%d "
          "len=%lld isClosed=%d isConst=%d exec=%d fromCache=%d",
          stepIdx, slot.opName.c_str(), i, slot.inputSourceIndices[i],
          (long long)inputs[i]->lengthOf(), db->isClosed() ? 1 : 0,
          db->isConstant ? 1 : 0, executeCount_,
          (outputSlots_[slot.inputSourceIndices[i]] == nullptr) ? 1 : 0);
      return Status::BAD_INPUT;
    }
#endif
  }


  // ── Step 2: Shape inference ──────────────────────────────────────────────
  LongType shapeKey = 0;
  bool cacheHit;
  if (shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid() &&
      !slot.outputShapeDependsOnInputValues) {
    cacheHit = true;

    // View-capable ops: check for stale empty cached shapes.
    // During Step 0 (first execution), KV cache inputs are empty, so shape
    // inference produces empty output shapes that get cached. On subsequent
    // executions the inputs become non-empty (concat grows KV), but
    // shapesFrozen_ prevents re-inference. Detect this and force a cache miss.
    if (slot.isViewCapableOp && !slot.cachedOutputShapes.empty()) {
      bool allCachedEmpty = true;
      for (const auto& s : slot.cachedOutputShapes) {
        if (!shape::isEmpty(const_cast<LongType*>(s))) {
          allCachedEmpty = false;
          break;
        }
      }
      if (allCachedEmpty && slot.numInputs > 0 && inputs[0] != nullptr &&
          !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "view-capable slot %d (%s): cached shapes ALL empty but input[0] "
            "non-empty (len=%lld) — forcing shape re-inference",
            stepIdx, slot.opName.c_str(), inputs[0]->lengthOf());
        cacheHit = false;
        shapeKey = computeShapeKey(slot, inputs.data(), slot.numInputs);
      }
    }
  } else {
    shapeKey = computeShapeKey(slot, inputs.data(), slot.numInputs);
    cacheHit = slot.shapeCacheValid() && (slot.cachedShapeKey == shapeKey);

    // View-capable ops: even with matching shape key, check for stale empty
    // cached shapes. computeShapeKey does NOT include the ARRAY_EMPTY flag,
    // so an input that transitions from empty (ARRAY_EMPTY) to non-empty with
    // the same dimensions produces an identical key but needs re-inference.
    if (cacheHit && slot.isViewCapableOp && !slot.cachedOutputShapes.empty()) {
      bool allCachedEmpty = true;
      for (const auto& s : slot.cachedOutputShapes) {
        if (!shape::isEmpty(const_cast<LongType*>(s))) {
          allCachedEmpty = false;
          break;
        }
      }
      if (allCachedEmpty && slot.numInputs > 0 && inputs[0] != nullptr &&
          !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "view-capable slot %d (%s): cache hit but shapes ALL empty while "
            "input[0] non-empty (len=%lld) — forcing re-inference (key match)",
            stepIdx, slot.opName.c_str(), inputs[0]->lengthOf());
        cacheHit = false;
      }
    }
  }

  std::vector<const LongType*> outputShapes;
  if (cacheHit) {
    outputShapes = slot.cachedOutputShapes;
  } else {
    const bool allowRestabilization =
        shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid() &&
        allowFrozenShapeRestabilization(stepIdx, slot, planPhase_,
                                        slots_, numSlots_,
                                        outputSlots_, totalOutputSlots_,
                                        externalArrays, numExt);

    // ── Phase violation: shape change during SHAPES_FROZEN ──
    // If shapes are frozen and we get a cache miss (shape changed), this is
    // a phase contract violation. Return a hard error — the caller's frozen
    // assumption is broken and continuing would produce wrong results or crash.
    if (shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid()) {
      // For value-dependent ops, shape key changes are EXPECTED because the key
      // hashes input VALUES (which change per step). But the OUTPUT SHAPE should
      // stay the same (shapes are frozen). Recompute shapes and verify they match.
      if (slot.outputShapeDependsOnInputValues) {
        // Value-dep op: key changed but shapes may be the same. Recompute and check.
        // Update the cached key to the new value so subsequent checks don't re-fire.
        slot.cachedShapeKey = shapeKey;
        DSP_DIAG(SHAPE, "VALUE_DEP_KEY_UPDATE: slot %d (%s) shape key updated "
                 "(value-dep op, input values changed but shapes frozen)",
                 stepIdx, slot.opName.c_str());
        // Fall through to shape recomputation below — it will verify actual shapes match
      } else if (allowRestabilization) {
        DSP_DIAG(SHAPE,
                 "FROZEN_RESTABILIZE: slot %d (%s) cache miss during SHAPES_FROZEN "
                 "is allowed because the shape-control chain is plan-internal and "
                 "planPhase=%d (< POINTERS_STABLE)",
                 stepIdx, slot.opName.c_str(), static_cast<int>(planPhase_));
      } else {
        // Non-value-dep op: shape key change means actual shapes changed → violation
        char errBuf[512];
        snprintf(errBuf, sizeof(errBuf),
                 "LIFECYCLE_ERROR: shape changed at slot %d (%s) during SHAPES_FROZEN phase "
                 "(execCount=%d). Shapes were assumed constant but input shapes changed. "
                 "oldKey=0x%llx newKey=0x%llx. Unfreeze shapes before changing input shapes.",
                 stepIdx, slot.opName.c_str(), executeCount_,
                 (long long)slot.cachedShapeKey, (long long)shapeKey);
        DSP_DIAG(SHAPE, "%s", errBuf);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
            static_cast<int>(Status::KERNEL_FAILURE));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
        return Status::KERNEL_FAILURE;
      }
    }

    auto& ctx = *contextPool_[stepIdx];

    for (int i = 0; i < slot.numInputs; i++) {
      ctx.setInputArray(i, inputs[i]);
    }

    if (slot.numIArgs > 0) ctx.setIArguments(slot.iArgs, slot.numIArgs);
    if (slot.numTArgs > 0) ctx.setTArguments(slot.tArgs, slot.numTArgs);
    if (slot.numBArgs > 0) ctx.setBArguments(slot.bArgs, slot.numBArgs);
    if (slot.numDArgs > 0) ctx.setDArguments(slot.dArgs, slot.numDArgs);
    ctx.getSArguments()->clear();
    if (slot.numSArgs > 0) {
      ctx.getSArguments()->insert(ctx.getSArguments()->end(), slot.sArgs, slot.sArgs + slot.numSArgs);
    }

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
      DSP_DIAG_SLOT(SHAPE, stepIdx, "shape inference EXCEPTION at slot %d (%s): %s",
                stepIdx, slot.opName.c_str(), e.what());
      // Propagate error detail to Java via errorReference (otherwise Java only sees "status 50")
      std::string inputShapeStr;
      for (int ii = 0; ii < slot.numInputs; ii++) {
        if (ii > 0) inputShapeStr += ", ";
        inputShapeStr += inputs[ii] ? ShapeUtils::shapeAsString(inputs[ii]) : "null";
      }
      std::string smallInputValueStr;
      for (int ii = 0; ii < slot.numInputs; ii++) {
        if (inputs[ii] == nullptr) continue;
        auto dt = inputs[ii]->dataType();
        auto len = inputs[ii]->lengthOf();
        if ((dt == INT32 || dt == INT64 || dt == BOOL) && len > 0 && len <= 16) {
          if (!smallInputValueStr.empty()) smallInputValueStr += " | ";
          smallInputValueStr += "input[";
          smallInputValueStr += std::to_string(ii);
          smallInputValueStr += "]=";
          smallInputValueStr += dspDumpHostDeviceValues(inputs[ii], 16);
        }
      }
      std::string upstreamTraceStr;
      for (int ii = 0; ii < slot.numInputs; ii++) {
        if (ii > 0) upstreamTraceStr += " | ";
        upstreamTraceStr += "input[";
        upstreamTraceStr += std::to_string(ii);
        upstreamTraceStr += "]=>";
        std::vector<int> seenSlotSteps;
        appendUpstreamControlTrace(upstreamTraceStr, slot.inputSourceIndices[ii], 3,
                                   slots_, numSlots_, outputSlots_, totalOutputSlots_,
                                   externalArrays, numExt, externalInputNames_,
                                   seenSlotSteps);
      }
      char errBuf[1024];
      snprintf(errBuf, sizeof(errBuf), "slot %d (%s) shape inference failed: %s | inputs=[%s] iArgs=%d",
               stepIdx, slot.opName.c_str(), e.what(), inputShapeStr.c_str(), slot.numIArgs);
      std::string errMsg = errBuf;
      if (!smallInputValueStr.empty()) {
        errMsg += " smallInputs=[";
        errMsg += smallInputValueStr;
        errMsg += "]";
      }
      if (!upstreamTraceStr.empty()) {
        errMsg += " upstream=[";
        errMsg += upstreamTraceStr;
        errMsg += "]";
      }
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg.c_str());
      return Status::KERNEL_FAILURE;
    }
    if (shapeList == nullptr || shapeList->size() == 0) {
      DSP_DIAG_SLOT(SHAPE, stepIdx, "shape inference returned null for slot %d (%s)",
                stepIdx, slot.opName.c_str());
      char errBuf[512];
      snprintf(errBuf, sizeof(errBuf), "slot %d (%s) shape inference returned null/empty",
               stepIdx, slot.opName.c_str());
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
      return Status::KERNEL_FAILURE;
    }

    outputShapes.resize(shapeList->size());
    for (int i = 0; i < static_cast<int>(shapeList->size()); i++) {
      try {
        auto cached = ConstantShapeHelper::getInstance().createFromExisting(
            const_cast<LongType*>(shapeList->at(i)));
        outputShapes[i] = cached;
      } catch (const std::exception& e) {
        DSP_DIAG_SLOT(SHAPE, stepIdx, "createFromExisting EXCEPTION at slot %d (%s) output[%d]: %s",
                  stepIdx, slot.opName.c_str(), i, e.what());
        delete shapeList;
        return Status::KERNEL_FAILURE;
      }
    }

    if (shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid() &&
        slot.outputShapeDependsOnInputValues && !slot.cachedOutputShapes.empty()) {
      bool sameOutputShapeCount = slot.cachedOutputShapes.size() == outputShapes.size();
      bool outputShapesMatch = sameOutputShapeCount;
      if (outputShapesMatch) {
        for (size_t i = 0; i < outputShapes.size(); i++) {
          const LongType* oldShape = slot.cachedOutputShapes[i];
          const LongType* newShape = outputShapes[i];
          if (oldShape == nullptr || newShape == nullptr ||
              !shape::equalsSoft(const_cast<LongType*>(oldShape),
                                 const_cast<LongType*>(newShape)) ||
              ArrayOptions::dataType(oldShape) != ArrayOptions::dataType(newShape)) {
            outputShapesMatch = false;
            break;
          }
        }
      }

      if (!outputShapesMatch) {
        if (allowRestabilization) {
          DSP_DIAG(SHAPE,
                   "FROZEN_RESTABILIZE: slot %d (%s) output shape changed during SHAPES_FROZEN "
                   "but the controlling value-shape chain is plan-internal and "
                   "planPhase=%d (< POINTERS_STABLE). Updating cached shapes.",
                   stepIdx, slot.opName.c_str(), static_cast<int>(planPhase_));
        } else {
          std::string oldShapesStr;
          std::string newShapesStr;
          for (size_t i = 0; i < std::max(slot.cachedOutputShapes.size(), outputShapes.size()); i++) {
            if (i > 0) {
              oldShapesStr += ", ";
              newShapesStr += ", ";
            }
            oldShapesStr += (i < slot.cachedOutputShapes.size() && slot.cachedOutputShapes[i] != nullptr)
                ? ShapeUtils::shapeAsString(slot.cachedOutputShapes[i])
                : "null";
            newShapesStr += (i < outputShapes.size() && outputShapes[i] != nullptr)
                ? ShapeUtils::shapeAsString(outputShapes[i])
                : "null";
          }

          std::string inputShapeStr;
          for (int ii = 0; ii < slot.numInputs; ii++) {
            if (ii > 0) inputShapeStr += ", ";
            inputShapeStr += inputs[ii] ? ShapeUtils::shapeAsString(inputs[ii]) : "null";
          }

          std::string smallInputValueStr;
          for (int ii = 0; ii < slot.numInputs; ii++) {
            if (inputs[ii] == nullptr) continue;
            auto dt = inputs[ii]->dataType();
            auto len = inputs[ii]->lengthOf();
            if ((dt == INT32 || dt == INT64 || dt == BOOL) && len > 0 && len <= 16) {
              if (!smallInputValueStr.empty()) smallInputValueStr += " | ";
              smallInputValueStr += "input[";
              smallInputValueStr += std::to_string(ii);
              smallInputValueStr += "]=";
              smallInputValueStr += dspDumpHostDeviceValues(inputs[ii], 16);
            }
          }

          std::string upstreamTraceStr;
          for (int ii = 0; ii < slot.numInputs; ii++) {
            if (ii > 0) upstreamTraceStr += " | ";
            upstreamTraceStr += "input[";
            upstreamTraceStr += std::to_string(ii);
            upstreamTraceStr += "]=>";
            std::vector<int> seenSlotSteps;
            appendUpstreamControlTrace(upstreamTraceStr, slot.inputSourceIndices[ii], 3,
                                       slots_, numSlots_, outputSlots_, totalOutputSlots_,
                                       externalArrays, numExt, externalInputNames_,
                                       seenSlotSteps);
          }

          std::string errMsg = "LIFECYCLE_ERROR: value-dependent output shape changed at slot " +
              std::to_string(stepIdx) + " (" + slot.opName + ") during SHAPES_FROZEN phase "
              "(execCount=" + std::to_string(executeCount_) + "). oldShapes=[" + oldShapesStr +
              "] newShapes=[" + newShapesStr + "] inputs=[" + inputShapeStr + "]";
          if (!smallInputValueStr.empty()) {
            errMsg += " smallInputs=[" + smallInputValueStr + "]";
          }
          if (!upstreamTraceStr.empty()) {
            errMsg += " upstream=[" + upstreamTraceStr + "]";
          }

          DSP_DIAG(SHAPE, "%s", errMsg.c_str());
          if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_VERIFY)) {
            dspDumpSlotInputs(stepIdx, slot.opName.c_str(),
                              slot.numInputs, slot.inputSourceIndices, slot.inputSourceTypes,
                              outputSlots_, totalOutputSlots_,
                              externalArrays, numExt, externalInputNames_,
                              "value-shape-mismatch", 8);
          }
          sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
              static_cast<int>(Status::KERNEL_FAILURE));
          sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg.c_str());
          delete shapeList;
          return Status::KERNEL_FAILURE;
        }
      }

      // Shapes are still frozen-stable. Update only the key, keep the original
      // cached shape-info pointers to avoid churn during steady-state replay.
      if (allowRestabilization && !outputShapesMatch) {
        slot.cachedShapeKey = shapeKey;
        slot.cachedOutputShapes = outputShapes;
        if (slot.state_ < NativeSlot::SlotState::SHAPE_CACHED) {
          slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;
        }
      } else {
        slot.cachedShapeKey = shapeKey;
        outputShapes = slot.cachedOutputShapes;
      }
    } else {
      slot.cachedShapeKey = shapeKey;
      slot.cachedOutputShapes = outputShapes;
      if (slot.state_ < NativeSlot::SlotState::SHAPE_CACHED)
        slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }

    delete shapeList;
  }

  // ── Step 3: Allocate/reuse outputs ───────────────────────────────────────
  int numActualOutputs = std::min(slot.numOutputs, static_cast<int>(outputShapes.size()));
  static thread_local std::vector<NDArray*> outputs;
  outputs.resize(numActualOutputs);

  // ── In-place fusion: reuse input buffer as output ──
  if (slot.inPlaceFused && slot.inPlaceFusedInputIdx >= 0 &&
      slot.inPlaceFusedInputIdx < slot.numInputs && numActualOutputs >= 1) {
    NDArray* inPlaceBuffer = inputs[slot.inPlaceFusedInputIdx];
    if (inPlaceBuffer != nullptr) {
      const LongType* expectedShape = outputShapes[0];
      if (shape::equalsSoft(inPlaceBuffer->shapeInfo(), expectedShape) &&
          ArrayOptions::dataType(inPlaceBuffer->shapeInfo()) == ArrayOptions::dataType(expectedShape)) {
        outputs[0] = inPlaceBuffer;
        int slotIdx = slot.outputSlotIndices[0];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          DSP_SLOT_WRITE(slotIdx, inPlaceBuffer, "in-place-fused");
        }

        for (int i = 1; i < numActualOutputs; i++) {
          int si = slot.outputSlotIndices[i];
          if (si < 0) {
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
            outputs[i] = DSP_NEW_ARRAY(const_cast<LongType*>(outputShapes[i]), true);
            if (cacheIdx < untrackedOutputCacheSize_) {
              untrackedOutputCache_[cacheIdx] = outputs[i];
            }
            continue;
          }
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
          std::vector<LongType> shape(rank);
          for (int d = 0; d < rank; d++) shape[d] = shapeInfo[d + 1];
          outputs[i] = DSP_NEW_ARRAY(order, shape, dt);
          DSP_SLOT_WRITE(si, outputs[i], "alloc-output");
          slotArrayCache_[si] = outputs[i];
        }

        goto step4_execute;
      }
    }
  }

  // ── View-capable ops: share input 0's DataBuffer for output 0 ──────────
  if (slot.isViewCapableOp && slot.numInputs >= 1 && numActualOutputs >= 1) {
    NDArray* input0 = inputs[0];
    if (input0 != nullptr) {
      NDArray* view = nullptr;
      LongType viewOffset = 0;
      ViewCreateResult vcr = tryCreateViewForSlot(
          stepIdx, slot, input0, outputShapes[0],
          inputs.data(), slot.numInputs,
          &view, &viewOffset);

      if (vcr == VIEW_STRIDED_SLICE_FAIL) {
        // Non-unit strides or newAxisMask — can't create contiguous view, fall through
        goto step3_allocate;
      } else if (vcr == VIEW_CREATED) {
        int slotIdx = slot.outputSlotIndices[0];
        outputs[0] = view;
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          // DIAGNOSTIC: trace view-op-install buffer stability for lifecycle debugging
          NDArray* oldCached = slotArrayCache_[slotIdx];
          NDArray* oldOutput = outputSlots_[slotIdx];
          auto* newDb = view->dataBuffer();
          int srcIdx0 = slot.inputSourceIndices[0];
          // Always trace for reshape ops during frozen execution
          bool isReshape = (slot.opName == "reshape" || slot.opName == "reshape_no_copy");
          if (isReshape && executeCount_ >= 2) {
            auto* oldCachedDb = oldCached != nullptr ? oldCached->dataBuffer() : nullptr;
            auto* oldOutputDb = oldOutput != nullptr ? oldOutput->dataBuffer() : nullptr;
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                "VIEW-BUF-TRACE: slot %d (%s) oldCachedDb=%p oldOutputDb=%p newDb=%p "
                "srcIdx0=%d exec=%d phase=%d cacheNull=%d outputNull=%d",
                stepIdx, slot.opName.c_str(), (void*)oldCachedDb, (void*)oldOutputDb, (void*)newDb,
                srcIdx0, executeCount_, (int)planPhase_,
                oldCached == nullptr ? 1 : 0, oldOutput == nullptr ? 1 : 0);
            if (oldOutputDb != newDb) {
              DSP_DIAG_SLOT(MEMORY, stepIdx,
                  "VIEW-BUF-CHANGE: slot %d (%s) outputSlot db changed! oldOutputDb=%p newDb=%p srcIdx0=%d exec=%d",
                  stepIdx, slot.opName.c_str(), (void*)oldOutputDb, (void*)newDb, srcIdx0, executeCount_);
            }
          }
          if (oldCached != nullptr) {
            auto* oldDb = oldCached->dataBuffer();
            int srcIdx0 = slot.inputSourceIndices[0];
            if (isReshape && executeCount_ >= 2) {
              if (srcIdx0 >= 0) {
                NDArray* srcArr = outputSlots_[srcIdx0];
                auto* srcDb = srcArr != nullptr ? srcArr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-INPUT: slot %d (%s) from slot %d (%s) srcDb=%p oldDb=%p newDb=%p exec=%d phase=%d dbMatch=%d",
                    stepIdx, slot.opName.c_str(), srcIdx0,
                    srcIdx0 < numSlots_ ? slots_[srcIdx0].opName.c_str() : "?",
                    (void*)srcDb, (void*)oldDb, (void*)newDb, executeCount_, (int)planPhase_,
                    oldDb == newDb ? 1 : 0);
              } else {
                int extIdx = -(srcIdx0 + 1);
                auto* extDb = (extIdx < numExt && externalArrays[extIdx] != nullptr)
                    ? externalArrays[extIdx]->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-INPUT: slot %d (%s) from ext[%d] extDb=%p oldDb=%p newDb=%p exec=%d phase=%d dbMatch=%d",
                    stepIdx, slot.opName.c_str(), extIdx,
                    (void*)extDb, (void*)oldDb, (void*)newDb, executeCount_, (int)planPhase_,
                    oldDb == newDb ? 1 : 0);
              }
            }
            if (oldDb != newDb && planPhase_ >= PlanPhase::POINTERS_STABLE && executeCount_ > 1) {
              // Buffer is changing — trace the source
              if (srcIdx0 >= 0) {
                NDArray* srcArr = outputSlots_[srcIdx0];
                auto* srcDb = srcArr != nullptr ? srcArr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-CHANGE: slot %d (%s) input from slot %d (%s) "
                    "srcDb=%p oldDb=%p newDb=%p exec=%d phase=%d",
                    stepIdx, slot.opName.c_str(), srcIdx0,
                    srcIdx0 < numSlots_ ? slots_[srcIdx0].opName.c_str() : "?",
                    (void*)srcDb, (void*)oldDb, (void*)newDb, executeCount_, (int)planPhase_);
              } else {
                int extIdx = -(srcIdx0 + 1);
                auto* extDb = (extIdx < numExt && externalArrays[extIdx] != nullptr)
                    ? externalArrays[extIdx]->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-CHANGE: slot %d (%s) input from ext[%d] "
                    "extDb=%p oldDb=%p newDb=%p exec=%d phase=%d",
                    stepIdx, slot.opName.c_str(), extIdx,
                    (void*)extDb, (void*)oldDb, (void*)newDb, executeCount_, (int)planPhase_);
              }
            }
          }

          // In Phase 2, slotArrayCache_ == outputSlots_ (unified). Deleting 'old'
          // would delete the slot's output array, leaving a dangling pointer for
          // writeOutputSlot. Skip the delete — writeOutputSlot replaces the slot
          // and the plan's cleanup handles the old array.
          NDArray* old = slotArrayCache_[slotIdx];
          if (old != nullptr && old != view && !tl_graphExecutionActive
              && slotArrayCache_ != outputSlots_) {
            if (!isSlotArrayShared(old, slotIdx)) {
              delete old;  // View wrapper only — no GPU memory freed
              // During graph capture, don't delete — it's the saved warmup array
            }
          }
          DSP_SLOT_WRITE(slotIdx, view, "view-op-install");
          slotArrayCache_[slotIdx] = view;
          slotIsViewProducer_[slotIdx] = true;
        }

        for (int i = 1; i < numActualOutputs; i++) {
          int si = slot.outputSlotIndices[i];
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
          std::vector<LongType> shape(rank);
          for (int d = 0; d < rank; d++) shape[d] = shapeInfo[d + 1];
          outputs[i] = DSP_NEW_ARRAY(order, shape, dt);
          if (si >= 0 && si < totalOutputSlots_) {
            DSP_SLOT_WRITE(si, outputs[i], "view-secondary-alloc");
            slotArrayCache_[si] = outputs[i];
          }
        }
        goto step4_execute;
      }
      // VIEW_NOT_POSSIBLE or VIEW_STALE_EMPTY_SHAPE — fall through to step3_allocate
    }
  }

step3_allocate:
  for (int i = 0; i < numActualOutputs; i++) {
    int slotIdx = slot.outputSlotIndices[i];
    if (slotIdx < 0) {
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
          delete cached;
          untrackedOutputCache_[cacheIdx] = nullptr;
        }
      }
      outputs[i] = DSP_NEW_ARRAY(const_cast<LongType*>(outputShapes[i]), true);
      if (cacheIdx < untrackedOutputCacheSize_) {
        untrackedOutputCache_[cacheIdx] = outputs[i];
      }
      continue;
    }

    const LongType* shapeInfo = outputShapes[i];
    auto dt = ArrayOptions::dataType(shapeInfo);
    auto order = shape::order(shapeInfo);
    LongType rank = shape::rank(shapeInfo);

    NDArray* cached = validateReusableSlotArray(slotIdx, slotArrayCache_[slotIdx],
                                                "step3-reuse-validate");
    if (cached != nullptr) {
      const LongType* cachedShape = cached->shapeInfo();
      if (shape::equalsSoft(cachedShape, shapeInfo) &&
          ArrayOptions::dataType(cachedShape) == dt) {
        // CRITICAL: Skip batch-zero for view-capable slots. Their outputs share
        // buffers with inputs and zeroing would corrupt input data.
        if (!isBatchZeroActive() && !shouldPreserveWarmupOutputsDuringCapture() &&
            !slot.isViewCapableOp) {
          if (isBatchZeroRegistering() && DSP_BUF(cached) != nullptr) {
            registerBatchZeroBuffer(DSP_BUF(cached),
                                    cached->dataBuffer()->getLenInBytes(),
                                    slotIdx);
          }
          cached->nullify();
        }
        outputs[i] = cached;
        DSP_SLOT_WRITE(slotIdx, cached, "cached-reuse");
        continue;
      } else {
        // Same plan = same shapes. Shape mismatch — replace inline.
        DSP_DIAG(EXECUTE, "SHAPE MISMATCH at slot %d (cached vs expected) — replacing inline",
                 slotIdx);
        discardCachedSlotArray(slotIdx, cached, "shape-mismatch");
      }
    }

    // Empty arrays: use full-shapeInfo constructor to preserve ARRAY_EMPTY.
    // Also check for zero-length shapes (any dim == 0) even if the flag
    // isn't set, since calculateOutputShape may not always set it.
    bool hasZeroDim = false;
    for (int d = 0; d < rank; d++) {
      if (shapeInfo[d + 1] == 0) { hasZeroDim = true; break; }
    }
    if (shape::isEmpty(const_cast<LongType*>(shapeInfo)) || hasZeroDim) {
      NDArray* emptyOut = DSP_NEW_ARRAY(const_cast<LongType*>(shapeInfo), true);
      outputs[i] = emptyOut;
      DSP_SLOT_WRITE(slotIdx, emptyOut, "empty-output-alloc");
      slotArrayCache_[slotIdx] = emptyOut;
      continue;
    }

    std::vector<LongType> shape(rank);
    for (int d = 0; d < rank; d++) {
      shape[d] = shapeInfo[d + 1];
    }

    // Check if this slot has a max-allocation size configured
    auto maxIt = outputSlotMaxSizes_.find(slotIdx);
    if (maxIt != outputSlotMaxSizes_.end() && maxIt->second > 0) {
      if (maxAllocatedSlots_.find(slotIdx) == maxAllocatedSlots_.end()) {
        LongType maxElements = maxIt->second;
        std::vector<LongType> maxShape = shape;

        if (rank == 4 && maxKvCacheLen_ > 0 && shape[2] > 0 && shape[2] < maxKvCacheLen_) {
          maxShape[2] = maxKvCacheLen_;
        } else if (rank == 4 && maxKvCacheLen_ > 0 && shape[1] > 0 && shape[1] < maxKvCacheLen_) {
          maxShape[1] = maxKvCacheLen_;
        } else {
          LongType currentElements = 1;
          for (int d = 0; d < rank; d++) currentElements *= shape[d];
          if (currentElements > 0 && maxElements > currentElements) {
            LongType scale = maxElements / currentElements;
            if (scale > 1) {
              maxShape[rank - 1] *= scale;
            }
          }
        }

        DSP_DIAG_SLOT(MEMORY, slotIdx, "max-allocating slot %d, current shape=[%lld,%lld,%lld,%lld], max shape=[%lld,%lld,%lld,%lld]",
                  slotIdx, shape[0], rank>1?shape[1]:0, rank>2?shape[2]:0, rank>3?shape[3]:0,
                  maxShape[0], maxShape.size()>1?maxShape[1]:0, maxShape.size()>2?maxShape[2]:0, maxShape.size()>3?maxShape[3]:0);

        NDArray* maxOut = nullptr;
        try {
          maxOut = DSP_NEW_ARRAY(order, maxShape, dt);
          if (!isBatchZeroActive() && !shouldPreserveWarmupOutputsDuringCapture()) {
            if (isBatchZeroRegistering() && DSP_BUF(maxOut) != nullptr) {
              registerBatchZeroBuffer(DSP_BUF(maxOut),
                                      maxOut->dataBuffer()->getLenInBytes(),
                                      slotIdx);
            }
            maxOut->nullify();
          }
        } catch (const std::exception& e) {
          DSP_DIAG_SLOT(MEMORY, stepIdx, "max-allocation FAILED at slot %d (%s): %s",
                    stepIdx, slot.opName.c_str(), e.what());
          maxOut = DSP_NEW_ARRAY(order, shape, dt);
          if (slot.needsZeroedOutput && !isBatchZeroActive() &&
              !shouldPreserveWarmupOutputsDuringCapture()) {
            if (isBatchZeroRegistering() && DSP_BUF(maxOut) != nullptr) {
              registerBatchZeroBuffer(DSP_BUF(maxOut),
                                      maxOut->dataBuffer()->getLenInBytes(),
                                      slotIdx);
            }
            maxOut->nullify();
          }
        }

        outputs[i] = maxOut;
        DSP_SLOT_WRITE(slotIdx, maxOut, "max-alloc-output");
        slotArrayCache_[slotIdx] = maxOut;
        maxAllocatedSlots_.insert(slotIdx);
        continue;
      }
      NDArray* cached2 = validateReusableSlotArray(slotIdx, slotArrayCache_[slotIdx],
                                                   "maxalloc-reuse-validate");
      if (cached2 != nullptr) {
        outputs[i] = cached2;
        DSP_SLOT_WRITE(slotIdx, cached2, "cached2-reuse");
        continue;
      }
    }

    NDArray* out = nullptr;
    try {
      out = DSP_NEW_ARRAY(order, shape, dt);
      // Register output for batch-zero if needed.
      // CRITICAL: Skip registration for view-capable slots. Their output buffers
      // share data with inputs (views), and zeroing them would corrupt the input
      // data. The input buffers must remain valid for downstream consumers.
      if (slot.needsZeroedOutput && !isBatchZeroActive() &&
          !shouldPreserveWarmupOutputsDuringCapture() &&
          !slot.isViewCapableOp) {
        if (isBatchZeroRegistering() && DSP_BUF(out) != nullptr) {
          registerBatchZeroBuffer(DSP_BUF(out),
                                  out->dataBuffer()->getLenInBytes(),
                                  slotIdx);
        }
        out->nullify();
      }
    } catch (const std::exception& e) {
      DSP_DIAG_SLOT(MEMORY, stepIdx, "output ALLOC EXCEPTION at slot %d (%s) output[%d]: %s",
                stepIdx, slot.opName.c_str(), i, e.what());
      return Status::KERNEL_FAILURE;
    }

    outputs[i] = out;
    DSP_SLOT_WRITE(slotIdx, out, "normal-alloc-output");
    slotArrayCache_[slotIdx] = out;
  }

  step4_execute:

  // Skip execution if all outputs are empty arrays — nothing to compute.
  // This prevents CUDA kernel launches on zero-element arrays which cause
  // illegal memory access errors (e.g., set_scalar on [1,3,3,0,64]).
  // EXCEPTION: view-capable ops (reshape, permute, etc.) should NOT be skipped
  // when their primary input is non-empty — the empty output is from stale shape
  // inference, and the op needs to execute to create a correct view.
  {
    bool allOutputsEmpty = true;
    for (int i = 0; i < numActualOutputs; i++) {
      if (outputs[i] != nullptr && !outputs[i]->isEmpty()) {
        allOutputsEmpty = false;
        break;
      }
    }
    if (allOutputsEmpty && numActualOutputs > 0) {
      // For view-capable ops, check if primary input is non-empty.
      // If so, the output shape is stale — re-derive from input and execute.
      bool isViewOp = slot.isViewCapableOp;
      bool primaryInputNonEmpty = (slot.numInputs > 0 && inputs[0] != nullptr &&
                                   !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0);
      if (isViewOp && primaryInputNonEmpty) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "view-capable slot %d (%s): outputs empty but input[0] non-empty "
                  "(length=%lld) — forcing execution instead of skip",
                  stepIdx, slot.opName.c_str(), inputs[0]->lengthOf());
        // Invalidate shape cache so next execution re-derives shapes
        slot.state_ = NativeSlot::SlotState::WARMUP;
      } else {
        DSP_DIAG_SLOT(EXECUTE, stepIdx, "skipping slot %d (%s): all %d outputs are empty",
                  stepIdx, slot.opName.c_str(), numActualOutputs);
        return Status::OK;
      }
    }
  }

  // ── Step 4: Configure context and execute ────────────────────────────────
  auto& ctx = *contextPool_[stepIdx];

  for (int i = 0; i < slot.numInputs; i++) {
    ctx.setInputArray(i, inputs[i]);
  }

  for (int i = 0; i < numActualOutputs; i++) {
    ctx.setOutputArray(i, outputs[i]);
  }

  if (slot.numIArgs > 0) ctx.setIArguments(slot.iArgs, slot.numIArgs);
  if (slot.numTArgs > 0) ctx.setTArguments(slot.tArgs, slot.numTArgs);
  if (slot.numBArgs > 0) ctx.setBArguments(slot.bArgs, slot.numBArgs);
  if (slot.numDArgs > 0) ctx.setDArguments(slot.dArgs, slot.numDArgs);
  ctx.getSArguments()->clear();
  if (slot.numSArgs > 0) {
    ctx.getSArguments()->insert(ctx.getSArguments()->end(), slot.sArgs, slot.sArgs + slot.numSArgs);
  }

  ctx.setShapeFunctionOverride(true);

  // Shape-key computation and shape functions may sync control tensors to host
  // in order to read value-dependent shape inputs. Re-sync those inputs before
  // execution so the kernel does not read stale device-side shape values.
  if (slot.isDataDependent || slot.outputShapeDependsOnInputValues) {
    for (int i = 0; i < slot.numInputs; i++) {
      NDArray* in = inputs[i];
      if (in == nullptr || in->isEmpty()) continue;
      auto* db = in->dataBuffer();
      if (db == nullptr || db->isClosed()) continue;

      const bool syncAllInputs = slot.isDataDependent;
      const bool isControlInput = isSmallIntegralControlArray(in);
      if (!syncAllInputs && !isControlInput) continue;

      // DIAGNOSTIC: trace small integral input values before sync for shape-control debugging
      bool traceThis = isControlInput && DSP_DIAG_ENABLED(SHAPE);
      if (traceThis && (stepIdx == 349 || stepIdx == 350 || stepIdx == 357 || stepIdx == 358)) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "PRE-SYNC-TRACE: slot %d (%s) input[%d] dtype=%d len=%lld pAct=%d sAct=%d "
            "values=[%s]",
            stepIdx, slot.opName.c_str(), i, (int)in->dataType(), (long long)in->lengthOf(),
            db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
            dspDumpHostDeviceValues(in, 16).c_str());
      }

      in->syncToDevice();
    }
  }

  // Log shapes for matmul gap slots to diagnose capture shape mismatches
  if (DSP_DIAG_ENABLED(EXECUTE) && normalizeOpName_slotexec(slot.opName) == "matmul") {
    std::string inStr, outStr, cachedStr;
    for (int i = 0; i < slot.numInputs; i++) {
      if (i > 0) inStr += ", ";
      inStr += inputs[i] ? ShapeUtils::shapeAsString(inputs[i]) : "null";
    }
    for (int i = 0; i < numActualOutputs; i++) {
      if (i > 0) outStr += ", ";
      outStr += outputs[i] ? ShapeUtils::shapeAsString(outputs[i]) : "null";
    }
    for (size_t i = 0; i < outputShapes.size(); i++) {
      if (i > 0) cachedStr += ", ";
      cachedStr += outputShapes[i] ? ShapeUtils::shapeAsString(outputShapes[i]) : "null";
    }
    DSP_DIAG(EXECUTE, "MATMUL PRE-EXEC: slot %d inputs=[%s] outputs=[%s] cachedShapes=[%s] "
              "cacheHit=%d frozen=%d execCount=%d",
              stepIdx, inStr.c_str(), outStr.c_str(), cachedStr.c_str(),
              (int)cacheHit, (int)shapesFrozen_, executeCount_);
    for (int i = 0; i < slot.numInputs; i++) {
      if (inputs[i] != nullptr) {
        auto* db = inputs[i]->dataBuffer();
        DSP_DIAG(EXECUTE, "MATMUL INPUT[%d]: slot %d srcIdx=%d addr=%p special=%p len=%lld "
                  "db=%p closed=%d const=%d",
                  i, stepIdx, slot.inputSourceIndices[i],
                  (void*)inputs[i], db ? db->special() : nullptr,
                  (long long)inputs[i]->lengthOf(),
                  (void*)db, db ? db->isClosed() : -1,
                  db ? (int)db->isConstant : -1);
      }
    }
  }

  // Set cublasLt epilogue if this matmul has fused bias+activation
  if (slot.ltEpilogueType > 0 && slot.ltEpilogueBiasSourceIdx != -1) {
    NDArray* biasArray = nullptr;
    int biasSrc = slot.ltEpilogueBiasSourceIdx;
    if (biasSrc >= 0) {
      biasArray = outputSlots_[biasSrc];
    } else {
      int extIdx = -(biasSrc + 1);
      if (extIdx < numExt) biasArray = externalArrays[extIdx];
    }
    if (biasArray != nullptr) {
#ifdef SD_CUDA
      biasArray->syncToDevice();
      MmulHelper::setLtEpilogue(slot.ltEpilogueType, biasArray->specialBuffer(),
                                 biasArray->lengthOf() * biasArray->sizeOfT());
#endif
      DSP_DIAG(FUSION, "set Lt epilogue type=%d bias=%p size=%lld for slot %d (%s)",
                slot.ltEpilogueType, biasArray->specialBuffer(),
                (long long)(biasArray->lengthOf() * biasArray->sizeOfT()),
                stepIdx, slot.opName.c_str());
    }
  }

  Status status;
  try {
    status = slot.op->execute(&ctx);
  } catch (const std::exception& e) {
    // Reshape failures and other exceptions land here — log with DSP_DIAG
    std::string inputShapes, outputShapesStr;
    for (int i = 0; i < slot.numInputs; i++) {
      if (i > 0) inputShapes += ", ";
      inputShapes += inputs[i] ? ShapeUtils::shapeAsString(inputs[i]) : "null";
    }
    for (int i = 0; i < numActualOutputs; i++) {
      if (i > 0) outputShapesStr += ", ";
      outputShapesStr += outputs[i] ? ShapeUtils::shapeAsString(outputs[i]) : "null";
    }
    std::string iArgsStr;
    for (int i = 0; i < slot.numIArgs; i++) {
      if (i > 0) iArgsStr += ",";
      iArgsStr += std::to_string(slot.iArgs[i]);
    }
    DSP_DIAG(EXECUTE, "SLOT EXEC EXCEPTION: slot %d (%s) exception='%s', inputs=[%s], outputs=[%s], "
              "iArgs=[%s], cacheHit=%d executeCount=%d shapesFrozen=%d",
              stepIdx, slot.opName.c_str(), e.what(),
              inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
              (int)cacheHit, executeCount_, (int)shapesFrozen_);
    // Propagate error detail to Java via errorReference
    {
      char errBuf[1024];
      snprintf(errBuf, sizeof(errBuf), "slot %d (%s) exec exception: %s | inputs=[%s] outputs=[%s] iArgs=[%s]",
               stepIdx, slot.opName.c_str(), e.what(),
               inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str());
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
    }
    // Clear epilogue state even on exception
#ifdef SD_CUDA
    if (slot.ltEpilogueType > 0) MmulHelper::clearLtEpilogue();
#endif
    return Status::KERNEL_FAILURE;
  }

  // Clear epilogue state after matmul execution
#ifdef SD_CUDA
  if (slot.ltEpilogueType > 0) MmulHelper::clearLtEpilogue();
#endif

  if (status != Status::OK) {
    std::string inputShapes, outputShapesStr;
    for (int i = 0; i < slot.numInputs; i++) {
      if (i > 0) inputShapes += ", ";
      inputShapes += inputs[i] ? ShapeUtils::shapeAsString(inputs[i]) : "null";
    }
    for (int i = 0; i < numActualOutputs; i++) {
      if (i > 0) outputShapesStr += ", ";
      outputShapesStr += outputs[i] ? ShapeUtils::shapeAsString(outputs[i]) : "null";
    }
    std::string iArgsStr;
    for (int i = 0; i < slot.numIArgs; i++) {
      if (i > 0) iArgsStr += ",";
      iArgsStr += std::to_string(slot.iArgs[i]);
    }
    // Propagate error detail to Java via errorReference
    {
      char errBuf[1024];
      snprintf(errBuf, sizeof(errBuf), "slot %d (%s) exec failed status=%d | inputs=[%s] outputs=[%s] iArgs=[%s]",
               stepIdx, slot.opName.c_str(), static_cast<int>(status),
               inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str());
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
    }
    DSP_DIAG(EXECUTE, "SLOT EXEC FAIL: slot %d (%s) status=%d, inputs=[%s], outputs=[%s], iArgs=[%s], "
              "cacheHit=%d executeCount=%d shapesFrozen=%d",
              stepIdx, slot.opName.c_str(), static_cast<int>(status),
              inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
              (int)cacheHit, executeCount_, (int)shapesFrozen_);
  }

  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      reconcileExecutedOutputActuality("normal-op-exec", stepIdx, slot, outputs[i]);
    }
  }

  if (status == Status::OK) {
    NDArray* tracedInput0 = slot.numInputs > 0 && !inputs.empty() ? inputs[0] : nullptr;
    NDArray* tracedOutput0 = numActualOutputs > 0 ? outputs[0] : nullptr;
    traceSmallControlSlotIO("normal-op-exec", stepIdx, slot,
                            tracedInput0, tracedOutput0,
                            executeCount_, planPhase_);
    traceSmallControlSlotTensors("normal-op-exec", stepIdx, slot,
                                 inputs.data(), slot.numInputs,
                                 outputs.data(), numActualOutputs,
                                 executeCount_, planPhase_);
  }

  // ── Step 5: View producer handling ────────────────────────────────────────
  {
    auto& ctxOutputs = ctx.fastpath_out();
    for (int i = 0; i < numActualOutputs && i < static_cast<int>(ctxOutputs.size()); i++) {
      int si = slot.outputSlotIndices[i];
      if (si < 0) continue;

      if (!viewProducerDetectionDone_) {
        if (ctxOutputs[i] != outputs[i]) {
          slotIsViewProducer_[si] = true;
          NDArray* oldCached = slotArrayCache_[si];
          // In Phase 2, slotArrayCache_ == outputSlots_. Deleting oldCached
          // would delete the slot's output array, leaving a dangling pointer.
          if (oldCached != nullptr && oldCached != ctxOutputs[i] && !tl_graphExecutionActive
              && slotArrayCache_ != outputSlots_) {
            if (!isSlotArrayShared(oldCached, si)) {
              delete oldCached;  // View wrapper only — no GPU memory freed
            }
          }
          DSP_SLOT_WRITE(si, ctxOutputs[i], "view-producer-detect");
          slotArrayCache_[si] = ctxOutputs[i];
        }
      } else if (slotIsViewProducer_[si]) {
        NDArray* oldCached = slotArrayCache_[si];
        // Same Phase 2 fix: don't delete the slot's own array
        if (oldCached != nullptr && oldCached != ctxOutputs[i] && !tl_graphExecutionActive
            && slotArrayCache_ != outputSlots_) {
          if (!isSlotArrayShared(oldCached, si)) {
            delete oldCached;  // View wrapper only — no GPU memory freed
          }
        }
        DSP_SLOT_WRITE(si, ctxOutputs[i], "view-producer-update");
        slotArrayCache_[si] = ctxOutputs[i];
      }
    }
  }

  // Promote slots to FROZEN state after successful execution under frozen shapes.
  // NOTE: This must run on the first frozen execution (executeCount_==0) too.
  // Previously gated on executeCount_ > 0, which prevented FROZEN promotion on
  // the first frozen execution. This forced ALL 2743 slots through normalExecution
  // on the SECOND frozen execution (step 3) instead of the frozen fast path,
  // because frozenContextReady() requires state_ >= FROZEN.
  if (shapesFrozen_ && executeCount_ >= 0 && status == Status::OK) {
    if (slot.state_ < NativeSlot::SlotState::FROZEN)
      slot.state_ = NativeSlot::SlotState::FROZEN;
  }

#ifdef SD_CUDA
  if (Environment::getInstance().tritonVerifyKernels() && status == Status::OK) {
    dspLogSlotOutput(stepIdx, slot.opName.c_str(), "OP_EXEC",
                        outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
  }
#endif

  return status;
}

// ─── Control Flow Slot Execution ─────────────────────────────────────────────
//
// Handles Switch/Merge/Enter/Exit/NextIteration/LoopCond ops.
// These ops manipulate the output slot array and dead-slot flags
// without calling any CUDA/C++ kernel.
//
// Returns:
//   Status::OK — normal completion
//   Status::MAYBE — NextIteration, caller should jump back to loopBackTarget
//   Negative int via KERNEL_FAILURE — error
//
// Note: This method is called from executeSegmentSlotBySlot (in _segments.cpp)
// which has been updated to handle backward jumps for loops.

// Helper: resolve an input for a CF slot
static NDArray* resolveCfInput(NativeSlot& slot, int inputIdx,
                               NDArray** outputSlots, int totalOutputSlots,
                               NDArray** externalInputs, int numExt) {
  if (inputIdx < 0 || inputIdx >= slot.numInputs) return nullptr;
  int srcIdx = slot.inputSourceIndices[inputIdx];
  if (srcIdx >= 0) {
    return (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
  } else {
    int extIdx = -(srcIdx + 1);
    return (extIdx < numExt) ? externalInputs[extIdx] : nullptr;
  }
}

}  // namespace graph
}  // namespace sd
