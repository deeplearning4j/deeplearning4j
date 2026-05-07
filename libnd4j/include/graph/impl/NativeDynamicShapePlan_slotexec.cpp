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
#include <graph/ModeContract.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/FusionPass.h>
#include <graph/LegacyOpTypeCodes.h>
#include <graph/PlanExecutionContext.h>
#include <ops/OpTraitTable.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_set>

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
  if (!slot.shapeCache.cachedOutputShapes.empty()) return;  // already populated
  if (slot.wiring.numOutputs <= 0) return;

  for (int o = 0; o < slot.wiring.numOutputs; o++) {
    int si = slot.wiring.outputSlotIndices[o];
    if (si >= 0 && si < totalOutputSlots && outputSlots[si] != nullptr) {
      auto* outArr = outputSlots[si];
      // Skip stale views whose shapeInfo was invalidated (e.g. view slot
      // whose source was closed between multi-frame executions).
      if (!outArr->hasValidShapeInfo()) continue;
      LongType* shapePtr = outArr->shapeInfo();
      auto cached = ConstantShapeHelper::getInstance().createFromExisting(
          const_cast<LongType*>(shapePtr));
      slot.shapeCache.cachedOutputShapes.push_back(cached);
    }
  }
  if (!slot.shapeCache.cachedOutputShapes.empty() && !slot.slotPhase.shapeCacheValid) {
    slot.slotPhase.markShapeCached();  // PRIMARY
    slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
  }
}

namespace {

// Delegate to shared utilities in DspAnalysisUtils.h
 int findProducingStepForOutputSlot(const NativeSlot* slots, int numSlots, int outputSlotIdx) {
  return dsp::findProducingStepForOutputSlot(slots, numSlots, outputSlotIdx);
}

static NDArray* resolveInputSourceArray(int srcIdx,
                                        NDArray** outputSlots,
                                        int totalOutputSlots,
                                        NDArray** externalArrays,
                                        int numExt) {
  return dsp::resolveInputSourceArray(srcIdx, outputSlots, totalOutputSlots, externalArrays, numExt);
}

static bool outputWrapperMatchesExpectedShape(NDArray* array, const LongType* expectedShape) {
  if (array == nullptr || expectedShape == nullptr) {
    return false;
  }

  bool validShapeInfo = false;
  try {
    validShapeInfo = array->hasValidShapeInfo();
  } catch (const std::exception& e) {
    DSP_DIAG(MEMORY,
             "SAFE_SHAPE_CHECK_EXCEPTION: array=%p exception=%s",
             (void*)array, e.what());
    return false;
  } catch (...) {
    DSP_DIAG(MEMORY,
             "SAFE_SHAPE_CHECK_EXCEPTION: array=%p unknown exception",
             (void*)array);
    return false;
  }

  if (!validShapeInfo) {
    return false;
  }

  const LongType* actualShape = array->shapeInfo();
  if (actualShape == nullptr) {
    return false;
  }

  return ArrayOptions::dataType(actualShape) == ArrayOptions::dataType(expectedShape) &&
         shape::shapeEquals(actualShape, expectedShape) &&
         shape::strideEquals(actualShape, expectedShape);
}

static bool safeHasValidShapeInfo(NDArray* array) {
  if (array == nullptr) return false;
  try {
    return array->hasValidShapeInfo();
  } catch (const std::exception& e) {
    DSP_DIAG(MEMORY,
             "SAFE_SHAPE_CHECK_EXCEPTION: array=%p exception=%s",
             (void*)array, e.what());
    return false;
  } catch (...) {
    DSP_DIAG(MEMORY,
             "SAFE_SHAPE_CHECK_EXCEPTION: array=%p unknown exception",
             (void*)array);
    return false;
  }
}

/**
 * Validate that an NDArray's _shapeInfoBuffer pointer is properly aligned.
 * Returns true if OK, false if corrupt. Emits a diagnostic with the tag
 * and slot so the first corruption site can be pinpointed.
 */
static bool validateSlotShapeBufferAlignment(
    NDArray* arr, int slotIdx, const char* tag, int execCount) {
  if (arr == nullptr) return true;
  auto* sib = arr->shapeInfoConstBuffer();
  if (sib == nullptr) return true;
  uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
  if (sibAddr % alignof(ConstantShapeBuffer) != 0) {
    sd_printf("CORRUPTION_PINPOINT: slot=%d tag=%s arr=%p "
              "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d\n",
              slotIdx, tag, (void*)arr, (void*)sib,
              static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
              execCount);
    fflush(stdout);
    return false;
  }
  return true;
}

/**
 * Scan ALL output slots for _shapeInfoBuffer alignment corruption.
 * Emits the FIRST corrupt slot found with the given checkpoint tag.
 * Call at strategic points to bracket the corruption window.
 */
static void scanAllSlotsForCorruption(
    NDArray** outputSlots, int totalOutputSlots,
    const char* checkpoint, int execCount) {
  for (int i = 0; i < totalOutputSlots; i++) {
    if (outputSlots[i] == nullptr) continue;
    auto* sib = outputSlots[i]->shapeInfoConstBuffer();
    if (sib == nullptr) continue;
    uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
    if (sibAddr % alignof(ConstantShapeBuffer) != 0) {
      sd_printf("CORRUPTION_SCAN_HIT: checkpoint=%s slot=%d arr=%p "
                "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d\n",
                checkpoint, i, (void*)outputSlots[i], (void*)sib,
                static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
                execCount);
      fflush(stdout);
      return;  // report first hit only
    }
  }
}

/**
 * Transitively check whether a slot has any isDynamicShape or
 * outputShapeDependsOnInputValues upstream. A simple one-hop check misses
 * chains like: rms_norm <- add <- concat where concat is dynamic but add
 * isn't directly flagged.
 *
 * IMPORTANT: external inputs (srcIdx < 0) are NOT treated as dynamic.
 * External inputs include both placeholders (whose shapes change between
 * prefill and decode) and constant weights/biases (whose shapes never
 * change). The shape key already captures external input shapes, so shape
 * drift from externals is handled by shape-key mismatch detection, not by
 * marking every transitive consumer as dynamic. Treating all externals as
 * dynamic caused nearly every slot to be isDynamicShape=true, preventing
 * frozen-state stabilization and causing massive allocation churn that led
 * to heap corruption (+1 byte _shapeInfoBuffer corruption).
 *
 * Uses iterative BFS with a fixed-size visited bitset (stack-allocated) to
 * avoid heap allocation on the hot path. Queue sized at 4096 to support
 * large VLM models where upstream chains can span 400+ slots.
 */
static bool hasTransitiveDynamicUpstream(
    const NativeSlot* slots, int numSlots, int startSlotIdx) {
  if (startSlotIdx < 0 || startSlotIdx >= numSlots) return false;

  const auto& startSlot = slots[startSlotIdx];
  if (startSlot.flags.isDynamicShape ||
      startSlot.flags.outputShapeDependsOnInputValues) {
    return true;
  }

  // BFS queue — stack-allocated. Must be large enough to hold all reachable
  // upstream slots. 256 was too small for VLM models with 420+ slots where
  // the upstream chain from a late slot (e.g. rms_norm at slot 420) back to
  // the SOURCE_PLACEHOLDER inputs exceeds 256 hops, causing the BFS to
  // silently truncate and return false when it should return true.
  constexpr int kMaxBfs = 4096;
  int queue[kMaxBfs];
  int qHead = 0, qTail = 0;

  // Visited bitset — covers up to 8192 slots (1 KB on stack)
  constexpr int kBitsetWords = 256; // 256 * 32 = 8192 bits
  uint32_t visited[kBitsetWords] = {};
  auto markVisited = [&](int idx) {
    if (idx >= 0 && idx < kBitsetWords * 32)
      visited[idx >> 5] |= (1u << (idx & 31));
  };
  auto isVisited = [&](int idx) -> bool {
    if (idx < 0 || idx >= kBitsetWords * 32) return false;
    return (visited[idx >> 5] & (1u << (idx & 31))) != 0;
  };

  // Seed with direct inputs of the start slot
  for (int i = 0; i < startSlot.wiring.numInputs; i++) {
    int srcIdx = startSlot.wiring.inputSourceIndices[i];
    if (srcIdx < 0) {
      // External placeholder inputs (e.g. KV cache) can change shape between
      // executions — treat them as dynamic. Constants/variables don't change.
      if (startSlot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
        return true;
      }
      continue;
    }
    if (srcIdx < numSlots && !isVisited(srcIdx)) {
      if (qTail >= kMaxBfs) {
        // BFS queue overflow — conservatively assume dynamic to avoid
        // incorrectly classifying as non-dynamic (which causes stale reuse).
        return true;
      }
      markVisited(srcIdx);
      queue[qTail++] = srcIdx;
    }
  }

  // BFS
  while (qHead < qTail) {
    int cur = queue[qHead++];
    const auto& curSlot = slots[cur];
    if (curSlot.flags.isDynamicShape ||
        curSlot.flags.outputShapeDependsOnInputValues) {
      return true;
    }
    for (int i = 0; i < curSlot.wiring.numInputs; i++) {
      int srcIdx = curSlot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        if (curSlot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
          return true;
        }
        continue;
      }
      if (srcIdx < numSlots && !isVisited(srcIdx)) {
        if (qTail >= kMaxBfs) {
          // BFS queue overflow — conservatively assume dynamic.
          return true;
        }
        markVisited(srcIdx);
        queue[qTail++] = srcIdx;
      }
    }
  }

  return false;
}

 bool isSmallIntegralControlArray(NDArray* arr) {
  if (arr == nullptr) return false;
  const auto dt = arr->dataType();
  if (dt != INT32 && dt != INT64 && dt != BOOL) return false;
  const auto len = arr->lengthOf();
  return len > 0 && len <= 32;
}

 void traceSmallControlSlotIO(const char* stage,
                                    int stepIdx,
                                    const NativeSlot& slot,
                                     NDArray* input0,
                                     NDArray* output0,
                                    int executeCount,
                                    const char* planPhaseName) {
  if (!DSP_DIAG_ENABLED(SHAPE)) return;
  if (!slot.flags.outputShapeDependsOnInputValues) return;
  if (!isSmallIntegralControlArray(input0) && !isSmallIntegralControlArray(output0)) return;

  const bool sharesBuffer =
      input0 != nullptr && output0 != nullptr &&
      input0->dataBuffer() != nullptr &&
      input0->dataBuffer() == output0->dataBuffer();

  DSP_DIAG(SHAPE,
           "CONTROL_SLOT_IO: stage=%s slot=%d (%s) state=%s exec=%d phase=%s "
           "input0=%s inputShape=%s output0=%s outputShape=%s sharesBuffer=%d",
           stage, stepIdx, slot.ident.opName.c_str(), slot.slotPhase.displayName(),
           executeCount, planPhaseName,
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
                                         const char* planPhaseName) {
  if (!DSP_DIAG_ENABLED(SHAPE)) return;
  if (!slot.flags.outputShapeDependsOnInputValues) return;

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
           "CONTROL_SLOT_TENSORS: stage=%s slot=%d (%s) state=%s exec=%d phase=%s inputs=[%s] outputs=[%s]",
           stage, stepIdx, slot.ident.opName.c_str(), slot.slotPhase.displayName(),
           executeCount, planPhaseName,
           inputStr.empty() ? "none" : inputStr.c_str(),
           outputStr.empty() ? "none" : outputStr.c_str());
}

// reconcileExecutedOutputActuality moved to platform dispatch:
// CUDA impl in _slotexec_cuda.cu, CPU stub in _cuda_stubs.cpp.
// Call sites now use platformReconcileOutputActuality() directly.

static bool slotHasOnlyPlanInternalControlInputs(const NativeSlot& slot,
                                                 NDArray** outputSlots,
                                                 int totalOutputSlots,
                                                 NDArray** externalArrays,
                                                 int numExt) {
  bool sawControlInput = false;
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    NDArray* arr = resolveInputSourceArray(slot.wiring.inputSourceIndices[i], outputSlots,
                                           totalOutputSlots, externalArrays, numExt);
    if (!isSmallIntegralControlArray(arr)) continue;
    sawControlInput = true;
    if (slot.wiring.inputSourceIndices[i] < 0) {
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
  if (slot.flags.outputShapeDependsOnInputValues &&
      slotHasOnlyPlanInternalControlInputs(slot, outputSlots, totalOutputSlots,
                                           externalArrays, numExt)) {
    return true;
  }

  for (int i = 0; i < slot.wiring.numInputs; i++) {
    const int srcIdx = slot.wiring.inputSourceIndices[i];
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
  if (planPhase >= PlanPhase::REPLAYING) return false;

  if (slot.flags.outputShapeDependsOnInputValues &&
      slotHasOnlyPlanInternalControlInputs(slot, outputSlots, totalOutputSlots,
                                           externalArrays, numExt)) {
    return true;
  }

  std::unordered_set<int> visitedSteps;
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    const int srcIdx = slot.wiring.inputSourceIndices[i];
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
    out += slots[producerStep].ident.opName;
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
  if (producer.wiring.numInputs > 0) {
    out += " <= {";
    for (int i = 0; i < producer.wiring.numInputs; i++) {
      if (i > 0) out += "; ";
      appendUpstreamControlTrace(out, producer.wiring.inputSourceIndices[i], depth - 1,
                                 slots, numSlots, outputSlots, totalOutputSlots,
                                 externalArrays, numExt, externalInputNames,
                                 seenSlotSteps);
    }
    out += "}";
  }
  seenSlotSteps.pop_back();
}

/**
 * Build shape info for a permute view: delegates to ShapeUtils::evalPermutedViewShapeInfo.
 * Unlike calculateOutputShape (which sets contiguous strides), the ShapeUtils function
 * preserves the input's actual stride pattern so the view reads data correctly.
 *
 * The permutation can come from two sources (matching the permute op itself):
 *   1. iArgs on the slot (when permute was created with explicit int dimensions)
 *   2. A second input array (when ONNX Transpose provides perm as a constant tensor)
 *
 * Returns a ConstantShapeHelper-managed pointer (not owned by caller).
 */
static const LongType* buildPermutedViewShapeInfo(const NDArray* input, const NativeSlot& slot,
                                                    NDArray** allInputs, int numInputs) {
  // Build the permutation array from iArgs, normalizing negative indices
  int rank = shape::rank(input->shapeInfo());
  if (rank <= 0) return nullptr;

  std::vector<int> permVec;
  if (slot.args.numIArgs > 0) {
    // Source 1: permutation stored as iArgs
    for (int i = 0; i < slot.args.numIArgs; i++) {
      int dim = static_cast<int>(slot.args.iArgs[i]);
      if (dim < 0) dim += rank;  // wrap negative
      permVec.push_back(dim);
    }
  } else if (numInputs > 1 && allInputs != nullptr && allInputs[1] != nullptr &&
             !allInputs[1]->isEmpty() && allInputs[1]->lengthOf() > 0) {
    // Source 2: permutation stored as second input array (ONNX Transpose import path)
    NDArray* permArr = allInputs[1];
    permArr->syncToHost();
    auto len = permArr->lengthOf();
    for (LongType i = 0; i < len; i++) {
      int dim = static_cast<int>(permArr->e<LongType>(i));
      if (dim < 0) dim += rank;  // wrap negative
      permVec.push_back(dim);
    }
  } else {
    // Source 3: no explicit permutation — default to plain transpose (reverse all dimensions).
    // This matches the permute op's own fallback: when width()==1 && numIArgs==0, it calls
    // x->transpose(). We must use the same semantics here so the view strides are correct.
    for (int i = rank - 1; i >= 0; i--) {
      permVec.push_back(i);
    }
  }

  return ShapeUtils::evalPermutedViewShapeInfo(input, permVec.data(), static_cast<int>(permVec.size()));
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
  if (slot.args.numIArgs < 5 || numInputs < 4) return -1;

  int beginMask = static_cast<int>(slot.args.iArgs[0]);
  // int ellipsisMask = static_cast<int>(slot.args.iArgs[1]); // not needed for offset
  // int endMask = static_cast<int>(slot.args.iArgs[2]);       // not needed for offset
  int newAxisMask = static_cast<int>(slot.args.iArgs[3]);
  int shrinkAxisMask = static_cast<int>(slot.args.iArgs[4]);

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
 * outputSlots_, slotIsViewProducer_, and context — this helper only
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
               stepIdx, slot.ident.opName.c_str(), input0->ordering(),
               shape::strideDescendingCAscendingF(const_cast<LongType*>(input0->shapeInfo())) ? 1 : 0,
               ShapeUtils::shapeAsString(input0).c_str(),
               ShapeUtils::shapeAsString(outShapeInfo).c_str(),
               static_cast<long long>(inLen), static_cast<long long>(outLen));
    }
    return VIEW_NOT_POSSIBLE;
  }

  // For permute: use permuted strides from input, not contiguous strides
  const LongType* effectiveShapeInfo = outShapeInfo;
  bool isPermute = (normalizeOpName_slotexec(slot.ident.opName) == "permute");
  if (isPermute) {
    const LongType* permSI = buildPermutedViewShapeInfo(input0, slot, allInputs, numInputs);
    if (permSI != nullptr) {
      effectiveShapeInfo = permSI;
    }
  }

  // Compute view offset for strided_slice
  LongType viewOffset = 0;
  if (slot.ident.op && slot.ident.op->getOpDescriptor() &&
      slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SLICE)) {
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
      slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr &&
      slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SLICE);
  bool elementCountCompatible =
      allowSubsetView ? (outLen > 0 && outLen <= inLen) : (outLen > 0 && outLen == inLen);
  bool fitsBackingBuffer =
      absoluteOffset >= 0 && absoluteOffset <= sourceBufferElems &&
      outLen <= (sourceBufferElems - absoluteOffset);

  if (elementCountCompatible && fitsBackingBuffer) {
    *outView = new NDArray(input0->dataBuffer(),
                           const_cast<LongType*>(effectiveShapeInfo),
                           LaunchContext::defaultContext(),
                           absoluteOffset);
    *outViewOffset = absoluteOffset;
    if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
      DSP_DIAG(SHAPE,
               "VIEW_TRACE_OK: slot %d (%s) zero-copy view created "
               "inShape=%s outShape=%s inLen=%lld outLen=%lld offset=%lld",
               stepIdx, slot.ident.opName.c_str(),
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
               stepIdx, slot.ident.opName.c_str(),
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
             stepIdx, slot.ident.opName.c_str(),
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

// ─── Stale view-producer wrapper refresh ────────────────────────────────────
// Walk the slots in a segment range and, for each view-producer slot whose
// cached output NDArray has a stale (freed/replaced) DataBuffer, re-create the
// view wrapper pointing at the current input's DataBuffer. This is the common
// case when a view op (squeeze/expand_dims/reshape/permute/strided_slice)
// directly wraps a placeholder: SameDiff replaces the placeholder's DataBuffer
// between calls, which invalidates the view wrapper held in outputSlots_.
//
// Unlike the slot-by-slot fallback, this does NOT invalidate the captured
// CUDA graph or reset executionCount. It only updates NDArray wrappers
// and signals argTableStable = false so refreshArgTablesForReplay picks up
// the new device-side addresses on the next replay.
//
// Iteration order [startSlot..endSlot] ensures a parent view-producer slot
// is refreshed before a child view that reads from it (e.g., squeeze(squeeze(x))).
int NativeDynamicShapePlan::refreshStaleViewWrappersInSegment(
    GraphSegment& seg, NDArray** externalArrays, int numExt) {
  if (slotIsViewProducer_ == nullptr || outputSlots_ == nullptr) {
    return 0;
  }

  int refreshedCount = 0;

  for (int stepIdx = seg.def.startSlot;
       stepIdx <= seg.def.endSlot && stepIdx < numSlots_; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    if (!slot.flags.isViewCapableOp) {
      continue;
    }
    if (slot.wiring.numOutputs <= 0) continue;

    static thread_local std::vector<NDArray*> viewInputs;
    viewInputs.resize(slot.wiring.numInputs);
    NDArray* input0 = nullptr;
    for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
      int srcIdx = slot.wiring.inputSourceIndices[ii];
      NDArray* resolved = nullptr;
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        resolved = outputSlots_[srcIdx];
      } else if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt) resolved = externalArrays[extIdx];
      }
      viewInputs[ii] = resolved;
      if (ii == 0) input0 = resolved;
    }

    for (int outIdx = 0; outIdx < slot.wiring.numOutputs; outIdx++) {
      int outSi = slot.wiring.outputSlotIndices[outIdx];
      if (outSi < 0 || outSi >= totalOutputSlots_) continue;
      if (!slotIsViewProducer_[outSi]) continue;

      NDArray* cached = outputSlots_[outSi];
      bool cachedValid = safeHasValidShapeInfo(cached);
      if (cached == nullptr) continue;
      if (cachedValid && cached->isEmpty()) continue;

      if (input0 == nullptr || input0->dataBuffer() == nullptr
          || !input0->dataBuffer()->isValid()) {
        DSP_DIAG_SLOT(MEMORY, stepIdx,
            "REFRESH_VIEW_FAIL: step %d (%s) outputSlot=%d input0 unresolved or invalid",
            stepIdx, slot.ident.opName.c_str(), outSi);
        return -1;
      }

      if (!slot.shapeCacheValid() ||
          outIdx >= static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) ||
          slot.shapeCache.cachedOutputShapes[outIdx] == nullptr) {
        DSP_DIAG_SLOT(MEMORY, stepIdx,
            "REFRESH_VIEW_FAIL: step %d (%s) outputSlot=%d has no cached output shape",
            stepIdx, slot.ident.opName.c_str(), outSi);
        return -1;
      }

      NDArray* newView = nullptr;
      LongType viewOffset = 0;
      ViewCreateResult vcr = tryCreateViewForSlot(
          stepIdx, slot, input0, slot.shapeCache.cachedOutputShapes[outIdx],
          viewInputs.data(), slot.wiring.numInputs,
          &newView, &viewOffset);

      if (vcr != VIEW_CREATED || newView == nullptr) {
        DSP_DIAG_SLOT(MEMORY, stepIdx,
            "REFRESH_VIEW_FAIL: step %d (%s) outputSlot=%d tryCreateViewForSlot returned %d",
            stepIdx, slot.ident.opName.c_str(), outSi, static_cast<int>(vcr));
        if (newView != nullptr) delete newView;
        return -1;
      }

      auto* cachedDb = cachedValid ? cached->dataBuffer() : nullptr;
      if (cachedValid && cachedDb != nullptr && cachedDb->isValid() &&
          safeHasValidShapeInfo(newView) &&
          newView->dataBuffer() == cachedDb &&
          newView->offset() == cached->offset() &&
          newView->dataType() == cached->dataType() &&
          shape::shapeEquals(newView->shapeInfo(), cached->shapeInfo()) &&
          shape::strideEquals(newView->shapeInfo(), cached->shapeInfo())) {
        delete newView;
        continue;
      }

      // Remove old cached array before installing the new view.
      // The old code (pre-refactor) did: erase(cached); delete cached;
      // Missing this causes a leak and leaves a stale NDArray on the heap
      // whose _shapeInfoBuffer field (offset 16) gets overwritten by malloc
      // metadata when the heap reuses the memory — manifesting as the +1
      // byte _shapeInfoBuffer corruption.
      outputSlots_[outSi] = nullptr;
      planOwnedArrays_.erase(cached);
      if (cachedValid) {
        delete cached;
      }
      // If !cachedValid, cached is already corrupt — don't delete it
      // (the destructor would SIGSEGV reading corrupt _shapeInfo).

      // Scan immediately after delete — if the freed NDArray's heap metadata
      // overwrites an adjacent NDArray's _shapeInfoBuffer, we catch it here.
      if (sd::Environment::getInstance().isDebug()) {
        char vdLabel[256];
        snprintf(vdLabel, sizeof(vdLabel),
                 "AFTER_viewRefresh_delete_slot%d_%s", outSi, slot.ident.opName.c_str());
        scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
            vdLabel, executeCount_);
      }

      writeOutputSlot(outSi, newView, "ff-view-install");
      // writeOutputSlot already handles slot assignment and ownership tracking

      // Validate newView itself and neighboring slots for corruption
      // immediately after view swap — bracket the exact mutation.
      if (sd::Environment::getInstance().isDebug()) {
        if (!validateSlotShapeBufferAlignment(newView, outSi,
                "AFTER_ff-view-swap-newView", executeCount_)) {
          return -1;
        }
        // Check neighbors: if the delete of 'cached' caused heap metadata to
        // overwrite an adjacent NDArray, we catch it here.
        if (outSi > 0 && outputSlots_[outSi - 1] != nullptr) {
          validateSlotShapeBufferAlignment(outputSlots_[outSi - 1], outSi - 1,
              "AFTER_ff-view-swap-prevSlot", executeCount_);
        }
        if (outSi + 1 < totalOutputSlots_ && outputSlots_[outSi + 1] != nullptr) {
          validateSlotShapeBufferAlignment(outputSlots_[outSi + 1], outSi + 1,
              "AFTER_ff-view-swap-nextSlot", executeCount_);
        }
      }

      if (contextPool_ != nullptr && contextPool_[stepIdx] != nullptr) {
        auto& ctx = *contextPool_[stepIdx];
        ctx.setOutputArray(outIdx, newView);
        for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
          ctx.setInputArray(ii, viewInputs[ii]);
        }
      }

      // Keep ownership metadata pointing at the new DataBuffer so downstream
      // scans (validateLifecycleForPhase etc.) see the refreshed buffer identity.
      if (slotOwnership_ != nullptr) {
        slotOwnership_[outSi].dataBuffer = newView->dataBuffer();
      }

      slot.bumpGeneration();
      dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
      if (!seg.exec.needsArgRefresh()) {
        seg.exec.bumpArgGeneration();
        seg.exec.argTableStable = false;
      }

      DSP_DIAG_SLOT(MEMORY, stepIdx,
          "REFRESH_VIEW_OK: step %d (%s) outputSlot=%d view refreshed arr=%p db=%p offset=%lld",
          stepIdx, slot.ident.opName.c_str(), outSi,
          static_cast<void*>(newView),
          static_cast<void*>(newView->dataBuffer()),
          static_cast<long long>(viewOffset));

      refreshedCount++;
    }
  }

  return refreshedCount;
}

// ─── Frozen constant detection ──────────────────────────────────────────────
// After the warmup execution (executeCount_ just went from 0 to 1), identify
// slots whose output never changes between decode steps. These slots are
// skipped entirely during subsequent executions (including graph capture),
// removing their kernels, memsets, and memcpys from the captured graph.

void NativeDynamicShapePlan::detectFrozenConstants() {
  if (planLifecycle_.isSlotBySlot() || executeCount_ != 1 || frozenConstantDetectionDone_) return;
  frozenConstantDetectionDone_ = true;

  // Ops whose output depends ONLY on input shapes, not input values.
  // When shapes are frozen, these produce identical output every step.
  // Classification now comes from OpDescriptor traits (OP_TRAIT_SHAPE_ONLY_OUTPUT).

  std::vector<bool> dependsOnExternal(totalOutputSlots_, false);
  std::vector<bool> isValueIndependentSlot(numSlots_, false);

  // Step 1: Identify output slots NOT produced by any op in the plan.
  // These are "Java-managed" slots — populated externally by Java before each
  // execution (constants, variables, placeholders). Any op reading from such a
  // slot must be treated as external-dependent because Java may change the data.
  //
  // This fixes the KV cache bug: static KV buffers are associated as variables
  // (srcIdx >= 0, SOURCE_VARIABLE) and mapped to output slots that no plan op
  // writes to. Without this, KV concat ops appeared to have only internal inputs
  // and were incorrectly frozen, leaving the KV cache unupdated.
  std::vector<bool> producedByPlanOp(totalOutputSlots_, false);
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    for (int o = 0; o < sl.wiring.numOutputs; o++) {
      int si = sl.wiring.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_) {
        producedByPlanOp[si] = true;
      }
    }
  }

  // Mark all Java-managed (non-op-produced) output slots as external-dependent.
  // Constants with SOURCE_CONSTANT are truly immutable, but output slot indices
  // are not external input indices: recurrent state buffers such as KV/GDN/conv
  // state can be wired as positive SOURCE_VARIABLE slot references that no plan
  // op writes. If these slots are treated as constants, downstream update ops are
  // frozen and generation reuses stale state after the freeze transition.
  for (int si = 0; si < totalOutputSlots_; si++) {
    if (!producedByPlanOp[si]) {
      dependsOnExternal[si] = true;
    }
  }

  // Propagate external dependency through the graph (topological order).
  // Value-independent ops do NOT propagate dependency — their outputs
  // are constant when shapes are frozen.
  int shapeOnlyCount = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];

    // Check if this op is value-independent via trait
    bool isShapeOnly = (sl.ident.op && sl.ident.op->getOpDescriptor() &&
                        sl.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT));
    if (isShapeOnly) {
      isValueIndependentSlot[s] = true;
      shapeOnlyCount++;
      continue;
    }

    bool anyInputDependsOnExternal = false;
    for (int i = 0; i < sl.wiring.numInputs; i++) {
      int srcIdx = sl.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        // External input: only propagate dependency if it's a VARIABLE external
        // input (changes per step). Constants are immutable — reading from a
        // constant does NOT make this op's output externally-dependent.
        int extIdx = -(srcIdx + 1);
        if (extIdx < (int)externalInputIsVariable_.size() && externalInputIsVariable_[extIdx]) {
          anyInputDependsOnExternal = true;
          break;
        }
        // If externalInputIsVariable_ is not yet populated (first execute before
        // markExternalInputVariable calls), fall back to conservative behavior.
        if (externalInputIsVariable_.empty()) {
          anyInputDependsOnExternal = true;
          break;
        }
        continue;
      }
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && dependsOnExternal[srcIdx]) {
        anyInputDependsOnExternal = true;
        break;
      }
    }
    if (anyInputDependsOnExternal) {
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
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
    for (int o = 0; o < sl.wiring.numOutputs; o++) {
      int si = sl.wiring.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && dependsOnExternal[si]) {
        allOutputsConstant = false;
        break;
      }
    }
    // Freeze if ALL outputs are constant AND the op won't produce different results.
    //
    // Data-dependent ops (concat with axis-in-last-arr, etc.) read input VALUES
    // to determine their output. They cannot be frozen if their inputs come from
    // external/variable sources. BUT if ALL of their inputs are themselves frozen
    // constants (no external dependency), then the values they read are the same
    // every step — so the op IS safe to freeze.
    //
    // This handles the SmolDocling case: 552 concat ops build shape vectors like
    // [batch,1,heads,dim] from frozen scalar constants. The concat is DATADEP
    // (conservatively, for axis-in-last-arr mode) but all inputs are frozen.
    bool effectivelyDataDependent = sl.flags.isDataDependent;
    if (effectivelyDataDependent && allOutputsConstant) {
      // Check if ALL inputs are from frozen-constant slots or frozen externals.
      // If so, the data the op depends on is itself frozen → safe to freeze.
      bool allInputsFrozen = true;
      for (int inp = 0; inp < sl.wiring.numInputs; inp++) {
        int srcIdx = sl.wiring.inputSourceIndices[inp];
        if (srcIdx < 0) {
          // External input — check if it's a non-variable (weight/constant).
          int extIdx = -(srcIdx + 1);
          if (extIdx >= 0 && extIdx < static_cast<int>(externalInputIsVariable_.size()) &&
              externalInputIsVariable_[extIdx]) {
            allInputsFrozen = false;
            break;
          }
          // Non-variable external (weight) → frozen by definition
        } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          // Internal slot output — check if producing slot is frozen
          if (dependsOnExternal[srcIdx]) {
            allInputsFrozen = false;
            break;
          }
        }
      }
      if (allInputsFrozen) {
        effectivelyDataDependent = false;
        DSP_DIAG_SLOT(SHAPE, s, "DATADEP op '%s' has all-frozen inputs → safe to freeze as constant",
                       sl.ident.opName.c_str());
      }
    }
    if (allOutputsConstant && !effectivelyDataDependent && !sl.flags.isDynamicShape) {
      sl.slotPhase.seal(true);  // PRIMARY: frozen constant
      sl.state_ = NativeSlot::SlotState::FROZEN_CONSTANT;  // legacy sync
      frozenConstCount++;
      if (isValueIndependentSlot[s]) valueIndepCount++;
    }
  }
  // Collect frozen output slot indices for quick lookup
  std::unordered_set<int> frozenOutputSlots;
  for (int s = 0; s < numSlots_; s++) {
    if (slots_[s].frozenConstantSlot()) {
      for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
        frozenOutputSlots.insert(slots_[s].wiring.outputSlotIndices[o]);
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
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
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
        sl.slotPhase.unseal(); sl.slotPhase.shapeCacheValid = true;  // PRIMARY: demote to SHAPE_CACHED
        sl.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
        // Remove from frozenOutputSlots
        for (int o = 0; o < sl.wiring.numOutputs; o++) {
          frozenOutputSlots.erase(sl.wiring.outputSlotIndices[o]);
        }
        viewAliasUnfrozen++;
        frozenConstCount--;
      }
    }
  }

  // Un-freeze any frozen constant whose output feeds (directly or transitively)
  // a downstream op with outputShapeDependsOnInputValues (e.g., reshape_no_copy
  // where input[1] is the shape tensor). When the plan is destroyed and recreated,
  // the frozen slot's DataBuffer is freed but the CUDA host allocator may reuse
  // its memory before the next plan's shape inference reads it — causing garbage
  // shape values. We also transitively un-freeze any frozen upstream of an
  // un-frozen slot, because an un-frozen slot's execution depends on its inputs
  // being alive and up-to-date.
  int valueDepUnfrozen = 0;
  int valueDepStableKept = 0;
  {
    // Build a map: output-slot-index → op-slot index that produces it
    std::unordered_map<int, int> outputSlotToOpSlot;
    for (int s = 0; s < numSlots_; s++) {
      for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
        int si = slots_[s].wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          outputSlotToOpSlot[si] = s;
        }
      }
    }
    // Seed: any frozen slot whose output feeds a value-dependent op.
    // OPTIMIZATION: If the value-dep op's ALL inputs come from frozen plan-internal
    // slots (no external inputs), its shape-tensor values are stable and it does not
    // need its upstream un-frozen. This keeps the entire shape-control ladder frozen
    // during decode (shape_of → concat → reshape_no_copy chains where all shapes
    // are constant). Only un-freeze when the value-dep op has external inputs whose
    // values may change between steps.
    std::unordered_set<int> toUnfreeze;
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      if (!sl.flags.outputShapeDependsOnInputValues) continue;

      // Check if ALL inputs to this value-dep op come from frozen plan-internal slots.
      // If so, the shape tensor values are stable — skip un-freezing upstream.
      bool hasUnstableInput = false;
      for (int i = 0; i < sl.wiring.numInputs; i++) {
        int srcIdx = sl.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          // External input — may change between steps, so upstream must be un-frozen
          hasUnstableInput = true;
          break;
        }
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          auto it = outputSlotToOpSlot.find(srcIdx);
          if (it != outputSlotToOpSlot.end() && !slots_[it->second].frozenConstantSlot()) {
            // Input comes from a non-frozen plan-internal slot — unstable
            hasUnstableInput = true;
            break;
          }
          // srcIdx might not map to any op (Java-managed slot) — treat as unstable
          if (it == outputSlotToOpSlot.end() && !producedByPlanOp[srcIdx]) {
            hasUnstableInput = true;
            break;
          }
        }
      }

      if (!hasUnstableInput) {
        // All inputs are frozen plan-internal — this value-dep op is stable.
        // Mark it frozen too (its outputs are constant when shapes are frozen).
        if (!sl.flags.isDataDependent && !sl.flags.isDynamicShape) {
          sl.slotPhase.seal(true);  // PRIMARY: frozen constant
          sl.state_ = NativeSlot::SlotState::FROZEN_CONSTANT;  // legacy sync
          for (int o = 0; o < sl.wiring.numOutputs; o++) {
            frozenOutputSlots.insert(sl.wiring.outputSlotIndices[o]);
          }
          frozenConstCount++;
          valueDepStableKept++;
          continue;
        }
        // isDataDependent=true (e.g. reshape_no_copy): this op WILL execute every step
        // because it can't be frozen, but its frozen upstream inputs must also execute
        // to keep producing valid data. Un-freeze them.
        for (int i = 0; i < sl.wiring.numInputs; i++) {
          int srcIdx = sl.wiring.inputSourceIndices[i];
          if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
            auto it = outputSlotToOpSlot.find(srcIdx);
            if (it != outputSlotToOpSlot.end() && slots_[it->second].frozenConstantSlot()) {
              toUnfreeze.insert(it->second);
            }
          }
        }
        continue;
      }

      // Has unstable input — un-freeze frozen upstream as before
      for (int i = 0; i < sl.wiring.numInputs; i++) {
        int srcIdx = sl.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          auto it = outputSlotToOpSlot.find(srcIdx);
          if (it != outputSlotToOpSlot.end() && slots_[it->second].frozenConstantSlot()) {
            toUnfreeze.insert(it->second);
          }
        }
      }
    }
    // Transitively un-freeze: any frozen upstream of an already-to-unfreeze slot
    std::vector<int> worklist(toUnfreeze.begin(), toUnfreeze.end());
    for (size_t wi = 0; wi < worklist.size(); wi++) {
      auto& sl = slots_[worklist[wi]];
      for (int i = 0; i < sl.wiring.numInputs; i++) {
        int srcIdx = sl.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          auto it = outputSlotToOpSlot.find(srcIdx);
          if (it != outputSlotToOpSlot.end() && slots_[it->second].frozenConstantSlot()
              && toUnfreeze.find(it->second) == toUnfreeze.end()) {
            toUnfreeze.insert(it->second);
            worklist.push_back(it->second);
          }
        }
      }
    }
    for (int s : toUnfreeze) {
      slots_[s].slotPhase.unseal(); slots_[s].slotPhase.shapeCacheValid = true;  // PRIMARY: demote
      slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
      for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
        frozenOutputSlots.erase(slots_[s].wiring.outputSlotIndices[o]);
      }
      valueDepUnfrozen++;
      frozenConstCount--;
    }
  }

  // Disable in-place fusion for any op that would overwrite a protected output buffer.
  // In-place fusion writes the op's output directly into its input buffer — if another
  // slot later reads that buffer for shape inference (outputShapeDependsOnInputValues)
  // or another downstream consumer, it gets garbage.
  //
  // Protected output slots:
  // 1. Frozen constant slots (must remain stable for frozen skip)
  // 2. Slots that serve as inputs to value-dependent ops (shape data must remain valid
  //    until the value-dep op executes — e.g., concat→reshape_no_copy chain)
  // 3. Slots consumed by multiple downstream ops (in-place overwrites corrupt the buffer
  //    for all other consumers that haven't read it yet)
  std::unordered_set<int> protectedSlots(frozenOutputSlots.begin(), frozenOutputSlots.end());

  // Count consumers per output slot to identify multi-consumer slots
  std::vector<int> slotConsumerCount(totalOutputSlots_, 0);
  for (int s = 0; s < numSlots_; s++) {
    for (int i = 0; i < slots_[s].wiring.numInputs; i++) {
      int srcIdx = slots_[s].wiring.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        slotConsumerCount[srcIdx]++;
      }
    }
  }

  for (int s = 0; s < numSlots_; s++) {
    // Protect inputs to value-dependent ops
    if (slots_[s].flags.outputShapeDependsOnInputValues) {
      for (int i = 0; i < slots_[s].wiring.numInputs; i++) {
        int srcIdx = slots_[s].wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          protectedSlots.insert(srcIdx);
        }
      }
    }
  }
  // Multi-consumer in-place protection: conservatively disable in-place for ANY
  // multi-consumer slot. When frozen-constant skip is active (executeCount_ >= 2),
  // allowing in-place writes to shared buffers corrupts data that frozen ops reuse
  // on subsequent iterations. The "last consumer" optimization is unsafe because
  // frozen ops don't re-execute — they rely on stable buffer contents from prior runs.

  int disabledInPlace = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    if (sl.flags.inPlaceFused && sl.flags.inPlaceFusedInputIdx >= 0 &&
        sl.flags.inPlaceFusedInputIdx < sl.wiring.numInputs) {
      int srcSlot = sl.wiring.inputSourceIndices[sl.flags.inPlaceFusedInputIdx];
      if (srcSlot < 0) continue;

      // Always protect frozen constant slots and value-dep op inputs
      if (protectedSlots.count(srcSlot)) {
        sl.flags.inPlaceFused = false;
        sl.flags.inPlaceFusedInputIdx = -1;
        disabledInPlace++;
        continue;
      }

      // Protect ALL multi-consumer slots — in-place overwrites corrupt the buffer
      // for any other consumer, and frozen-skip relies on stable buffer contents
      if (slotConsumerCount[srcSlot] > 1) {
        sl.flags.inPlaceFused = false;
        sl.flags.inPlaceFusedInputIdx = -1;
        disabledInPlace++;
      }
    }
  }

  // Count Java-managed slots for diagnostics
  int javaManagedCount = 0;
  for (int si = 0; si < totalOutputSlots_; si++) {
    if (!producedByPlanOp[si]) javaManagedCount++;
  }

  DSP_DIAG(SHAPE, "frozen constant detection: %d/%d slots are frozen constants "
            "(%d shape-only-trait, %d value-independent, %d value-dep-stable-kept, "
            "%d in-place disabled, %d view-alias unfrozen, "
            "%d value-dep unfrozen, %d java-managed output slots)",
            frozenConstCount, numSlots_, shapeOnlyCount, valueIndepCount, valueDepStableKept,
            disabledInPlace, viewAliasUnfrozen, valueDepUnfrozen, javaManagedCount);

  // Populate execution context with frozen constant stats so downstream
  // diagnostics (logExecutionSummary, dumpFlowLog on failure) show what happened.
  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  if (execCtx != nullptr) {
    execCtx->frozenConstCount = frozenConstCount;
    execCtx->shapeOnlyTraitCount = shapeOnlyCount;
    execCtx->valueIndepCount = valueIndepCount;
    execCtx->viewAliasUnfrozen = viewAliasUnfrozen;
    execCtx->javaManagedSlots = javaManagedCount;
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::FROZEN_CONST_DETECT,
                         frozenConstCount, numSlots_);
  }

  // Log external-dependency summary for frozen constant debugging.
  if (DSP_DIAG_ENABLED(SHAPE)) {
    int extDepCount = 0;
    for (int si = 0; si < totalOutputSlots_; si++) {
      if (dependsOnExternal[si]) extDepCount++;
    }
    DSP_DIAG(SHAPE, "frozen constant detection: %d/%d output slots depend on external input "
             "(%d from java-managed slots)",
             extDepCount, totalOutputSlots_, javaManagedCount);

    // At FULL level, log per-slot details for the configured trace slot (if any).
    int ts = DspDiagnostics::getInstance().traceSlot();
    if (ts >= 0 && ts < totalOutputSlots_) {
      bool depExt = dependsOnExternal[ts];
      int producingStep = -1;
      bool isFrozenConst = false, isShapeStatic = false;
      bool found = false;
      for (int s = 0; s < numSlots_ && !found; s++) {
        for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
          if (slots_[s].wiring.outputSlotIndices[o] == ts) {
            producingStep = s;
            isFrozenConst = slots_[s].frozenConstantSlot();
            isShapeStatic = slots_[s].shapeCache.shapeStatic;
            found = true;
            break;
          }
        }
      }
      DSP_DIAG(SHAPE, "trace slot outSlot=%d dependsOnExternal=%d "
               "producingStep=%d frozenConst=%d shapeStatic=%d opName=%s",
               ts, depExt ? 1 : 0, producingStep,
               isFrozenConst ? 1 : 0, isShapeStatic ? 1 : 0,
               producingStep >= 0 ? slots_[producingStep].ident.opName.c_str() : "?");
    }
  }
}

// ─── Shape key computation ──────────────────────────────────────────────────

LongType NativeDynamicShapePlan::computeShapeKey(
    NativeSlot& slot, NDArray** inputs, int numInputs) {
  uint64_t key = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&key](LongType val) {
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  // Mix op identity
  mix(slot.ident.opHash);

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

  // Mix literal values ONLY for value-dependent slots (ops whose output
  // shape is a function of input VALUES, not just input shapes — e.g. Reshape
  // with a shape-tensor input, Gather where output rank depends on indices).
  //
  // For non-value-dep ops (the common case — Add, MatMul, Gather output of
  // fixed-shape tables, etc.) mixing values produces false-positive shape-key
  // changes: every decode step changes position indices / KV offsets, so the
  // key changes even though the output shape is identical. That triggers
  // POST_SEAL_SHAPE_CHANGE → mid-execution Triton compile, wrecking performance
  // and accuracy (outputs get reallocated every step).
  //
  // For value-dep ops the value mixing is required: if we skipped it, Step 2
  // with the same input SHAPE but different VALUE would get a cache hit on
  // stale cached output shapes. Line 2171 in executeSegmentSlotBySlot then
  // recomputes and verifies shapes match for the VALUE_DEP_KEY_UPDATE path.
  if (slot.flags.outputShapeDependsOnInputValues) {
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
  }

  return key;
}

// ─── Unified pre-segment zero pass ─────────────────────────────────────────
//
// Delegates to platform-specific implementation:
//   CUDA: batched cudaMemsetAsync (in _slotexec_cuda.cu)
//   CPU:  individual nullify (in _cuda_stubs.cpp)
//
// Safe to call during CUDA graph capture: memsets get recorded into the graph.
void NativeDynamicShapePlan::prezeroSegmentOutputs(const GraphSegment& seg, void* stream) {
  platformPrezeroSegmentOutputs(seg, stream);
}

// ── Slot exec error context helpers ──────────────────────────────────────
// Builds shape/arg context strings for error reporting. Shared between
// the exception handler and the status-check path in executeSlot.
static void dspFormatSlotExecContext(const NativeSlot& slot,
                                      NDArray** inputs, int numInputs,
                                      NDArray** outputs, int numOutputs,
                                      std::string& inputShapes,
                                      std::string& outputShapes,
                                      std::string& iArgs) {
  for (int i = 0; i < numInputs; i++) {
    if (i > 0) inputShapes += ", ";
    inputShapes += inputs[i] ? ShapeUtils::shapeAsString(inputs[i]) : "null";
  }
  for (int i = 0; i < numOutputs; i++) {
    if (i > 0) outputShapes += ", ";
    outputShapes += outputs[i] ? ShapeUtils::shapeAsString(outputs[i]) : "null";
  }
  for (int i = 0; i < slot.args.numIArgs; i++) {
    if (i > 0) iArgs += ",";
    iArgs += std::to_string(slot.args.iArgs[i]);
  }
}

static void dspPropagateSlotError(int stepIdx, const NativeSlot& slot,
                                   const char* detail,
                                   const std::string& inShapes,
                                   const std::string& outShapes,
                                   const std::string& iArgs) {
  char errBuf[1024];
  snprintf(errBuf, sizeof(errBuf),
           "slot %d (%s) %s | inputs=[%s] outputs=[%s] iArgs=[%s]",
           stepIdx, slot.ident.opName.c_str(), detail,
           inShapes.c_str(), outShapes.c_str(), iArgs.c_str());
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
}

// ─── Per-slot execution ─────────────────────────────────────────────────────

Status NativeDynamicShapePlan::executeSlot(
    int stepIdx, NDArray** externalArrays, int numExt, void* stream) {
  NativeSlot& slot = slots_[stepIdx];
  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  const bool frozenSlotBySlot =
      !planLifecycle_.isSlotBySlot() && execCtx != nullptr &&
      ModeContract::forMode(execCtx->graphExecutionMode).forcesSyncOnFrozen;
  auto prezeroThisSlot = [&]() {
    if (slot.frozenConstantSlot()) return;
    if (!slot.flags.needsZeroedOutput) return;
    if (slot.flags.isViewCapableOp) return;
    if (slot.flags.isIdentityOp) return;
    if (slot.flags.inPlaceFused) return;
    if (slot.fusedChain.isFusedChainTail) return;
    if (slot.slotPhase.isSealed() && slot.flags.isFullyWriting) return;

    GraphSegment zeroSeg;
    zeroSeg.def.startSlot = stepIdx;
    zeroSeg.def.endSlot = stepIdx;
    prezeroSegmentOutputs(zeroSeg, stream);
  };

  // Trace slot entry during warmup only (steady-state uses the frozen fast path).
  if (executeCount_ < 2 && DSP_DIAG_ENABLED(SHAPE)) {
    DSP_DIAG_SLOT(SHAPE, stepIdx,
        "EXEC_SLOT_ENTRY: slot %d (%s) state=%s frozenContextReady=%d shapeValid=%d "
        "isViewCapable=%d isDataDep=%d outputShapeDep=%d execCount=%d shapesFrozen=%d",
        stepIdx, slot.ident.opName.c_str(),
        slot.slotPhase.displayName(),
        slot.frozenContextReady() ? 1 : 0,
        slot.shapeCacheValid() ? 1 : 0,
        slot.flags.isViewCapableOp ? 1 : 0,
        slot.flags.isDataDependent ? 1 : 0,
        slot.flags.outputShapeDependsOnInputValues ? 1 : 0,
        executeCount_,
        shapesFrozen_ ? 1 : 0);
  }

  // ── Input validation ───────────────────────────────────────────────────────
  // Skip ALL input validation in frozen steady-state (executeCount >= 3):
  // by that point all slot inputs have been validated on prior executions
  // and buffers are stable. This eliminates ~7500 pointer checks + ~1500
  // validateSlotInputs calls per decode step.
  if (executeCount_ < 3 || planLifecycle_.isSlotBySlot()) {
    // Raw pointer sanity check: catch freed NDArrays before they crash.
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      NDArray* inp = nullptr;
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        inp = outputSlots_[srcIdx];
      } else if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx >= 0 && extIdx < numExt) inp = externalArrays[extIdx];
      }
      if (inp == nullptr) continue;
      uintptr_t addr = reinterpret_cast<uintptr_t>(inp);
      if (addr < 0x10000) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "DSP LIFECYCLE ERROR: slot %d (%s) input[%d] is a stale/freed pointer "
                 "%p (srcIdx=%d). This indicates a plan lifecycle bug — an upstream slot's "
                 "output was freed but its outputSlot entry was not nulled. "
                 "execCount=%d shapesFrozen=%d planPhase=%s",
                 stepIdx, slot.ident.opName.c_str(), i, (void*)inp, srcIdx,
                 executeCount_, shapesFrozen_ ? 1 : 0, planLifecycle_.displayName());
        THROW_EXCEPTION(msg);
      }
      // Alignment check: detect +1 byte _shapeInfoBuffer corruption early
      auto* sib = inp->shapeInfoConstBuffer();
      if (sib != nullptr) {
        uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
        if (sibAddr % alignof(ConstantShapeBuffer) != 0) {
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "DSP LIFECYCLE ERROR: slot %d (%s) input[%d] has misaligned "
                   "_shapeInfoBuffer=%p (offset %zu from %zu-byte alignment, srcIdx=%d). "
                   "This indicates heap corruption from an upstream op. "
                   "execCount=%d shapesFrozen=%d planPhase=%s",
                   stepIdx, slot.ident.opName.c_str(), i, (void*)sib,
                   static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
                   alignof(ConstantShapeBuffer), srcIdx,
                   executeCount_, shapesFrozen_ ? 1 : 0, planLifecycle_.displayName());
          THROW_EXCEPTION(msg);
        }
      }
    }

    char slotErr[512] = {};
    int badInputs = validateSlotInputs(
        stepIdx, slot.ident.opName.c_str(),
        outputSlots_, totalOutputSlots_,
        externalArrays, numExt,
        slot.wiring.inputSourceIndices, slot.wiring.numInputs,
        executeCount_, planLifecycle_.toLegacyCode(),
        slotErr, sizeof(slotErr));
    if (badInputs > 0) {
      DSP_THROW(MEMORY,
               "SLOT_INPUT_INVALID: %d invalid input(s) detected BEFORE slot %d (%s) "
               "execution: %s",
               badInputs, stepIdx, slot.ident.opName.c_str(), slotErr);
    }
  }

  // ── FAST FROZEN PATH ─────────────────────────────────────────────────────
  // Once shapes are frozen and we've executed at least twice (past warmup +
  // capture), the context pool entry has valid output arrays with correct shapes
  // and allocated buffers. All we need is: refresh inputs → nullify if needed →
  // execute → write outputs. Skip shape calc, allocation, identity detection,
  // view creation, and all first-time-only setup.
  //
  // Excluded from this path (they have their own fast early returns):
  //   - Identity ops (simple alias, no execution needed)
  //   - Frozen constants (output never changes, skip execution entirely)
  //   - Fused chain heads/tails (complex multi-slot dispatch)
  // View-capable ops are handled inline: if the installed view still shares a
  // buffer with the current input, return immediately. If stale, attempt inline
  // re-creation via tryCreateViewForSlot (avoids the heavy frozen context path).
  // Only falls through to the heavy path when view creation is not possible.
  do {
    // Gate frozen fast-path at executeCount_ >= 4 to give 4 full execution
    // passes (prefill + 3 decode) before freezing. The BFS classifies
    // external non-placeholder inputs as NOT dynamic, so slots whose shapes
    // change only during prefill-to-decode aren't marked isDynamicShape.
    // With >= 2 the fast-path activates on the first step after warmup
    // closes, reusing cached outputs that may be prezero zeros. >= 4
    // matches executeSteadyState conservatism.
    //
    // CRITICAL: Skip frozen fast-path during composite replay gap execution.
    // Gap ops in compositeReplay are called via executeSlot, but their inputs
    // (embeddings, masks, positions) change every decode step. The frozen
    // fast-path reuses cached outputs from the previous execution — wrong for
    // variable inputs. tl_dspReplayActive is set by compositeReplay for gap units.
    // Triton islands handle this correctly via re-executed compiled kernels.
    if (!(!planLifecycle_.isSlotBySlot() && !frozenSlotBySlot && executeCount_ >= 4 &&
#ifdef SD_CUDA
        !tl_dspReplayActive &&
#endif
        contextPool_[stepIdx] != nullptr && slot.frozenContextReady() &&
        !slot.flags.isIdentityOp &&
        !slot.frozenConstantSlot() &&
        !slot.fusedChain.isFusedChainHead &&
        !slot.fusedChain.isFusedChainTail &&
        !slot.flags.isDynamicShape)) break;

    // Skip frozen fast-path for slots that read directly from variable ext inputs.
    // Variable ext inputs (KV caches, embeddings, masks) change every decode step,
    // so the cached frozen output from the previous step is stale.
    {
      bool hasVariableExtInput = false;
      if (!externalInputIsVariable_.empty()) {
        for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
          int srcIdx = slot.wiring.inputSourceIndices[ii];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < (int)externalInputIsVariable_.size() && externalInputIsVariable_[extIdx]) {
              hasVariableExtInput = true;
              break;
            }
          }
        }
      }
      if (hasVariableExtInput) break;
    }

    // View-capable ops: verify the installed view still shares a buffer with
    // the current input. If the upstream gap op replaced the input array with
    // a new allocation, the view is stale and we must fall through.
    if (slot.flags.isViewCapableOp && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
      int outSi = slot.wiring.outputSlotIndices[0];
      if (outSi < 0 || outSi >= totalOutputSlots_) break;

      NDArray* currentOut = outputSlots_[outSi];
      int srcIdx = slot.wiring.inputSourceIndices[0];
      NDArray* input0 = nullptr;
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        input0 = outputSlots_[srcIdx];
      } else if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt) input0 = externalArrays[extIdx];
      }
      // View is stale or missing — attempt inline re-creation using cached
      // output shapes, avoiding the heavy frozen context path below.
      //
      // IMPORTANT: sharing the same backing DataBuffer is not sufficient to
      // prove a view wrapper is still valid. reshape/permute/squeeze chains can
      // legitimately keep the same buffer while changing rank/strides/offset.
      // Recreate the expected wrapper and only reuse the existing one if it is
      // an exact duplicate of that wrapper.
      if (input0 != nullptr && slot.shapeCacheValid() &&
          !slot.shapeCache.cachedOutputShapes.empty()) {
        static thread_local std::vector<NDArray*> ffViewInputs;
        ffViewInputs.resize(slot.wiring.numInputs);
        ffViewInputs[0] = input0;
        for (int ii = 1; ii < slot.wiring.numInputs; ii++) {
          int iiSrc = slot.wiring.inputSourceIndices[ii];
          if (iiSrc >= 0 && iiSrc < totalOutputSlots_) {
            ffViewInputs[ii] = outputSlots_[iiSrc];
          } else if (iiSrc < 0) {
            int iiExt = -(iiSrc + 1);
            ffViewInputs[ii] = (iiExt < numExt) ? externalArrays[iiExt] : nullptr;
          }
        }

        NDArray* ffNewView = nullptr;
        LongType ffViewOffset = 0;
        ViewCreateResult ffVcr = tryCreateViewForSlot(
            stepIdx, slot, input0, slot.shapeCache.cachedOutputShapes[0],
            ffViewInputs.data(), slot.wiring.numInputs,
            &ffNewView, &ffViewOffset);

        if (ffVcr == VIEW_CREATED) {
          if (currentOut != nullptr &&
              currentOut->hasValidShapeInfo() &&
              currentOut->dataBuffer() == input0->dataBuffer() &&
              currentOut->offset() == ffViewOffset &&
              ffNewView != nullptr &&
              currentOut->dataType() == ffNewView->dataType() &&
              shape::shapeEquals(currentOut->shapeInfo(), ffNewView->shapeInfo()) &&
              shape::strideEquals(currentOut->shapeInfo(), ffNewView->shapeInfo())) {
            delete ffNewView;
            backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
            return Status::OK;
          }
          if (sd::Environment::getInstance().isDebug()) {
            char bffLabel[256];
            snprintf(bffLabel, sizeof(bffLabel),
                     "BEFORE_ffViewInstall_step%d_%s", stepIdx, slot.ident.opName.c_str());
            scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                bffLabel, executeCount_);
          }
          writeOutputSlot(outSi, ffNewView, "ff-view-install");
          if (sd::Environment::getInstance().isDebug()) {
            char affLabel[256];
            snprintf(affLabel, sizeof(affLabel),
                     "AFTER_ffViewInstall_step%d_%s", stepIdx, slot.ident.opName.c_str());
            scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                affLabel, executeCount_);
          }
          DSP_DIAG_SLOT_WRITE(outSi, slot.ident.opName.c_str(),
                              ffNewView != nullptr && ffNewView->dataBuffer() != nullptr
                                  ? ffNewView->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "ff-view-install");
          slotIsViewProducer_[outSi] = true;
          auto& ffViewCtx = *contextPool_[stepIdx];
          ffViewCtx.setOutputArray(0, ffNewView);
          ffViewCtx.setInputArray(0, input0);
          outputSlots_[outSi] = ffNewView;
          // Keep ownership metadata pointing at the new DataBuffer
          if (slotOwnership_ != nullptr) {
            slotOwnership_[outSi].dataBuffer = ffNewView->dataBuffer();
          }
          // Slot-write site: fast-frozen view install. Bump generation AFTER
          // the view has been wired into the slot array.
          slot.bumpGeneration();
          dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
          backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
          return Status::OK;
        } else if (ffVcr == VIEW_STRIDED_SLICE_FAIL) {
          if (slot.slotPhase.isSealed()) {
            slot.slotPhase.unseal(); slot.slotPhase.shapeCacheValid = true;  // PRIMARY: demote
            slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
          }
          break;  // fall through to normal path
        } else if (ffVcr == VIEW_STALE_EMPTY_SHAPE) {
          slot.slotPhase.reset();  // PRIMARY
          slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
          break;  // fall through to normal path
        }
        // VIEW_NOT_POSSIBLE — fall through
      }
      break;  // view-capable but couldn't handle — fall through
    }

    // Non-view-capable ops: refresh inputs → conditional nullify → execute → write outputs
    {
      auto& ffCtx = *contextPool_[stepIdx];

      // ── Lifecycle assertion: validate context pool outputs ──
      // Only during early executions (< 4): by that point context pool outputs
      // have been validated repeatedly and are stable. This eliminates ~1000
      // pointer-check iterations per decode step in frozen steady-state.
      if (executeCount_ < 4) {
        auto& ffOuts = ffCtx.fastpath_out();
        for (int i = 0; i < slot.wiring.numOutputs && i < static_cast<int>(ffOuts.size()); i++) {
          NDArray* outArr = ffOuts[i];
          if (outArr == nullptr) continue;
          uintptr_t addr = reinterpret_cast<uintptr_t>(outArr);
          if (addr < 0x10000) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "DSP LIFECYCLE ERROR: frozen fast-path slot %d (%s) has stale "
                     "context pool output[%d] = %p (freed NDArray). "
                     "execCount=%d shapesFrozen=%d",
                     stepIdx, slot.ident.opName.c_str(), i, (void*)outArr,
                     executeCount_, shapesFrozen_ ? 1 : 0);
            THROW_EXCEPTION(msg);
          }
          auto* sib = outArr->shapeInfoConstBuffer();
          if (sib != nullptr) {
            uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
            if (sibAddr < 0x10000 || sibAddr % alignof(ConstantShapeBuffer) != 0) {
              char msg[512];
              snprintf(msg, sizeof(msg),
                       "DSP LIFECYCLE ERROR: frozen fast-path slot %d (%s) context pool "
                       "output[%d] has corrupted shapeInfoConstBuffer = %p "
                       "(alignOffset=%zu). execCount=%d shapesFrozen=%d",
                       stepIdx, slot.ident.opName.c_str(), i, (void*)sib,
                       static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
                       executeCount_, shapesFrozen_ ? 1 : 0);
              THROW_EXCEPTION(msg);
            }
          }
        }
      }

      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
            ffCtx.setInputArray(i, externalArrays[extIdx]);
          }
        } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          if (outputSlots_[srcIdx] != nullptr) {
            ffCtx.setInputArray(i, outputSlots_[srcIdx]);
          }
        }
      }

      // Composite replay gap ops: graph replay writes device memory directly
      // without updating actuality flags (no registerSpecialUse in CUDA graph
      // replay). When forceSync_ is true (set by compositeReplay for gap ops),
      // we MUST call prepareSpecialUse to ensure inputs are synced from device
      // and registerSpecialUse to tick the output's device-write counter.
      // Without this, gap ops (especially matmul for logits) read stale data
      // and produce zeros from exec=2 onward.
      if (forceSync_) {
        NDArray::prepareSpecialUse(ffCtx.fastpath_out(), ffCtx.fastpath_in());
      }

      // Skip shape inference in frozen steady-state: outputs are already
      // allocated with correct shapes from warmup. calculateOutputShape +
      // prepareOutputs is the dominant per-op overhead (~50ms of the ~60ms
      // per-op cost on CPU). shapeFunctionOverride makes prepareOutputs
      // return immediately, bypassing shape inference and allocation.
      if (executeCount_ >= 3) {
        ffCtx.setShapeFunctionOverride(true);
      }

      // Pre-execute corruption scan: check this slot's outputs BEFORE op runs.
      if (sd::Environment::getInstance().isDebug()) {
        for (int i = 0; i < slot.wiring.numOutputs; i++) {
          int si = slot.wiring.outputSlotIndices[i];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
            if (!validateSlotShapeBufferAlignment(outputSlots_[si], si,
                    "BEFORE_ff_op_execute", executeCount_)) {
              char msg[512];
              snprintf(msg, sizeof(msg),
                       "CORRUPTION_BEFORE_EXEC: slot %d (%s) output[%d] at outputSlot %d "
                       "already corrupt BEFORE op->execute. Corruption from prior op or "
                       "refreshStaleViewWrappers. execCount=%d",
                       stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
              THROW_EXCEPTION(msg);
            }
          }
        }
      }

      Status ffStatus = Status::OK;
      if (shapeOnlyMode_) {
        DSP_DIAG(EXECUTE,
                 "SHAPE_ONLY: skipping op execution for slot %d (%s) [fast-frozen path]",
                 stepIdx, slot.ident.opName.c_str());
      } else {
        prezeroThisSlot();
        ffStatus = slot.ident.op->execute(&ffCtx);
      }

      // Post-execute corruption scan: check this slot's outputs AFTER op runs.
      if (sd::Environment::getInstance().isDebug()) {
        for (int i = 0; i < slot.wiring.numOutputs; i++) {
          int si = slot.wiring.outputSlotIndices[i];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
            if (!validateSlotShapeBufferAlignment(outputSlots_[si], si,
                    "AFTER_ff_op_execute", executeCount_)) {
              char msg[512];
              snprintf(msg, sizeof(msg),
                       "CORRUPTION_AFTER_EXEC: slot %d (%s) output[%d] at outputSlot %d "
                       "corrupted BY op->execute. This op caused the heap overrun. "
                       "execCount=%d",
                       stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
              THROW_EXCEPTION(msg);
            }
          }
        }
      }

      // Full slot scan after each op execute — catches cross-slot heap overruns.
      // Gated on isDebug() to avoid steady-state overhead.
      if (sd::Environment::getInstance().isDebug()) {
        char fsLabel[256];
        snprintf(fsLabel, sizeof(fsLabel),
                 "AFTER_ff_exec_step%d_%s", stepIdx, slot.ident.opName.c_str());
        scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
            fsLabel, executeCount_);
      }

      if (forceSync_) {
        NDArray::registerSpecialUse(ffCtx.fastpath_out(), ffCtx.fastpath_in());
      }

      auto& ffOuts = ffCtx.fastpath_out();

      // Targeted per-slot output fingerprint — active when debug+verbose
      // Only fingerprints at exec count >= 2 to avoid flooding during warmup
      if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose()) {
        for (int i = 0; i < slot.wiring.numOutputs && i < static_cast<int>(ffOuts.size()); i++) {
          if (ffOuts[i] != nullptr && !ffOuts[i]->isEmpty() && ffOuts[i]->lengthOf() > 0) {
            ffOuts[i]->syncToHost();
            double sum = 0.0;
            double firstVal = ffOuts[i]->e<double>(0);
            sd::LongType len = ffOuts[i]->lengthOf();
            sd::LongType fpLen = sd::math::sd_min(64LL, len);
            for (sd::LongType fi = 0; fi < fpLen; fi++) {
              sum += ffOuts[i]->e<double>(fi);
            }
            DSP_DIAG(VERIFY,
                     "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld first=%.8g sum64=%.8g",
                     executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                     ShapeUtils::shapeAsString(ffOuts[i]).c_str(),
                     DataTypeUtils::asString(ffOuts[i]->dataType()).c_str(),
                     (long long)len, firstVal, sum);
          }
        }
      }

      for (int i = 0; i < slot.wiring.numOutputs && i < static_cast<int>(ffOuts.size()); i++) {
        if (ffOuts[i] != nullptr) {
          // Post-execution output integrity check: detect corruption from
          // the op that just ran (buffer overrun, type-size mismatch, etc.)
          if (sd::Environment::getInstance().isDebug()) {
            auto* postSib = ffOuts[i]->shapeInfoConstBuffer();
            if (postSib != nullptr) {
              uintptr_t postSibAddr = reinterpret_cast<uintptr_t>(postSib);
              if (postSibAddr % alignof(ConstantShapeBuffer) != 0) {
                char msg[512];
                snprintf(msg, sizeof(msg),
                         "DSP CORRUPTION DETECTED: slot %d (%s) output[%d] _shapeInfoBuffer=%p "
                         "is misaligned AFTER op execution (alignOffset=%zu). "
                         "The op itself corrupted this output's shape buffer. "
                         "execCount=%d shapesFrozen=%d",
                         stepIdx, slot.ident.opName.c_str(), i, (void*)postSib,
                         static_cast<size_t>(postSibAddr % alignof(ConstantShapeBuffer)),
                         executeCount_, shapesFrozen_ ? 1 : 0);
                THROW_EXCEPTION(msg);
              }
            }
          }
          int si = slot.wiring.outputSlotIndices[i];
          if (si >= 0 && si < totalOutputSlots_) {
            // In frozen steady-state, the context pool output IS the slot output —
            // same pointer every step. Skip reconcile/writeOutputSlot/diag overhead
            // when the pointer hasn't changed (the common case post-warmup).
            if (ffOuts[i] != outputSlots_[si]) {
              platformReconcileOutputActuality("fast-frozen", stepIdx, slot, ffOuts[i]);
              writeOutputSlot(si, ffOuts[i], "fast-frozen");
              DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                                  ffOuts[i]->dataBuffer() != nullptr
                                      ? ffOuts[i]->dataBuffer()->getLenInBytes()
                                      : 0,
                                  stream, "fast-frozen");
              outputSlots_[si] = ffOuts[i];
            }
          }
        }
      }

      if (ffStatus == Status::OK) {
        slot.bumpGeneration();
        for (int i = 0; i < slot.wiring.numOutputs; i++) {
          int si = slot.wiring.outputSlotIndices[i];
          if (si >= 0 && si < totalOutputSlots_) dirtySlotGenerations_[si] = currentDirtyGeneration_;
        }
      }
      return ffStatus;
    }
  } while (false);

  auto discardCachedSlotArray = [&](int slotIdx, NDArray* cached, const char* tag) {
    if (cached == nullptr) return;

    // Clear the live slot before deleting to avoid a dangling pointer when
    // a shape transition forces inline replacement.
    if (outputSlots_ != nullptr &&
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
      if (sd::Environment::getInstance().isDebug()) {
        char dcLabel[256];
        snprintf(dcLabel, sizeof(dcLabel),
                 "AFTER_discardCached_slot%d_%s", slotIdx, tag);
        scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
            dcLabel, executeCount_);
      }
      DSP_DIAG(MEMORY, "discardCachedSlotArray: slot=%d tag=%s deleted=%p",
               slotIdx, tag, (void*)cached);
    } else {
      DSP_DIAG(MEMORY, "discardCachedSlotArray: slot=%d tag=%s preserved=%p sharedOrCapturing=%d",
               slotIdx, tag, (void*)cached,
               (tl_graphExecutionActive || isSlotArrayShared(cached, slotIdx)) ? 1 : 0);
    }

    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      outputSlots_[slotIdx] = nullptr;
    }
  };

  auto validateReusableSlotArray = [&](int slotIdx, NDArray* cached,
                                       const char* tag) -> NDArray* {
    if (cached == nullptr) return nullptr;

    // Stale views (destructed NDArrays still referenced via dangling pointers)
    // have _shapeInfo == nullptr. Use safeHasValidShapeInfo to catch +1 byte
    // _shapeInfoBuffer corruption without crashing.
    if (!safeHasValidShapeInfo(cached)) return nullptr;
    auto* db = cached->dataBuffer();
    bool invalid = (db == nullptr) || db->isClosed() || !db->isValid();
    if (!invalid && !platformValidateReusableSlotBuffer(cached)) {
      invalid = true;
    }

    if (!invalid) return cached;

    // View-capable ops legitimately recreate their output wrapper on each
    // execution. When the upstream input buffer changes (fresh placeholder feed
    // or an earlier view in the chain being rebuilt), the previously cached
    // wrapper can point at a closed DataBuffer by the time Step 3 reaches it.
    // Treat that as a cache miss and let the view-install path below recreate
    // the wrapper from current inputs. Non-view slots remain a hard error.
    if (slot.flags.isViewCapableOp ||
        slot.flags.outputShapeDependsOnInputValues ||
        executeCount_ == 0) {
      // View-capable ops recreate their output wrapper each step.
      // Value-dependent ops (Where, NonZero) produce variable-length output
      // that can be empty (null DataBuffer) one step and non-empty the next.
      // In both cases, a stale/empty cached output is a normal cache miss —
      // drop it and let the allocation path below create a fresh one.
      DSP_DIAG_SLOT(MEMORY, stepIdx,
          "STALE_REBUILD: slot %d (%s) dropping stale cached output tag=%s "
          "arr=%p db=%p valid=%d closed=%d exec=%d",
          slotIdx, slot.ident.opName.c_str(), tag,
          (void*)cached, (void*)db,
          db != nullptr && db->isValid() ? 1 : 0,
          db != nullptr && db->isClosed() ? 1 : 0,
          executeCount_);
      if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
        outputSlots_[slotIdx] = nullptr;
      }
      return nullptr;
    }

    // A freed/closed DataBuffer at a bound slot is an invariant violation.
    // Slots are immutable once bound; if the backing buffer was freed while
    // still referenced by a slot, that is a lifetime-management bug in the
    // plan or its owner — not something we should silently paper over by
    // discarding the cached array and allocating a replacement.
    DSP_THROW(MEMORY,
             "NativeDynamicShapePlan: stale/closed DataBuffer at bound slot %d (%s) tag=%s "
             "arr=%p db=%p valid=%d closed=%d len=%lld exec=%d. "
             "Slots are immutable once bound; a freed backing buffer indicates a "
             "lifetime-management bug — fix the owner of this DataBuffer.",
             slotIdx, slot.ident.opName.c_str(), tag, (void*)cached, (void*)db,
             db != nullptr && db->isValid() ? 1 : 0,
             db != nullptr && db->isClosed() ? 1 : 0,
             (long long)cached->lengthOf(), executeCount_);
    return nullptr;  // unreachable
  };

  // ── Fast path: identity ops ──────────────────────────────────────────────
  if (slot.flags.isIdentityOp && slot.wiring.numInputs == 1 && slot.wiring.numOutputs >= 1) {
    int srcIdx = slot.wiring.inputSourceIndices[0];
    NDArray* input = nullptr;
    if (srcIdx >= 0) {
      input = outputSlots_[srcIdx];
    } else {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) input = externalArrays[extIdx];
    }
    if (input != nullptr) {
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          writeOutputSlot(si, input, "identity-alias");  // expected: alias, not new allocation
          DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                              input != nullptr && input->dataBuffer() != nullptr
                                  ? input->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "identity-alias");
        }
      }
      platformLogSlotOutput(stepIdx, slot.ident.opName.c_str(), "IDENTITY",
                            slot.wiring.outputSlotIndices, slot.wiring.numOutputs);
      // Slot-write site: identity alias. Even though no kernel ran, the slot's
      // output NDArray pointer has just been (re)assigned, so any stale-read
      // check would need to see a fresh generation.
      slot.bumpGeneration();
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
      return Status::OK;
    }
  }

  // ── Frozen constant optimization ──────────────────────────────────────────
  // Skip frozen constant slots from executeCount_ >= 2 onward. This ensures
  // detectFrozenConstants() (which runs at executeCount_ == 1) has completed
  // and all frozen slot outputs are populated from prior executions. The
  // allOutputsPopulated check below provides an additional safety gate.
  if (slot.frozenConstantSlot() && !planLifecycle_.isSlotBySlot() && executeCount_ >= 2) {
    // Verify all outputs are populated before skipping. If outputSlots_[si]
    // is null (first execution), fall through to execute the slot so
    // downstream consumers get a valid array.
    bool allOutputsPopulated = true;
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int si = slot.wiring.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
        allOutputsPopulated = false;
        break;
      }
    }
    if (allOutputsPopulated) {
      DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s [SKIPPED:frozen-const]", stepIdx, slot.ident.opName.c_str());
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
#if defined(SD_CUDA)
      if (tl_graphExecutionActive && stepIdx == 273) {
        cudaStream_t retStr = *LaunchContext::defaultContext()->getCudaStream();
        cudaStreamCaptureStatus retStat = cudaStreamCaptureStatusNone;
        cudaStreamGetCaptureInfo_v2(retStr, &retStat, nullptr, nullptr, nullptr, nullptr);
        printf("SLOT_FCONST_RETURN: slot 273 returning early (frozen-const skip) capStat=%d\n", (int)retStat);
        fflush(stdout);
      }
#endif
      return Status::OK;
    }
    // Fall through to execute — output slot is null, must re-execute to populate it
    DSP_DIAG(EXECUTE, "SLOT_EXEC step=%d op=%s frozen-const but output null, re-executing",
             stepIdx, slot.ident.opName.c_str());
  }

  // ── Fused chain tail skip ─────────────────────────────────────────────────
  if (slot.fusedChain.isFusedChainTail) {
    DSP_DIAG(EXECUTE, "FUSED_TAIL_SKIP: step=%d op=%s chainSlot0=%d chainLen=%d — returning OK without execution",
             stepIdx, slot.ident.opName.c_str(),
             slot.fusedChain.fusedChainSlots[0], slot.fusedChain.fusedChainLength);
    return Status::OK;
  }

  // ── Fused chain head dispatch + frozen context block ───────────────────────
  // Wrapped in do/while(false) so break exits to normalExecution below.
  do {
  if (slot.fusedChain.isFusedChainHead && slot.fusedChain.fusedChainLength >= 2) {
    // 1. Gather primary input (head slot's first input)
    NDArray* primaryInput = nullptr;
    int primarySrcIdx = slot.wiring.inputSourceIndices[0];
    if (primarySrcIdx >= 0) {
      primaryInput = outputSlots_[primarySrcIdx];
    } else {
      int extIdx = -(primarySrcIdx + 1);
      if (extIdx < numExt) primaryInput = externalArrays[extIdx];
    }
    if (primaryInput == nullptr) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx, "NULL fused head input for slot %d (%s), primarySrcIdx=%d",
                stepIdx, slot.ident.opName.c_str(), primarySrcIdx);
      return Status::BAD_INPUT;
    }

    bool headIsBinary = (slot.fusedChain.fusedChainSecondaryInputSources[0] != INT32_MIN);
    if (headIsBinary && slot.wiring.numInputs == 2) {
      int secSrc = slot.fusedChain.fusedChainSecondaryInputSources[0];
      for (int k = 0; k < slot.wiring.numInputs; k++) {
        if (slot.wiring.inputSourceIndices[k] != secSrc) {
          int chainSrcIdx = slot.wiring.inputSourceIndices[k];
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

    for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
      fusedOps[ci] = static_cast<sd::ops::helpers::FusedElemOp>(slot.fusedChain.fusedChainOpCodes[ci]);

      int secSrc = slot.fusedChain.fusedChainSecondaryInputSources[ci];
      if (secSrc != INT32_MIN) {
        if (secSrc >= 0) {
          secondaryInputs[ci] = outputSlots_[secSrc];
            } else {
          int extIdx = -(secSrc + 1);
          if (extIdx < numExt) secondaryInputs[ci] = externalArrays[extIdx];
        }
      }
      if (ci == 0 && secondaryInputs[ci] == nullptr &&
          sd::ops::helpers::isBinaryFusedOp(fusedOps[ci]) &&
          slot.wiring.numInputs == 2 &&
          slot.wiring.inputSourceIndices[0] == slot.wiring.inputSourceIndices[1]) {
        secondaryInputs[ci] = primaryInput;
      }
    }

    // 3. Determine output shape considering broadcasting
    const LongType* outputShapeInfo = primaryInput->shapeInfo();
    bool needsBroadcast = false;
    for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
      if (secondaryInputs[ci] != nullptr && sd::ops::helpers::isBinaryFusedOp(fusedOps[ci])) {
        const LongType* secShape = secondaryInputs[ci]->shapeInfo();
        if (!shape::equalsSoft(outputShapeInfo, secShape)) {
          needsBroadcast = true;
          break;
        }
      }
    }

    if (needsBroadcast) {
      slot.fusedChain.isFusedChainHead = false;
      for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
        int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
        if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_) {
          slots_[chainSlotIdx].fusedChain.isFusedChainTail = false;
        }
      }
      slot.fusedChain.fusedChainLength = 0;
      break;  // → normalExecution
    }

    // 4. Allocate/reuse output for the LAST chain slot
    int lastSlotIdx = slot.fusedChain.fusedChainSlots[slot.fusedChain.fusedChainLength - 1];
    int lastOutputSlotIdx = slots_[lastSlotIdx].wiring.outputSlotIndices[0];

    NDArray* output = nullptr;
    if (lastOutputSlotIdx >= 0 && lastOutputSlotIdx < totalOutputSlots_) {
      output = outputSlots_[lastOutputSlotIdx];
        if (output != nullptr) {
          if (!shape::equalsSoft(output->shapeInfo(), outputShapeInfo) ||
              ArrayOptions::dataType(output->shapeInfo()) != ArrayOptions::dataType(outputShapeInfo)) {
            // Transitively check the head slot AND all chain slots for any
            // external input or isDynamicShape upstream. The BFS helper walks
            // the full dependency graph so it catches chains like:
            //   fused_head <- intermediate <- ext[N]
            bool hasDynamicUpstream = hasTransitiveDynamicUpstream(
                slots_, numSlots_, stepIdx);
            if (!hasDynamicUpstream) {
              for (int ci = 0; ci < slot.fusedChain.fusedChainLength && !hasDynamicUpstream; ci++) {
                int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
                if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_) {
                  hasDynamicUpstream = hasTransitiveDynamicUpstream(
                      slots_, numSlots_, chainSlotIdx);
                }
              }
            }

            if (executeCount_ <= 1 || hasDynamicUpstream) {
              // Warmup (execCount 0 = prefill, 1 = first decode) or dynamic
              // upstream: shape drift expected. Reassign.
              DSP_DIAG_SLOT(SHAPE, lastSlotIdx,
                  "fused-chain-reassign: slot %d shape changed (execCount=%d dynamicUpstream=%d) — replacing",
                  lastSlotIdx, executeCount_, hasDynamicUpstream ? 1 : 0);
              // Must use discardCachedSlotArray to erase from planOwnedArrays_
              // before deleting — raw delete leaves a ghost pointer that causes
              // heap corruption when malloc reuses the freed chunk.
              discardCachedSlotArray(lastOutputSlotIdx, outputSlots_[lastOutputSlotIdx], "fused-chain-reassign");
              output = nullptr;
              if (sd::Environment::getInstance().isDebug()) {
                char fcDelLabel[256];
                snprintf(fcDelLabel, sizeof(fcDelLabel),
                         "AFTER_fusedChainDelete_slot%d", lastOutputSlotIdx);
                scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                    fcDelLabel, executeCount_);
              }
              auto& lastSlot = slots_[lastSlotIdx];
              lastSlot.slotPhase.reset();  // PRIMARY
              lastSlot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
              lastSlot.shapeCache.cachedOutputShapes.clear();
              // Only mark as permanently dynamic if genuinely dynamic upstream.
              if (hasDynamicUpstream) {
                lastSlot.flags.isDynamicShape = true;
                slot.flags.isDynamicShape = true;
              }
            } else {
              DSP_THROW(SHAPE,
                       "NativeDynamicShapePlan: fused-chain slot reassignment on shape drift "
                       "is forbidden post-warmup at slot %d (%s) — output shape changed but "
                       "chain has no dynamic upstream (execCount=%d hasDynamicUpstream=%d). "
                       "This indicates heap corruption or a missing isDynamicShape flag.",
                       lastSlotIdx, slot.ident.opName.c_str(), executeCount_,
                       hasDynamicUpstream ? 1 : 0);
            }
          }
        }
      if (output == nullptr) {
        output = new NDArray(const_cast<LongType*>(outputShapeInfo), true, LaunchContext::defaultContext());
        outputSlots_[lastOutputSlotIdx] = output;
      }
    } else {
      DSP_DIAG(EXECUTE, "invalid output slot index for fused chain tail slot %d", lastSlotIdx);
      return Status::BAD_INPUT;
    }

    // 5. Call fused kernel
    {
      // Dump fused chain dispatch details
      for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
        DSP_DIAG(EXECUTE, "FUSED_CHAIN_DISPATCH step=%d op=%s ci=%d fusedOp=%d secondary=%p inputLen=%lld",
                 stepIdx, slot.ident.opName.c_str(), ci, (int)fusedOps[ci],
                 (void*)secondaryInputs[ci], (long long)primaryInput->lengthOf());
      }
    }
    LaunchContext* lc = LaunchContext::defaultContext();
    sd::ops::helpers::fusedElementwiseChain(
        primaryInput, output, fusedOps, slot.fusedChain.fusedChainLength,
        secondaryInputs, nullptr, nullptr, lc);

    // Dump first 4 output values after kernel — relu output must have no negatives
    {
      auto len = output->lengthOf();
      int show = len < 4 ? (int)len : 4;
      if (output->dataType() == FLOAT32 && output->buffer() != nullptr) {
        auto buf = output->bufferAsT<float>();
        float oMin = buf[0], oMax = buf[0];
        for (sd::LongType i = 1; i < len; i++) { if (buf[i] < oMin) oMin = buf[i]; if (buf[i] > oMax) oMax = buf[i]; }
        DSP_DIAG(EXECUTE, "FUSED_CHAIN_POST step=%d outMin=%.6f outMax=%.6f chainLen=%d first4=[%.4f,%.4f,%.4f,%.4f]",
                 stepIdx, oMin, oMax, slot.fusedChain.fusedChainLength,
                 show > 0 ? buf[0] : 0.f, show > 1 ? buf[1] : 0.f,
                 show > 2 ? buf[2] : 0.f, show > 3 ? buf[3] : 0.f);
      }
    }

    output->tickWriteDevice();

    // 6. Register result at all chain slots' output indices
    writeOutputSlot(lastOutputSlotIdx, output, "fused-chain-head");
    DSP_DIAG_SLOT_WRITE(lastOutputSlotIdx, slot.ident.opName.c_str(),
                        output != nullptr && output->dataBuffer() != nullptr
                            ? output->dataBuffer()->getLenInBytes()
                            : 0,
                        stream, "fused-chain-head");
    // Populate intermediate chain slots with their OWN separate arrays.
    // CRITICAL: do NOT alias intermediate slots to the final output pointer.
    // The OpenVINO/OneDNN backend re-executes the segment and binds EACH slot's
    // output buffer independently. If intermediate and final slots share the same
    // NDArray, OV writes the intermediate result (e.g., add-only) on top of the
    // final result (add+relu), corrupting it. Each slot needs its own buffer so
    // the backend writes don't interfere.
    for (int ci = 0; ci < slot.fusedChain.fusedChainLength - 1; ci++) {
      int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
      int chainOutputSlotIdx = slots_[chainSlotIdx].wiring.outputSlotIndices[0];
      if (chainOutputSlotIdx >= 0 && chainOutputSlotIdx < totalOutputSlots_) {
        NDArray* memberOut = outputSlots_[chainOutputSlotIdx];
        if (memberOut == nullptr) {
          // Allocate a separate array for this intermediate slot
          memberOut = new NDArray(const_cast<LongType*>(outputShapeInfo), true, LaunchContext::defaultContext());
          writeOutputSlot(chainOutputSlotIdx, memberOut, "fused-chain-member-alloc");
        }
        DSP_DIAG_SLOT_WRITE(chainOutputSlotIdx, slot.ident.opName.c_str(),
                            memberOut != nullptr && memberOut->dataBuffer() != nullptr
                                ? memberOut->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "fused-chain-member");
      }
    }

    {
      int outIndices[1] = { lastOutputSlotIdx };
      platformLogSlotOutput(stepIdx, slot.ident.opName.c_str(), "FUSED_HEAD",
                            outIndices, 1);
    }

    // Slot-write site: fused chain. The fused kernel has finished writing
    // `output` and all chain-member slot indices have been pointed at it.
    // Bump the head slot AND every chain-member slot so the generation
    // counter reflects that their outputs are now freshly written.
    slot.bumpGeneration();
    for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
      int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
      if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_ && chainSlotIdx != stepIdx) {
        slots_[chainSlotIdx].bumpGeneration();
      }
    }

    // Mark all fused chain output slots as dirty.
    dirtySlotGenerations_[lastOutputSlotIdx] = currentDirtyGeneration_;
    for (int ci = 0; ci < slot.fusedChain.fusedChainLength - 1; ci++) {
      int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
      int chainOutputSlotIdx = slots_[chainSlotIdx].wiring.outputSlotIndices[0];
      if (chainOutputSlotIdx >= 0 && chainOutputSlotIdx < totalOutputSlots_) {
        dirtySlotGenerations_[chainOutputSlotIdx] = currentDirtyGeneration_;
      }
    }

    // Backfill shapes for head AND all tail slots in the fused chain
    backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
    for (int ci = 0; ci < slot.fusedChain.fusedChainLength; ci++) {
      int chainSlotIdx = slot.fusedChain.fusedChainSlots[ci];
      if (chainSlotIdx >= 0 && chainSlotIdx < numSlots_) {
        backfillCachedOutputShapes(slots_[chainSlotIdx], outputSlots_, totalOutputSlots_);
      }
    }

    return Status::OK;
  }

  // ── Fast path: frozen context ────────────────────────────────────────────
  // Dynamic-shape slots always bypass this path: their output shape changes each
  // step, so the frozen context's cached shapes/buffers are stale. They must
  // fall through to the normal execution path below.
  if (!frozenSlotBySlot && slot.frozenContextReady() && !slot.flags.isDynamicShape) {

    // ── View-capable fast path (reshape/expand_dims/squeeze/strided_slice) ──
    if (slot.flags.isViewCapableOp && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
      int si = slot.wiring.outputSlotIndices[0];
      if (si >= 0 && si < totalOutputSlots_) {
        // Resolve input0 from slot source indices
        int srcIdx = slot.wiring.inputSourceIndices[0];
        NDArray* input0 = nullptr;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          input0 = outputSlots_[srcIdx];
        } else if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt) input0 = externalArrays[extIdx];
        }

        if (input0 != nullptr && slot.shapeCacheValid() && !slot.shapeCache.cachedOutputShapes.empty()) {
          // Trace frozen view installation during warmup only.
          if (executeCount_ < 2 && DSP_DIAG_ENABLED(SHAPE)) {
            auto* db = input0->dataBuffer();
            DSP_DIAG_SLOT(SHAPE, stepIdx,
                "FROZEN-VIEW-TRACE: slot %d (%s) input0 len=%lld dtype=%d srcIdx=%d "
                "pAct=%d sAct=%d cachedShape=%s values=%s",
                stepIdx, slot.ident.opName.c_str(),
                (long long)input0->lengthOf(), (int)input0->dataType(), srcIdx,
                db ? db->isPrimaryActual() : -1,
                db ? db->isSpecialActual() : -1,
                ShapeUtils::shapeAsString(slot.shapeCache.cachedOutputShapes[0]).c_str(),
                dspDumpHostDeviceValues(input0, 16).c_str());
          }
          // Gather all inputs for strided_slice offset computation
          static thread_local std::vector<NDArray*> ssInputs;
          ssInputs.resize(slot.wiring.numInputs);
          ssInputs[0] = input0;
          for (int ii = 1; ii < slot.wiring.numInputs; ii++) {
            int iiSrc = slot.wiring.inputSourceIndices[ii];
            if (iiSrc >= 0 && iiSrc < totalOutputSlots_) {
              ssInputs[ii] = outputSlots_[iiSrc];
            } else if (iiSrc < 0) {
              int iiExt = -(iiSrc + 1);
              ssInputs[ii] = (iiExt < numExt) ? externalArrays[iiExt] : nullptr;
            }
          }

          NDArray* newView = nullptr;
          LongType viewOffset = 0;
          // For view ops (reshape/expand_dims/squeeze), the output type MUST
          // match the input type — the view shares the input's data buffer.
          // The cached output shape may have a stale type if the input source
          // changed between executions (e.g., attention mask growth changes
          // what ext input feeds into this slot).
          const LongType* frozenOutShape = slot.shapeCache.cachedOutputShapes[0];
          static thread_local LongType correctedShapeBuffer[SD_MAX_SHAPEINFOLENGTH];
          if (frozenOutShape != nullptr && input0 != nullptr &&
              ArrayOptions::dataType(frozenOutShape) != input0->dataType()) {
            int shapeLen = shape::shapeInfoLength(shape::rank(frozenOutShape));
            std::memcpy(correctedShapeBuffer, frozenOutShape, shapeLen * sizeof(LongType));
            ArrayOptions::setDataType(correctedShapeBuffer, input0->dataType());
            frozenOutShape = correctedShapeBuffer;
          }
          ViewCreateResult vcr = tryCreateViewForSlot(
              stepIdx, slot, input0, frozenOutShape,
              ssInputs.data(), slot.wiring.numInputs,
              &newView, &viewOffset);

          if (vcr == VIEW_STRIDED_SLICE_FAIL) {
            // Can't create view (non-unit strides or newAxisMask) — fall through to normal execution
            if (slot.slotPhase.isSealed()) {
              slot.slotPhase.unseal(); slot.slotPhase.shapeCacheValid = true;  // PRIMARY: demote
              slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
            }
            break;  // → normalExecution
          } else if (vcr == VIEW_STALE_EMPTY_SHAPE) {
            if (executeCount_ > 1) {
              DSP_DIAG_SLOT(VERIFY, stepIdx,
                  "BUG: Frozen output empty post-warmup at slot %d (%s) — persistence bug. executeCount=%d",
                  stepIdx, slot.ident.opName.c_str(), executeCount_);
            }
            // Stale empty shape from Step 0 — input is now non-empty.
            // Invalidate frozen state and fall through to normal path
            // for shape re-inference with actual input shapes.
            DSP_DIAG_SLOT(SHAPE, stepIdx,
                "view-capable slot %d (%s): frozen shape empty but input "
                "non-empty (len=%lld) — re-inferring via normal path (warmup)",
                stepIdx, slot.ident.opName.c_str(), input0->lengthOf());
            slot.slotPhase.reset();  // PRIMARY
            slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
            break;  // → normalExecution
          } else if (vcr == VIEW_CREATED) {
            // Check if existing view already matches — skip re-creation
            NDArray* currentOut = outputSlots_[si];
            if (currentOut != nullptr &&
                currentOut->hasValidShapeInfo() &&
                newView != nullptr &&
                newView->hasValidShapeInfo() &&
                currentOut->dataBuffer() == newView->dataBuffer() &&
                currentOut->offset() == newView->offset() &&
                currentOut->dataType() == newView->dataType() &&
                shape::shapeEquals(currentOut->shapeInfo(), newView->shapeInfo()) &&
                shape::strideEquals(currentOut->shapeInfo(), newView->shapeInfo())) {
              // Trace only during warmup — in steady-state these iterate
              // inputs/outputs and format strings per step.
              if (executeCount_ < 2) {
                NDArray* tracedInputs[1] = {input0};
                NDArray* tracedOutputs[1] = {currentOut};
                traceSmallControlSlotIO("frozen-view-reuse", stepIdx, slot,
                                        input0, currentOut,
                                        executeCount_, planLifecycle_.displayName());
                traceSmallControlSlotTensors("frozen-view-reuse", stepIdx, slot,
                                             tracedInputs, 1, tracedOutputs, 1,
                                             executeCount_, planLifecycle_.displayName());
              }
              delete newView;  // Discard redundant view
              backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
              return Status::OK;
            }
            // Install the new view
            if (sd::Environment::getInstance().isDebug()) {
              char bvLabel[256];
              snprintf(bvLabel, sizeof(bvLabel),
                       "BEFORE_viewInstall_step%d_%s", stepIdx, slot.ident.opName.c_str());
              scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                  bvLabel, executeCount_);
            }
            writeOutputSlot(si, newView, "view-install");
            if (sd::Environment::getInstance().isDebug()) {
              char avLabel[256];
              snprintf(avLabel, sizeof(avLabel),
                       "AFTER_viewInstall_step%d_%s", stepIdx, slot.ident.opName.c_str());
              scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                  avLabel, executeCount_);
            }
            DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                                newView != nullptr && newView->dataBuffer() != nullptr
                                    ? newView->dataBuffer()->getLenInBytes()
                                    : 0,
                                stream, "view-install");
            slotIsViewProducer_[si] = true;
            auto& ctx2 = *contextPool_[stepIdx];
            ctx2.setOutputArray(0, newView);
            ctx2.setInputArray(0, input0);
            outputSlots_[si] = newView;
            // Keep ownership metadata pointing at the new DataBuffer so
            // validateLifecycleForPhase sees the refreshed buffer identity.
            if (slotOwnership_ != nullptr) {
              slotOwnership_[si].dataBuffer = newView->dataBuffer();
            }
            // Slot-write site: frozen view install. Output slot pointer has
            // been updated to the new view; bump generation so readers can
            // detect a fresh view rotation.
            slot.bumpGeneration();
            dirtySlotGenerations_[si] = currentDirtyGeneration_;
            // Trace only during warmup — steady-state skips string formatting
            if (executeCount_ < 2) {
              NDArray* tracedInputs[1] = {input0};
              NDArray* tracedOutputs[1] = {newView};
              traceSmallControlSlotIO("frozen-view-install", stepIdx, slot,
                                      input0, newView,
                                      executeCount_, planLifecycle_.displayName());
              traceSmallControlSlotTensors("frozen-view-install", stepIdx, slot,
                                           tracedInputs, 1, tracedOutputs, 1,
                                           executeCount_, planLifecycle_.displayName());
            }
            backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
            return Status::OK;
          }
          // VIEW_NOT_POSSIBLE — fall through to frozen fallback / normal execution below
        }
      }
    }

    // Fallback: view-capable op in frozen path but view fast path didn't handle it.
    // If input is non-empty but cached outputs are empty, re-infer via normal path.
    if (slot.flags.isViewCapableOp && slot.wiring.numInputs >= 1) {
      int srcIdx0 = slot.wiring.inputSourceIndices[0];
      NDArray* inp0 = nullptr;
      if (srcIdx0 >= 0 && srcIdx0 < totalOutputSlots_) inp0 = outputSlots_[srcIdx0];
      else if (srcIdx0 < 0) {
        int extIdx = -(srcIdx0 + 1);
        if (extIdx < numExt) inp0 = externalArrays[extIdx];
      }
      if (inp0 != nullptr && inp0->hasValidShapeInfo() && !inp0->isEmpty() && inp0->lengthOf() > 0) {
        // Check if all frozen outputs are empty
        bool allEmpty = true;
        for (int oi = 0; oi < slot.wiring.numOutputs; oi++) {
          int si = slot.wiring.outputSlotIndices[oi];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr &&
              outputSlots_[si]->hasValidShapeInfo() && !outputSlots_[si]->isEmpty()) {
            allEmpty = false;
            break;
          }
        }
        if (allEmpty) {
          if (executeCount_ > 1) {
            DSP_DIAG_SLOT(VERIFY, stepIdx,
                "BUG: Frozen outputs empty post-warmup at slot %d (%s) — persistence bug. executeCount=%d",
                stepIdx, slot.ident.opName.c_str(), executeCount_);
          }
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "view-capable slot %d (%s): frozen outputs empty but input non-empty "
              "(len=%lld) — re-inferring via normal path (fallback, warmup)",
              stepIdx, slot.ident.opName.c_str(), inp0->lengthOf());
          slot.slotPhase.reset();  // PRIMARY
          slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
          break;  // → normalExecution
        }
      }
    }

    auto& ctx = *contextPool_[stepIdx];

    // Refresh ALL inputs from current outputSlots_[].
    // Gap ops (shape_of, gather, matmul) re-execute each step and write new
    // arrays to outputSlots_[]. Without refreshing ALL slot-sourced inputs,
    // this frozen op's context retains stale pointers from warmup.
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
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

    // Log attention op inputs for frozen tracking / lineage debugging.
    // Only during warmup — in steady-state (executeCount_ >= 2), attention ops
    // execute with the same shapes every step. Per-step logging adds overhead
    // from getOpName(), strcmp, and per-input formatting.
    if (executeCount_ < 2 && DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
      const char* opName = slot.ident.op->getOpName()->c_str();
      if (strcmp(opName, "onnx_multi_head_attention") == 0 || strcmp(opName, "dot_product_attention_v2") == 0) {
        int nIn = (int)ctx.fastpath_in().size();
        DSP_DIAG(EXECUTE, "ATTN_OP_INPUTS: %s step=%d slot=%d numInputs=%d planInputs=%d",
                 opName, executeCount_, stepIdx, nIn, slot.wiring.numInputs);
        int detailLimit = std::min(nIn, sd::graph::DspDiagnostics::getInstance().diagDetailLimit());
        for (int ai = 0; ai < detailLimit; ai++) {
          NDArray* inp = ctx.fastpath_in()[ai];
          DSP_DIAG(EXECUTE, "ATTN_OP_INPUT[%d]: ptr=%p type=%d len=%lld",
                   ai, inp, inp ? (int)inp->dataType() : -1,
                   (inp && inp->lengthOf() > 0) ? (long long)inp->lengthOf() : -1);
        }
      }
    }

    // ── DSP debug: mirror standard path's isDebugAndVerbose output + slot info ──
    // Only during warmup — in frozen steady-state (executeCount_ >= 2), this block
    // calls syncToHost() + asString() + heap allocation for EVERY input of EVERY
    // frozen op per step. For a 2743-slot plan this is thousands of syncToHost
    // round-trips and string allocations per decode step, even when isDebugAndVerbose
    // is false (the branch still evaluates the condition).
    if (executeCount_ < 2 && Environment::getInstance().isDebugAndVerbose()) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
                    "DSP_SLOT op=%s numInputs=%d numOutputs=%d",
                    slot.ident.opName.c_str(), slot.wiring.numInputs, slot.wiring.numOutputs);
      auto& ctxIns = ctx.fastpath_in();
      for (int i = 0; i < (int)ctxIns.size() && i < slot.wiring.numInputs; i++) {
        auto* array = ctxIns[i];
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (array != nullptr) {
          auto shape = ShapeUtils::shapeAsString(array);
          auto type = DataTypeUtils::asString(array->dataType());
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
                        "  input[%d] shape=%s dtype=%s src=%d",
                        i, shape.c_str(), type.c_str(), srcIdx);
        } else {
          DSP_DIAG_SLOT(EXECUTE, stepIdx, "  input[%d] NULL src=%d", i, srcIdx);
        }
      }
    }

    // Ensure CUDA host↔device coherency before op execution.
    // The standard path gets this via NativeOps entry points (prepareSpecialUse/registerSpecialUse).
    // DSP dispatches ops directly — without this, input arrays may have stale GPU data
    // and output actuality counters won't be ticked, causing compounding errors.
    //
    // PERFORMANCE: In frozen steady-state (executeCount_ >= 2), all data is already
    // device-resident and actuality flags are correct. Skipping these calls eliminates
    // ~5486 syncToDevice calls per decode step (2743 ops × 2 calls).
    //
    // forceSync_: Pre-capture warmup at exec=2+ sets this to force sync even when
    // executeCount_ >= 2. The warmup needs coherency because prior segments'
    // composite captures may have changed actuality flags.
    bool needsSync = forceSync_ || planLifecycle_.isSlotBySlot() || executeCount_ < 2;
    if (needsSync) {
        NDArray::prepareSpecialUse(ctx.fastpath_out(), ctx.fastpath_in());
    } else {
#if defined(SD_CUDA)
      // Assertion 4: Actuality flag consistency when sync is skipped.
      // When needsSync is false (frozen steady-state), we rely on the device
      // being the authoritative copy for ALL inputs. If any input has
      // isSpecialActual()==false, the device data is stale but we're about to
      // execute the op without syncing — this will silently read wrong values.
      //
      // This fires as DSP_DIAG ERROR (always logged when MEMORY category enabled)
      // because it indicates a missing tickWriteDevice() call somewhere upstream,
      // typically a missed registerSpecialUse() site after a CUDA graph island.
      auto& fpin_assert = ctx.fastpath_in();
      for (size_t ii = 0; ii < fpin_assert.size(); ii++) {
        NDArray* a = fpin_assert[ii];
        if (a == nullptr) continue;
        DataBuffer* db = a->dataBuffer();
        if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
        bool sAct = db->isSpecialActual();
        if (!sAct) {
          // isSpecialActual()==false with sync skipped: device data is stale.
          // The op will read whatever was last in device memory, not the
          // current host value. Log as ERROR so it's visible even at low
          // diagnostic verbosity.
          DSP_DIAG(MEMORY,
                   "ACTUALITY_SYNC_SKIP_ERROR: slot=%d op=%s inIdx=%zu db=%p len=%lld "
                   "sAct=0 pAct=%d exec=%d — sync SKIPPED but device data is NOT actual; "
                   "op will read stale device values. Missing tickWriteDevice upstream?",
                   stepIdx, slot.ident.opName.c_str(), ii, (void*)db,
                   (long long)db->getLenInBytes(),
                   db->isPrimaryActual() ? 1 : 0, executeCount_);
          REQUIRE_TRUE(sAct, 0,
                       "ACTUALITY_SYNC_SKIP_ERROR: input not device-actual when sync skipped "
                       "for slot %d (%s).", stepIdx, slot.ident.opName.c_str());
        }
      }
#endif  // SD_CUDA

      if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_MEMORY)) {
        // Existing SYNC_SKIP_ANOMALY_IN trace (pAct=1 AND sAct=0 is the interesting case
        // where the host has newer data that might incorrectly override device state later).
        auto& fpin = ctx.fastpath_in();
        for (size_t ii = 0; ii < fpin.size(); ii++) {
          NDArray* a = fpin[ii];
          if (a == nullptr) continue;
          DataBuffer* db = a->dataBuffer();
          if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
          bool pAct = db->isPrimaryActual();
          bool sAct = db->isSpecialActual();
          if (pAct && !sAct) {
            DSP_DIAG(MEMORY,
                     "SYNC_SKIP_ANOMALY_IN: slot=%d op=%s inIdx=%zu db=%p len=%lld "
                     "pAct=1 sAct=0 exec=%d — host poisoned device-authoritative buffer",
                     stepIdx, slot.ident.opName.c_str(), ii, (void*)db,
                     (long long)db->getLenInBytes(), executeCount_);
          }
        }
      }
    }

    // Skip shape inference in frozen steady-state for the frozen context path.
    if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 3) {
      ctx.setShapeFunctionOverride(true);
    }

    // ── Frozen context output dtype validation ─────────────────────────
    // Validate that the output arrays in this frozen context have dtypes
    // consistent with the CURRENT inputs.  The cachedOutputShapes may have
    // stale dtypes from a prior execution (e.g. FLOAT32 from matmul dtype
    // promotion when inputs temporarily had mixed types).
    //
    // This is diagnostic-only: segDispatchWarmup demotes FROZEN→WARMUP before
    // warmup execution, preventing stale frozen contexts from firing during
    // the critical warmup phase.  In steady-state replay, dtypes should be
    // stable.  Log mismatches as errors for debugging.
    if (executeCount_ < 2 && DSP_DIAG_ENABLED(SHAPE) &&
        slot.shapeCacheValid() && !slot.shapeCache.cachedOutputShapes.empty()) {
      DataType expectedDt = DataType::UNKNOWN;
      auto& fpIn = ctx.fastpath_in();
      for (int i = 0; i < static_cast<int>(fpIn.size()); i++) {
        if (fpIn[i] != nullptr) {
          expectedDt = fpIn[i]->dataType();
          break;
        }
      }
      if (expectedDt != DataType::UNKNOWN && DataTypeUtils::isR(expectedDt)) {
        auto& fpOut = ctx.fastpath_out();
        for (int i = 0; i < slot.wiring.numOutputs &&
                         i < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) &&
                         i < static_cast<int>(fpOut.size()); i++) {
          NDArray* outArr = fpOut[i];
          const LongType* cachedShape = slot.shapeCache.cachedOutputShapes[i];
          if (cachedShape == nullptr) continue;
          auto cachedDt = ArrayOptions::dataType(cachedShape);
          auto arrDt = (outArr != nullptr) ? outArr->dataType() : DataType::UNKNOWN;
          bool cachedBad = DataTypeUtils::isR(cachedDt) && cachedDt != expectedDt;
          bool outputBad = outArr != nullptr && DataTypeUtils::isR(arrDt) && arrDt != expectedDt;
          if (cachedBad || outputBad) {
            DSP_DIAG(SHAPE,
                "FROZEN_CTX_DTYPE_MISMATCH: slot %d (%s) output[%d] "
                "arrDtype=%s cachedDtype=%s inputDtype=%s (execCount=%d)",
                stepIdx, slot.ident.opName.c_str(), i,
                outArr ? DataTypeUtils::asString(arrDt).c_str() : "null",
                DataTypeUtils::asString(cachedDt).c_str(),
                DataTypeUtils::asString(expectedDt).c_str(), executeCount_);
          }
        }
      }
    }

    // Pre-execute corruption scan (frozen context path)
    if (sd::Environment::getInstance().isDebug()) {
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          if (!validateSlotShapeBufferAlignment(outputSlots_[si], si,
                  "BEFORE_frozen_ctx_execute", executeCount_)) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "CORRUPTION_BEFORE_EXEC_FC: slot %d (%s) output[%d] at outputSlot %d "
                     "already corrupt BEFORE frozen-ctx op->execute. execCount=%d",
                     stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
            THROW_EXCEPTION(msg);
          }
        }
      }
    }

    Status status = Status::OK;
    if (shapeOnlyMode_) {
      DSP_DIAG(EXECUTE,
               "SHAPE_ONLY: skipping op execution for slot %d (%s) [frozen context path]",
               stepIdx, slot.ident.opName.c_str());
    } else {
      prezeroThisSlot();
      try {
        status = slot.ident.op->execute(&ctx);
      } catch (const std::exception& e) {
        std::string inputShapes, outputShapesStr, iArgsStr;
        dspFormatSlotExecContext(slot, ctx.fastpath_in().data(), slot.wiring.numInputs,
                                 ctx.fastpath_out().data(), slot.wiring.numOutputs,
                                 inputShapes, outputShapesStr, iArgsStr);
        DSP_DIAG(EXECUTE,
                 "SLOT EXEC EXCEPTION [frozen-ctx]: slot %d (%s) exception='%s', "
                 "inputs=[%s] outputs=[%s] iArgs=[%s] execCount=%d",
                 stepIdx, slot.ident.opName.c_str(), e.what(),
                 inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
                 executeCount_);
        dspPropagateSlotError(stepIdx, slot,
                              (std::string("frozen-ctx exec exception: ") + e.what()).c_str(),
                              inputShapes, outputShapesStr, iArgsStr);
        if (slot.flags.ltEpilogueType > 0) platformClearLtEpilogue();
        return Status::KERNEL_FAILURE;
      }
    }

    // Post-execute corruption scan (frozen context path)
    if (sd::Environment::getInstance().isDebug()) {
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          if (!validateSlotShapeBufferAlignment(outputSlots_[si], si,
                  "AFTER_frozen_ctx_execute", executeCount_)) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "CORRUPTION_AFTER_EXEC_FC: slot %d (%s) output[%d] at outputSlot %d "
                     "corrupted BY frozen-ctx op->execute. execCount=%d",
                     stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
            THROW_EXCEPTION(msg);
          }
        }
      }
    }

    // Full slot scan after frozen-context execute — catch cross-slot heap overruns
    if (sd::Environment::getInstance().isDebug()) {
      char fcLabel[256];
      snprintf(fcLabel, sizeof(fcLabel),
               "AFTER_frozenctx_exec_step%d_%s", stepIdx, slot.ident.opName.c_str());
      scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
          fcLabel, executeCount_);
    }

    if (needsSync) {
        NDArray::registerSpecialUse(ctx.fastpath_out(), ctx.fastpath_in());
    } else if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_MEMORY)) {
      // Trace post-exec output state. When registerSpecialUse is skipped, the
      // output counters aren't ticked — if the op wrote to host rather than
      // device (pAct=1/sAct=0 post-exec) we MUST know, since subsequent reads
      // will trigger a bad syncToDevice.
      auto& fpout = ctx.fastpath_out();
      for (size_t oi = 0; oi < fpout.size(); oi++) {
        NDArray* a = fpout[oi];
        if (a == nullptr) continue;
        DataBuffer* db = a->dataBuffer();
        if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
        bool pAct = db->isPrimaryActual();
        bool sAct = db->isSpecialActual();
        if (!sAct) {
          DSP_DIAG(MEMORY,
                   "SYNC_SKIP_ANOMALY_OUT: slot=%d op=%s outIdx=%zu db=%p len=%lld "
                   "pAct=%d sAct=0 exec=%d — op did not tick device-write counter",
                   stepIdx, slot.ident.opName.c_str(), oi, (void*)db,
                   (long long)db->getLenInBytes(), (int)pAct, executeCount_);
        }
      }
    }

    auto& ctxOuts = ctx.fastpath_out();

    // Per-slot output fingerprint — active when debug+verbose at exec >= 2
    if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose()) {
      for (int i = 0; i < slot.wiring.numOutputs && i < (int)ctxOuts.size(); i++) {
        auto* array = ctxOuts[i];
        if (array != nullptr && !array->isEmpty() && array->lengthOf() > 0) {
          array->syncToHost();
          double sum = 0.0;
          double firstVal = array->e<double>(0);
          sd::LongType len = array->lengthOf();
          sd::LongType fpLen = sd::math::sd_min(64LL, len);
          for (sd::LongType fi = 0; fi < fpLen; fi++) {
            sum += array->e<double>(fi);
          }
          DSP_DIAG(VERIFY,
                   "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld first=%.8g sum64=%.8g",
                   executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                   ShapeUtils::shapeAsString(array).c_str(),
                   DataTypeUtils::asString(array->dataType()).c_str(),
                   (long long)len, firstVal, sum);
        }
      }
    }

    for (int i = 0; i < slot.wiring.numOutputs && i < static_cast<int>(ctxOuts.size()); i++) {
      if (ctxOuts[i] != nullptr) {
        // Post-execution output integrity check (normal path)
        auto* postSib = ctxOuts[i]->shapeInfoConstBuffer();
        if (postSib != nullptr) {
          uintptr_t postSibAddr = reinterpret_cast<uintptr_t>(postSib);
          if (postSibAddr % alignof(ConstantShapeBuffer) != 0) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "DSP CORRUPTION DETECTED: slot %d (%s) output[%d] _shapeInfoBuffer=%p "
                     "is misaligned AFTER op execution (alignOffset=%zu). "
                     "The op itself corrupted this output's shape buffer. "
                     "execCount=%d shapesFrozen=%d",
                     stepIdx, slot.ident.opName.c_str(), i, (void*)postSib,
                     static_cast<size_t>(postSibAddr % alignof(ConstantShapeBuffer)),
                     executeCount_, shapesFrozen_ ? 1 : 0);
            THROW_EXCEPTION(msg);
          }
        }
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          NDArray* expectedOutput = outputSlots_[si];
          if (expectedOutput != nullptr && expectedOutput->hasValidShapeInfo() &&
              !outputWrapperMatchesExpectedShape(ctxOuts[i], expectedOutput->shapeInfo())) {
            DSP_DIAG(SHAPE,
                     "CTX_OUTPUT_REJECT(frozen): slot=%d op=%s outIdx=%d expected=%s actual=%s",
                     stepIdx, slot.ident.opName.c_str(), i,
                     ShapeUtils::shapeAsString(expectedOutput).c_str(),
                     ShapeUtils::shapeAsString(ctxOuts[i]).c_str());
            continue;
          }

          platformReconcileOutputActuality("frozen-op-exec", stepIdx, slot, ctxOuts[i]);
          writeOutputSlot(si, ctxOuts[i], "frozen-op-exec");
          DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                              ctxOuts[i]->dataBuffer() != nullptr
                                  ? ctxOuts[i]->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "frozen-op-exec");
          outputSlots_[si] = ctxOuts[i];
        }
      }
    }

    platformLogSlotOutput(stepIdx, slot.ident.opName.c_str(), "OP_EXEC(frozen)",
                          slot.wiring.outputSlotIndices, slot.wiring.numOutputs);

    if (status == Status::OK) {
      // Slot-write site: frozen-op-exec heavy path. The kernel has finished
      // and all output slots have been updated. Bump generation so any later
      // reader can detect a stale snapshot taken before this execution.
      slot.bumpGeneration();
      // Trace only during warmup — in steady-state the fast frozen path handles
      // non-view ops and doesn't call these. The heavy path only runs for view
      // fallback ops in steady-state, which don't need per-step control tracing.
      if (executeCount_ < 2) {
        NDArray* tracedInput0 = slot.wiring.numInputs > 0 && !ctx.fastpath_in().empty()
                                    ? ctx.fastpath_in()[0]
                                    : nullptr;
        NDArray* tracedOutput0 = slot.wiring.numOutputs > 0 && !ctxOuts.empty()
                                     ? ctxOuts[0]
                                     : nullptr;
        traceSmallControlSlotIO("frozen-op-exec", stepIdx, slot,
                                tracedInput0, tracedOutput0,
                                executeCount_, planLifecycle_.displayName());
        traceSmallControlSlotTensors("frozen-op-exec", stepIdx, slot,
                                     ctx.fastpath_in().data(),
                                     static_cast<int>(ctx.fastpath_in().size()),
                                     ctxOuts.data(),
                                     static_cast<int>(ctxOuts.size()),
                                     executeCount_, planLifecycle_.displayName());
      }
      backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
    }
    return status;
  }

  } while (false);  // end fused chain + frozen context block

  // ── Step 1: Gather inputs ────────────────────────────────────────────────
  static thread_local std::vector<NDArray*> inputs;
  inputs.resize(slot.wiring.numInputs);
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx >= 0) {
      inputs[i] = outputSlots_[srcIdx];
    } else {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) {
        inputs[i] = externalArrays[extIdx];
      } else {
        inputs[i] = nullptr;
      }
    }

    if (inputs[i] == nullptr) {
      // Enhanced diagnostic: identify which step produces the missing output slot
      // and whether it's a self-reference, forward-reference, or simply unpopulated.
      const char* srcOpName = "?";
      int srcStep = -1;
      bool isSelfRef = false;
      bool isForwardRef = false;
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        // Find which step produces this output slot
        for (int fs = 0; fs < numSlots_; fs++) {
          for (int fo = 0; fo < slots_[fs].wiring.numOutputs; fo++) {
            if (slots_[fs].wiring.outputSlotIndices[fo] == srcIdx) {
              srcStep = fs;
              srcOpName = slots_[fs].ident.opName.c_str();
              break;
            }
          }
          if (srcStep >= 0) break;
        }
        // Check for self-reference (this step's output slot is also its input source)
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          if (slot.wiring.outputSlotIndices[o] == srcIdx) {
            isSelfRef = true;
            break;
          }
        }
        isForwardRef = (srcStep > stepIdx);
      }
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
                "NULL input for slot %d (%s) input %d, srcIdx=%d "
                "producerStep=%d producerOp=%s selfRef=%d forwardRef=%d "
                "execCount=%d frozen=%d planAddr=%p planFP=0x%016llx "
                "outputSlot[srcIdx]=%p",
                stepIdx, slot.ident.opName.c_str(), i, slot.wiring.inputSourceIndices[i],
                srcStep, srcOpName, isSelfRef ? 1 : 0, isForwardRef ? 1 : 0,
                executeCount_, shapesFrozen_ ? 1 : 0,
                (void*)this, (unsigned long long)identityFingerprint_,
                (srcIdx >= 0 && srcIdx < totalOutputSlots_) ? (void*)outputSlots_[srcIdx] : nullptr);
      return Status::BAD_INPUT;
    }

    // Validate shapeInfo is present — an NDArray with nullptr _shapeInfo is
    // uninitialized or destroyed. Using it would crash inside op->execute().
    // This catches use-after-free and stale cache entries early.
    {
      if (!inputs[i]->hasValidShapeInfo()) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
            "NULL shapeInfo for slot %d (%s) input %d, srcIdx=%d "
            "ptr=%p db=%p exec=%d",
            stepIdx, slot.ident.opName.c_str(), i, slot.wiring.inputSourceIndices[i],
            (void*)inputs[i], (void*)inputs[i]->dataBuffer(), executeCount_);
        // A null shapeInfo is an invariant violation: slots are immutable once bound.
        // Silently nulling the cache entry is a workaround that hides use-after-free
        // or stale-pointer bugs in the caller.  Throw a hard error instead so the
        // root cause is visible immediately.
        DSP_THROW(EXECUTE,
                 "NativeDynamicShapePlan: null shapeInfo on bound input slot (slot=%d op=%s "
                 "input=%d srcIdx=%d ptr=%p db=%p exec=%d). "
                 "Slots are immutable once bound; a null shapeInfo indicates a use-after-free "
                 "or stale pointer — fix the producer of this slot rather than nulling the cache.",
                 stepIdx, slot.ident.opName.c_str(), i, slot.wiring.inputSourceIndices[i],
                 (void*)inputs[i], (void*)inputs[i]->dataBuffer(), executeCount_);
        return Status::BAD_INPUT;  // unreachable, but keeps return-type analysis clean
      }
    }

    // Validate GPU buffer is alive — detect freed DataBuffers that would cause
    // CUDA illegal memory access (error 700) downstream.
    auto* db = inputs[i]->dataBuffer();
    if (db != nullptr && db->isClosed()) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
          "CLOSED DataBuffer for slot %d (%s) input %d, srcIdx=%d "
          "specialBuf=%p isConst=%d exec=%d",
          stepIdx, slot.ident.opName.c_str(), i, slot.wiring.inputSourceIndices[i],
          db->special(), db->isConstant ? 1 : 0, executeCount_);
      return Status::BAD_INPUT;
    }
    // Validate platform buffer (GPU special buffer or CPU primary buffer)
    if (!platformValidateSlotInputBuffer(stepIdx, slot, i, inputs[i])) {
      return Status::BAD_INPUT;
    }
  }


  // ── Step 2: Shape inference ──────────────────────────────────────────────
  LongType shapeKey = 0;
  bool cacheHit;
  if (!planLifecycle_.isSlotBySlot() && executeCount_ > 0 && slot.shapeCacheValid() &&
      !slot.flags.outputShapeDependsOnInputValues) {
    // ALWAYS compute the shape key — skipping this masks shape drift and
    // hides corruption. The +1 byte _shapeInfoBuffer corruption was invisible
    // because forced cache hits prevented shape mismatch detection.
    shapeKey = computeShapeKey(slot, inputs.data(), slot.wiring.numInputs);
    cacheHit = (slot.shapeCache.cachedShapeKey == shapeKey);

    // View-capable ops: check for stale empty cached shapes.
    // During Step 0 (first execution), KV cache inputs are empty, so shape
    // inference produces empty output shapes that get cached. On subsequent
    // executions the inputs become non-empty (concat grows KV), but
    // shapesFrozen_ prevents re-inference. Detect this and force a cache miss.
    if (cacheHit && slot.flags.isViewCapableOp && !slot.shapeCache.cachedOutputShapes.empty()) {
      bool allCachedEmpty = true;
      for (const auto& s : slot.shapeCache.cachedOutputShapes) {
        if (!shape::isEmpty(const_cast<LongType*>(s))) {
          allCachedEmpty = false;
          break;
        }
      }
      if (allCachedEmpty && slot.wiring.numInputs > 0 && inputs[0] != nullptr &&
          !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "view-capable slot %d (%s): cached shapes ALL empty but input[0] "
            "non-empty (len=%lld) — forcing shape re-inference",
            stepIdx, slot.ident.opName.c_str(), inputs[0]->lengthOf());
        cacheHit = false;
      }
    }
  } else {
    shapeKey = computeShapeKey(slot, inputs.data(), slot.wiring.numInputs);
    cacheHit = slot.shapeCacheValid() && (slot.shapeCache.cachedShapeKey == shapeKey);

    // View-capable ops: even with matching shape key, check for stale empty
    // cached shapes. computeShapeKey does NOT include the ARRAY_EMPTY flag,
    // so an input that transitions from empty (ARRAY_EMPTY) to non-empty with
    // the same dimensions produces an identical key but needs re-inference.
    if (cacheHit && slot.flags.isViewCapableOp && !slot.shapeCache.cachedOutputShapes.empty()) {
      bool allCachedEmpty = true;
      for (const auto& s : slot.shapeCache.cachedOutputShapes) {
        if (!shape::isEmpty(const_cast<LongType*>(s))) {
          allCachedEmpty = false;
          break;
        }
      }
      if (allCachedEmpty && slot.wiring.numInputs > 0 && inputs[0] != nullptr &&
          !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "view-capable slot %d (%s): cache hit but shapes ALL empty while "
            "input[0] non-empty (len=%lld) — forcing re-inference (key match)",
            stepIdx, slot.ident.opName.c_str(), inputs[0]->lengthOf());
        cacheHit = false;
      }
    }
  }

  std::vector<const LongType*> outputShapes;
  if (cacheHit) {
    outputShapes = slot.shapeCache.cachedOutputShapes;
  } else {
    bool allowRestabilization =
        !planLifecycle_.isSlotBySlot() && executeCount_ > 0 && slot.shapeCacheValid() &&
        allowFrozenShapeRestabilization(stepIdx, slot, getPlanPhase(),
                                        slots_, numSlots_,
                                        outputSlots_, totalOutputSlots_,
                                        externalArrays, numExt);

    // ── Phase violation: shape change during SHAPES_FROZEN ──
    // If shapes are frozen and we get a cache miss (shape changed), this is
    // a phase contract violation. Return a hard error — the caller's frozen
    // assumption is broken and continuing would produce wrong results or crash.
    if (!planLifecycle_.isSlotBySlot() && executeCount_ > 0 && slot.shapeCacheValid()) {
      // For value-dependent ops, shape key changes are EXPECTED because the key
      // hashes input VALUES (which change per step). But the OUTPUT SHAPE should
      // stay the same (shapes are frozen). Recompute shapes and verify they match.
      if (slot.flags.outputShapeDependsOnInputValues) {
        // Value-dep op: key changed but shapes may be the same. Recompute and check.
        // Update the cached key to the new value so subsequent checks don't re-fire.
        slot.shapeCache.cachedShapeKey = shapeKey;
        DSP_DIAG(SHAPE, "VALUE_DEP_KEY_UPDATE: slot %d (%s) shape key updated "
                 "(value-dep op, input values changed but shapes frozen)",
                 stepIdx, slot.ident.opName.c_str());
        // Fall through to shape recomputation below — it will verify actual shapes match
      } else if (allowRestabilization) {
        DSP_DIAG(SHAPE,
                 "FROZEN_RESTABILIZE: slot %d (%s) cache miss during SHAPES_FROZEN "
                 "is allowed because the shape-control chain is plan-internal and "
                 "planPhase=%s (< SEALED)",
                 stepIdx, slot.ident.opName.c_str(), planLifecycle_.displayName());
      } else if (hasTransitiveDynamicUpstream(slots_, numSlots_, stepIdx)) {
        // Slot transitively depends on external placeholder inputs whose
        // shapes/dtypes change between prefill and decode. Shape key change
        // is expected — update key and allow shape recomputation. The output
        // shapes will be updated below after calculateOutputShape runs.
        slot.shapeCache.cachedShapeKey = shapeKey;
        // Allow restabilization so the recomputed shapes replace the stale
        // cached shapes (which may have a different dtype due to type
        // promotion with mixed-dtype inputs during prefill).
        allowRestabilization = true;
        DSP_DIAG(SHAPE,
                 "EXT_INPUT_SHAPE_CHANGE: slot %d (%s) shape key changed because "
                 "slot transitively depends on external inputs (execCount=%d) — "
                 "allowing shape restabilization",
                 stepIdx, slot.ident.opName.c_str(), executeCount_);
      } else if (planLifecycle_.compilationDone) {
        // Post-seal shape change: external input shapes changed after the plan
        // was sealed. This is a mid-execution compile event — the sealed
        // assumption is broken, but we can recover by re-running shape inference
        // and letting the allocation path below replace the stale cached output
        // arrays. Record the event in the counter so benchmarks can detect it,
        // then fall through to shape recomputation.
        recordMidExecutionCompile(stepIdx, stepIdx,
                                  "slot-by-slot shape change after seal");
        DSP_DIAG(COMPILE,
                 "POST_SEAL_SHAPE_CHANGE: slot %d (%s) shape key changed after "
                 "compilation seal (oldKey=0x%llx newKey=0x%llx). Recomputing "
                 "shapes and reallocating outputs.",
                 stepIdx, slot.ident.opName.c_str(),
                 (long long)slot.shapeCache.cachedShapeKey, (long long)shapeKey);
      } else {
        // Non-value-dep op: shape key change means actual shapes changed → violation
        char errBuf[512];
        snprintf(errBuf, sizeof(errBuf),
                 "LIFECYCLE_ERROR: shape changed at slot %d (%s) during SHAPES_FROZEN phase "
                 "(execCount=%d). Shapes were assumed constant but input shapes changed. "
                 "oldKey=0x%llx newKey=0x%llx. Unfreeze shapes before changing input shapes.",
                 stepIdx, slot.ident.opName.c_str(), executeCount_,
                 (long long)slot.shapeCache.cachedShapeKey, (long long)shapeKey);
        DSP_DIAG(SHAPE, "%s", errBuf);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
            static_cast<int>(Status::KERNEL_FAILURE));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errBuf);
        return Status::KERNEL_FAILURE;
      }
    }

    auto& ctx = *contextPool_[stepIdx];

    for (int i = 0; i < slot.wiring.numInputs; i++) {
      ctx.setInputArray(i, inputs[i]);
    }

    if (slot.args.numIArgs > 0) ctx.setIArguments(slot.args.iArgs, slot.args.numIArgs);
    if (slot.args.numTArgs > 0) ctx.setTArguments(slot.args.tArgs, slot.args.numTArgs);
    if (slot.args.numBArgs > 0) ctx.setBArguments(slot.args.bArgs, slot.args.numBArgs);
    if (slot.args.numDArgs > 0) ctx.setDArguments(slot.args.dArgs, slot.args.numDArgs);
    ctx.getSArguments()->clear();
    if (slot.args.numSArgs > 0) {
      ctx.getSArguments()->insert(ctx.getSArguments()->end(), slot.args.sArgs, slot.args.sArgs + slot.args.numSArgs);
    }

    // Legacy reduce/broadcast ops read reduction dims from block.getAxis(), not iArgs.
    // The Java compiler packs dims into iArgs; mirror them into axis for these op types.
    if (legacyOpReadsAxisFromIArgs(slot.legacy.legacyOpType)) {
      ctx.getAxis()->clear();
      for (int i = 0; i < slot.args.numIArgs; i++) {
        ctx.getAxis()->emplace_back(static_cast<sd::LongType>(slot.args.iArgs[i]));
      }
    }

    ShapeList inputShapes;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (inputs[i] != nullptr) {
        inputShapes.push_back(inputs[i]->shapeInfo());
      }
    }


    ShapeList* shapeList = nullptr;
    try {
      shapeList = slot.ident.op->calculateOutputShape(&inputShapes, ctx);
    } catch (const std::exception& e) {
      DSP_DIAG_SLOT(SHAPE, stepIdx, "shape inference EXCEPTION at slot %d (%s): %s",
                stepIdx, slot.ident.opName.c_str(), e.what());
      // Propagate error detail to Java via errorReference (otherwise Java only sees "status 50")
      std::string inputShapeStr;
      for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
        if (ii > 0) inputShapeStr += ", ";
        inputShapeStr += inputs[ii] ? ShapeUtils::shapeAsString(inputs[ii]) : "null";
      }
      // Build a short srcIdx summary FIRST (won't be truncated)
      std::string srcIdxSummary;
      for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
        if (ii > 0) srcIdxSummary += ",";
        int srcIdx = slot.wiring.inputSourceIndices[ii];
        srcIdxSummary += "i" + std::to_string(ii) + "=";
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          srcIdxSummary += "ext" + std::to_string(extIdx);
        } else {
          int producerStep = findProducingStepForOutputSlot(slots_, numSlots_, srcIdx);
          if (producerStep >= 0) {
            srcIdxSummary += "s" + std::to_string(producerStep);
            srcIdxSummary += "(";
            srcIdxSummary += slots_[producerStep].ident.opName;
            srcIdxSummary += slots_[producerStep].frozenConstantSlot() ? ",FR" : "";
            srcIdxSummary += slots_[producerStep].flags.isDynamicShape ? ",DYN" : "";
            srcIdxSummary += ")";
          } else {
            srcIdxSummary += "slot" + std::to_string(srcIdx);
          }
        }
      }

      std::string smallInputValueStr;
      for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
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
      for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
        if (ii > 0) upstreamTraceStr += " | ";
        upstreamTraceStr += "input[";
        upstreamTraceStr += std::to_string(ii);
        upstreamTraceStr += "]=>";
        std::vector<int> seenSlotSteps;
        appendUpstreamControlTrace(upstreamTraceStr, slot.wiring.inputSourceIndices[ii], 3,
                                   slots_, numSlots_, outputSlots_, totalOutputSlots_,
                                   externalArrays, numExt, externalInputNames_,
                                   seenSlotSteps);
      }
      char errBuf[2048];
      snprintf(errBuf, sizeof(errBuf),
               "slot %d (%s) shape inference failed: %s | inputs=[%s] iArgs=%d frozenConst=%d state=%s execCount=%d src=[%s]",
               stepIdx, slot.ident.opName.c_str(), e.what(), inputShapeStr.c_str(), slot.args.numIArgs,
               slot.frozenConstantSlot() ? 1 : 0, slot.slotPhase.displayName(), executeCount_,
               srcIdxSummary.c_str());
      std::string errMsg = errBuf;
      if (!smallInputValueStr.empty()) {
        errMsg += " vals=[";
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
                stepIdx, slot.ident.opName.c_str());
      char errBuf[512];
      snprintf(errBuf, sizeof(errBuf), "slot %d (%s) shape inference returned null/empty",
               stepIdx, slot.ident.opName.c_str());
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
                  stepIdx, slot.ident.opName.c_str(), i, e.what());
        delete shapeList;
        return Status::KERNEL_FAILURE;
      }
    }

    if (!planLifecycle_.isSlotBySlot() && executeCount_ > 0 && slot.shapeCacheValid() &&
        slot.flags.outputShapeDependsOnInputValues && !slot.flags.isDynamicShape &&
        !slot.shapeCache.cachedOutputShapes.empty()) {
      bool sameOutputShapeCount = slot.shapeCache.cachedOutputShapes.size() == outputShapes.size();
      bool outputShapesMatch = sameOutputShapeCount;
      if (outputShapesMatch) {
        for (size_t i = 0; i < outputShapes.size(); i++) {
          const LongType* oldShape = slot.shapeCache.cachedOutputShapes[i];
          const LongType* newShape = outputShapes[i];
          // Compare shapes (dimensions) and data type only — NOT the full extras field.
          // reshape_no_copy can produce identical shapes with different flags
          // (ARRAY_NEEDS_COPY vs ARRAY_COPY_OFFSET_INPUT_0) depending on whether
          // reshapeNoAlloc succeeds, and those flag differences are not shape changes.
          if (oldShape == nullptr || newShape == nullptr ||
              !shape::equalsSoft(const_cast<LongType*>(oldShape),
                                 const_cast<LongType*>(newShape)) ||
              ArrayOptions::dataType(oldShape) != ArrayOptions::dataType(newShape)) {
            // Before reporting mismatch, double-check it's a real shape/type change
            // and not just a flags-only difference (e.g. ARRAY_NEEDS_COPY toggle)
            if (oldShape != nullptr && newShape != nullptr &&
                shape::equalsSoft(const_cast<LongType*>(oldShape),
                                   const_cast<LongType*>(newShape)) &&
                ArrayOptions::dataType(oldShape) == ArrayOptions::dataType(newShape)) {
              // Shapes and dtype match — this is a flags-only difference, not a shape change.
              // Update cached shape info to pick up new flags but don't error.
              continue;
            }
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
                   "planPhase=%s (< SEALED). Updating cached shapes.",
                   stepIdx, slot.ident.opName.c_str(), planLifecycle_.displayName());
        } else {
          std::string oldShapesStr;
          std::string newShapesStr;
          for (size_t i = 0; i < std::max(slot.shapeCache.cachedOutputShapes.size(), outputShapes.size()); i++) {
            if (i > 0) {
              oldShapesStr += ", ";
              newShapesStr += ", ";
            }
            oldShapesStr += (i < slot.shapeCache.cachedOutputShapes.size() && slot.shapeCache.cachedOutputShapes[i] != nullptr)
                ? ShapeUtils::shapeAsString(slot.shapeCache.cachedOutputShapes[i])
                : "null";
            newShapesStr += (i < outputShapes.size() && outputShapes[i] != nullptr)
                ? ShapeUtils::shapeAsString(outputShapes[i])
                : "null";
          }

          std::string inputShapeStr;
          for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
            if (ii > 0) inputShapeStr += ", ";
            inputShapeStr += inputs[ii] ? ShapeUtils::shapeAsString(inputs[ii]) : "null";
          }

          std::string smallInputValueStr;
          for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
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
          for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
            if (ii > 0) upstreamTraceStr += " | ";
            upstreamTraceStr += "input[";
            upstreamTraceStr += std::to_string(ii);
            upstreamTraceStr += "]=>";
            std::vector<int> seenSlotSteps;
            appendUpstreamControlTrace(upstreamTraceStr, slot.wiring.inputSourceIndices[ii], 3,
                                       slots_, numSlots_, outputSlots_, totalOutputSlots_,
                                       externalArrays, numExt, externalInputNames_,
                                       seenSlotSteps);
          }

          std::string errMsg = "LIFECYCLE_ERROR: value-dependent output shape changed at slot " +
              std::to_string(stepIdx) + " (" + slot.ident.opName + ") during SHAPES_FROZEN phase "
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
            dspDumpSlotInputs(stepIdx, slot.ident.opName.c_str(),
                              slot.wiring.numInputs, slot.wiring.inputSourceIndices, slot.wiring.inputSourceTypes,
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
        slot.shapeCache.cachedShapeKey = shapeKey;
        slot.shapeCache.cachedOutputShapes = outputShapes;
        if (!slot.slotPhase.shapeCacheValid) {
          slot.slotPhase.markShapeCached();  // PRIMARY
          slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
        }
      } else {
        slot.shapeCache.cachedShapeKey = shapeKey;
        outputShapes = slot.shapeCache.cachedOutputShapes;
      }
    } else {
      slot.shapeCache.cachedShapeKey = shapeKey;
      slot.shapeCache.cachedOutputShapes = outputShapes;
      if (!slot.slotPhase.shapeCacheValid) {
        slot.slotPhase.markShapeCached();  // PRIMARY
        slot.state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
      }
    }

    delete shapeList;
  }

  // ── Step 3: Allocate/reuse outputs ───────────────────────────────────────
  // Use the shape function's output count (not the graph wiring count) so the
  // op sees the same outputWidth() as InferenceSession. Extra outputs beyond
  // numWiredOutputs are untracked scratch arrays.
  int numShapeFnOutputs = static_cast<int>(outputShapes.size());
  int numWiredOutputs = slot.wiring.numOutputs;
  int numActualOutputs = numShapeFnOutputs;
  static thread_local std::vector<NDArray*> outputs;
  outputs.resize(numActualOutputs);
  // Zero-init all entries (resize doesn't zero for pointers)
  for (int i = 0; i < numActualOutputs; i++) outputs[i] = nullptr;
  bool outputsReady = false;

  // ── In-place fusion: reuse input buffer as output ──
  if (slot.flags.inPlaceFused && slot.flags.inPlaceFusedInputIdx >= 0 &&
      slot.flags.inPlaceFusedInputIdx < slot.wiring.numInputs && numActualOutputs >= 1) {
    NDArray* inPlaceBuffer = inputs[slot.flags.inPlaceFusedInputIdx];
    if (inPlaceBuffer != nullptr) {
      const LongType* expectedShape = outputShapes[0];
      if (shape::equalsSoft(inPlaceBuffer->shapeInfo(), expectedShape) &&
          ArrayOptions::dataType(inPlaceBuffer->shapeInfo()) == ArrayOptions::dataType(expectedShape)) {
        outputs[0] = inPlaceBuffer;
        int slotIdx = slot.wiring.outputSlotIndices[0];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          writeOutputSlot(slotIdx, inPlaceBuffer, "in-place-fused");
          DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                              inPlaceBuffer != nullptr && inPlaceBuffer->dataBuffer() != nullptr
                                  ? inPlaceBuffer->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "in-place-fused");
        }

        for (int i = 1; i < numActualOutputs; i++) {
          int si = (i < numWiredOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
          if (si < 0) {
            int cacheIdx = stepIdx * MAX_OUTPUTS_PER_SLOT + i;
            if (cacheIdx < untrackedOutputCacheSize_) {
              NDArray* cached = untrackedOutputCache_[cacheIdx];
              if (cached != nullptr && cached->hasValidShapeInfo() &&
                  shape::equalsSoft(cached->shapeInfo(), outputShapes[i]) &&
                  ArrayOptions::dataType(cached->shapeInfo()) == ArrayOptions::dataType(outputShapes[i])) {
                size_t requiredBytes = static_cast<size_t>(shape::length(outputShapes[i])) *
                                       DataTypeUtils::sizeOf(ArrayOptions::dataType(outputShapes[i]));
                auto* cachedDb = cached->dataBuffer();
                size_t availableBytes = (cachedDb != nullptr) ? cachedDb->getLenInBytes() : 0;
                if (requiredBytes <= availableBytes) {
                  outputs[i] = cached;
                  continue;
                }
                DSP_DIAG(MEMORY,
                         "UNTRACKED_CACHE_SKIP: step %d output %d buffer too small "
                         "(required=%zu available=%zu) — reallocating",
                         stepIdx, i, requiredBytes, availableBytes);
              }
              delete cached;
              untrackedOutputCache_[cacheIdx] = nullptr;
              if (sd::Environment::getInstance().isDebug()) {
                scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                    "AFTER_untrackedCache1_delete", executeCount_);
              }
            }
            outputs[i] = new NDArray(const_cast<LongType*>(outputShapes[i]), true);
            if (cacheIdx < untrackedOutputCacheSize_) {
              untrackedOutputCache_[cacheIdx] = outputs[i];
            }
            continue;
          }
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          outputs[i] = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
          writeOutputSlot(si, outputs[i], "alloc-output");
          DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                              outputs[i]->dataBuffer() != nullptr
                                  ? outputs[i]->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "alloc-output");
          outputSlots_[si] = outputs[i];
        }

        outputsReady = true;
      }
    }
  }

  // ── View-capable ops: share input 0's DataBuffer for output 0 ──────────
  if (!outputsReady && slot.flags.isViewCapableOp && slot.wiring.numInputs >= 1 && numActualOutputs >= 1) {
    NDArray* input0 = inputs[0];
    if (input0 != nullptr) {
      NDArray* view = nullptr;
      LongType viewOffset = 0;
      ViewCreateResult vcr = tryCreateViewForSlot(
          stepIdx, slot, input0, outputShapes[0],
          inputs.data(), slot.wiring.numInputs,
          &view, &viewOffset);

      if (vcr == VIEW_STRIDED_SLICE_FAIL) {
        // Non-unit strides or newAxisMask — can't create contiguous view, fall through to allocate
      } else if (vcr == VIEW_CREATED) {
        int slotIdx = slot.wiring.outputSlotIndices[0];
        outputs[0] = view;
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          // DIAGNOSTIC: trace view-op-install buffer stability for lifecycle debugging
          NDArray* oldCached = outputSlots_[slotIdx];
          NDArray* oldOutput = outputSlots_[slotIdx];
          auto* newDb = view->dataBuffer();
          int srcIdx0 = slot.wiring.inputSourceIndices[0];
          // Always trace for reshape ops during frozen execution
          bool isReshape = (slot.ident.opName == "reshape" || slot.ident.opName == "reshape_no_copy");
          if (DSP_DIAG_ENABLED(MEMORY) && isReshape && executeCount_ >= 2) {
            auto* oldCachedDb = oldCached != nullptr ? oldCached->dataBuffer() : nullptr;
            auto* oldOutputDb = oldOutput != nullptr ? oldOutput->dataBuffer() : nullptr;
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                "VIEW-BUF-TRACE: slot %d (%s) oldCachedDb=%p oldOutputDb=%p newDb=%p "
                "srcIdx0=%d exec=%d phase=%s cacheNull=%d outputNull=%d",
                stepIdx, slot.ident.opName.c_str(), (void*)oldCachedDb, (void*)oldOutputDb, (void*)newDb,
                srcIdx0, executeCount_, planLifecycle_.displayName(),
                oldCached == nullptr ? 1 : 0, oldOutput == nullptr ? 1 : 0);
            if (oldOutputDb != newDb) {
              DSP_DIAG_SLOT(MEMORY, stepIdx,
                  "VIEW-BUF-CHANGE: slot %d (%s) outputSlot db changed! oldOutputDb=%p newDb=%p srcIdx0=%d exec=%d",
                  stepIdx, slot.ident.opName.c_str(), (void*)oldOutputDb, (void*)newDb, srcIdx0, executeCount_);
            }
          }
          if (oldCached != nullptr) {
            auto* oldDb = oldCached->dataBuffer();
            int srcIdx0 = slot.wiring.inputSourceIndices[0];
            if (DSP_DIAG_ENABLED(MEMORY) && isReshape && executeCount_ >= 2) {
              if (srcIdx0 >= 0) {
                NDArray* srcArr = outputSlots_[srcIdx0];
                auto* srcDb = srcArr != nullptr ? srcArr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-INPUT: slot %d (%s) from slot %d (%s) srcDb=%p oldDb=%p newDb=%p exec=%d phase=%s dbMatch=%d",
                    stepIdx, slot.ident.opName.c_str(), srcIdx0,
                    srcIdx0 < numSlots_ ? slots_[srcIdx0].ident.opName.c_str() : "?",
                    (void*)srcDb, (void*)oldDb, (void*)newDb, executeCount_, planLifecycle_.displayName(),
                    oldDb == newDb ? 1 : 0);
              } else {
                int extIdx = -(srcIdx0 + 1);
                auto* extDb = (extIdx < numExt && externalArrays[extIdx] != nullptr)
                    ? externalArrays[extIdx]->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-INPUT: slot %d (%s) from ext[%d] extDb=%p oldDb=%p newDb=%p exec=%d phase=%s dbMatch=%d",
                    stepIdx, slot.ident.opName.c_str(), extIdx,
                    (void*)extDb, (void*)oldDb, (void*)newDb, executeCount_, planLifecycle_.displayName(),
                    oldDb == newDb ? 1 : 0);
              }
            }
            if (oldDb != newDb && planLifecycle_.isReplaying() && executeCount_ > 1) {
              // Buffer is changing — trace the source
              if (srcIdx0 >= 0) {
                NDArray* srcArr = outputSlots_[srcIdx0];
                auto* srcDb = srcArr != nullptr ? srcArr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-CHANGE: slot %d (%s) input from slot %d (%s) "
                    "srcDb=%p oldDb=%p newDb=%p exec=%d phase=%s",
                    stepIdx, slot.ident.opName.c_str(), srcIdx0,
                    srcIdx0 < numSlots_ ? slots_[srcIdx0].ident.opName.c_str() : "?",
                    (void*)srcDb, (void*)oldDb, (void*)newDb, executeCount_, planLifecycle_.displayName());
              } else {
                int extIdx = -(srcIdx0 + 1);
                auto* extDb = (extIdx < numExt && externalArrays[extIdx] != nullptr)
                    ? externalArrays[extIdx]->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-CHANGE: slot %d (%s) input from ext[%d] "
                    "extDb=%p oldDb=%p newDb=%p exec=%d phase=%s",
                    stepIdx, slot.ident.opName.c_str(), extIdx,
                    (void*)extDb, (void*)oldDb, (void*)newDb, executeCount_, planLifecycle_.displayName());
              }
            }
          }

          // ── Frozen-phase reuse ──
          // If we already have a slot array whose DataBuffer matches this freshly
          // created view's DataBuffer (and offset/shape match), reuse the existing
          // wrapper instead of installing a new pointer. The first slot-by-slot
          // pass installed a view here pointing at the same input0 buffer; on the
          // second execute() (post-AUTO_SEAL) the upstream input0 buffer is still
          // the same (frozen shapes), so the new view's backing buffer is
          // identical. Replacing the slot pointer in this case is the sole cause
          // of the LIFECYCLE VIOLATION at writeOutputSlot's frozen-phase guard.
          //
          // This mirrors the existing fast-path checks at line ~1067 (very fast
          // path) and ~1593 (frozen view fast path), but applies in the heavy
          // path that runs on the FIRST frozen replay (executeCount_ == 1, before
          // slot.state_ has been promoted to FROZEN).
          NDArray* existing = outputSlots_[slotIdx];
          if (existing != nullptr && existing != view &&
              existing->hasValidShapeInfo() &&
              existing->dataBuffer() == view->dataBuffer() &&
              existing->offset() == view->offset() &&
              shape::shapeEquals(existing->shapeInfo(), view->shapeInfo()) &&
              shape::strideEquals(existing->shapeInfo(), view->shapeInfo()) &&
              existing->dataType() == view->dataType()) {
            // The freshly minted view is an exact duplicate of the wrapper
            // already living at this slot. Drop the new wrapper and reuse the
            // existing one — that keeps the slot pointer (and the snapshot
            // identity captured later in execute()) stable across replays.
            delete view;
            view = existing;
            outputs[0] = existing;
            // outputSlots_[slotIdx] already == existing; no write needed.
            // writeOutputSlot still called with old==new so the frozen guard
            // is satisfied AND the WRITE_SLOT diagnostics fire as usual.
            writeOutputSlot(slotIdx, existing, "view-op-reuse");
            DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                                existing->dataBuffer() != nullptr
                                    ? existing->dataBuffer()->getLenInBytes()
                                    : 0,
                                stream, "view-op-reuse");
            slotIsViewProducer_[slotIdx] = true;
          } else {
            // Note: writeOutputSlot handles ownership and deferred cleanup safely.
            // It deletes the old array when safe (not referenced by other slots,
            // not a protected weight buffer, plan-owned).
            writeOutputSlot(slotIdx, view, "view-op-install");
            DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                                view != nullptr && view->dataBuffer() != nullptr
                                    ? view->dataBuffer()->getLenInBytes()
                                    : 0,
                                stream, "view-op-install");
            outputSlots_[slotIdx] = view;
            slotIsViewProducer_[slotIdx] = true;

          }
        }

        for (int i = 1; i < numActualOutputs; i++) {
          int si = (i < numWiredOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          // Frozen-phase reuse: if the slot already has a compatible cached
          // wrapper (same shape + dtype) reuse it instead of allocating a fresh
          // NDArray, which would trip the frozen-phase pointer-replacement
          // guard at writeOutputSlot.
          if (si >= 0 && si < totalOutputSlots_) {
            NDArray* existingSecondary = outputSlots_[si];
            if (existingSecondary != nullptr &&
                existingSecondary->hasValidShapeInfo() &&
                shape::shapeEquals(existingSecondary->shapeInfo(), shapeInfo) &&
                shape::strideEquals(existingSecondary->shapeInfo(), shapeInfo) &&
                existingSecondary->dataType() == dt) {
              outputs[i] = existingSecondary;
              writeOutputSlot(si, existingSecondary, "view-secondary-reuse");
              DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                                  existingSecondary->dataBuffer() != nullptr
                                      ? existingSecondary->dataBuffer()->getLenInBytes()
                                      : 0,
                                  stream, "view-secondary-reuse");
              continue;
            }
          }
          // Use shapeInfo-preserving constructor to match standard path strides/ordering
          outputs[i] = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
          if (si >= 0 && si < totalOutputSlots_) {
            writeOutputSlot(si, outputs[i], "view-secondary-alloc");
            DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                                outputs[i]->dataBuffer() != nullptr
                                    ? outputs[i]->dataBuffer()->getLenInBytes()
                                    : 0,
                                stream, "view-secondary-alloc");
            outputSlots_[si] = outputs[i];
          } else if (si < 0) {
            // Overflow output (shape function returned more than graph wires) —
            // cache as untracked so it can be reused across steps.
            int cacheIdx = stepIdx * MAX_OUTPUTS_PER_SLOT + i;
            if (cacheIdx < untrackedOutputCacheSize_) {
              delete untrackedOutputCache_[cacheIdx];
              untrackedOutputCache_[cacheIdx] = outputs[i];
            }
          }
        }
        outputsReady = true;
      }
      // VIEW_NOT_POSSIBLE or VIEW_STALE_EMPTY_SHAPE — fall through to allocate
    }
  }

  // ── Step 3: Allocate outputs (skip if in-place or view already set them) ──
  if (!outputsReady)
  for (int i = 0; i < numActualOutputs; i++) {
    // Guard: outputSlotIndices only has numWiredOutputs entries.
    // Overflow outputs (from shape function returning more than graph wires)
    // are treated as untracked (slotIdx = -1).
    int slotIdx = (i < numWiredOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
    if (slotIdx < 0) {
      int cacheIdx = stepIdx * MAX_OUTPUTS_PER_SLOT + i;
      if (cacheIdx < untrackedOutputCacheSize_) {
        NDArray* cached = untrackedOutputCache_[cacheIdx];
        if (cached != nullptr) {
          if (!cached->hasValidShapeInfo()) {
            delete cached;
            untrackedOutputCache_[cacheIdx] = nullptr;
            if (sd::Environment::getInstance().isDebug()) {
              scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                  "AFTER_untrackedCache2a_delete", executeCount_);
            }
          } else {
            const LongType* cachedShape = cached->shapeInfo();
            if (shape::equalsSoft(cachedShape, outputShapes[i]) &&
                ArrayOptions::dataType(cachedShape) == ArrayOptions::dataType(outputShapes[i])) {
              size_t requiredBytes = static_cast<size_t>(shape::length(outputShapes[i])) *
                                     DataTypeUtils::sizeOf(ArrayOptions::dataType(outputShapes[i]));
              auto* cachedDb = cached->dataBuffer();
              size_t availableBytes = (cachedDb != nullptr) ? cachedDb->getLenInBytes() : 0;
              if (requiredBytes <= availableBytes) {
                outputs[i] = cached;
                continue;
              }
              DSP_DIAG(MEMORY,
                       "UNTRACKED_CACHE_SKIP: step %d output %d buffer too small "
                       "(required=%zu available=%zu) — reallocating",
                       stepIdx, i, requiredBytes, availableBytes);
            }
            delete cached;
            untrackedOutputCache_[cacheIdx] = nullptr;
            if (sd::Environment::getInstance().isDebug()) {
              scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                  "AFTER_untrackedCache2b_delete", executeCount_);
            }
          }
        }
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
    LongType rank = shape::rank(shapeInfo);

    NDArray* cached = validateReusableSlotArray(slotIdx, outputSlots_[slotIdx],
                                                "step3-reuse-validate");
    if (cached != nullptr) {
      if (!cached->hasValidShapeInfo()) {
        outputSlots_[slotIdx] = nullptr;
        cached = nullptr;
      } else {
        const LongType* cachedShape = cached->shapeInfo();
        auto cachedDt = ArrayOptions::dataType(cachedShape);
        if (cachedDt != dt) {
          DSP_DIAG(SHAPE,
              "DTYPE_MISMATCH: slot %d (%s) outSlotIdx=%d cached dtype=%s inferred dtype=%s — will reassign",
              slotIdx, slot.ident.opName.c_str(), slotIdx,
              DataTypeUtils::asString(cachedDt).c_str(),
              DataTypeUtils::asString(dt).c_str());
        }
        if (shape::equalsSoft(cachedShape, shapeInfo) &&
            cachedDt == dt) {
          // Buffer capacity guard: verify the cached DataBuffer is large enough
          // to hold the current output shape before reusing it.  If the shape
          // changed between prefill and decode (e.g. sequence-length growth),
          // equalsSoft may still return true while the DataBuffer is too small,
          // leading to a silent out-of-bounds write inside the op.
          size_t requiredBytes = static_cast<size_t>(shape::length(shapeInfo)) *
                                 DataTypeUtils::sizeOf(dt);
          auto* cachedDb = cached->dataBuffer();
          size_t availableBytes = (cachedDb != nullptr) ? cachedDb->getLenInBytes() : 0;
          if (requiredBytes > availableBytes) {
            // Buffer too small — release the under-sized cached array and fall
            // through to fresh allocation below.
            DSP_DIAG(MEMORY,
                     "CACHED_REUSE_SKIP: slot %d (%s) buffer too small for current shape "
                     "(required=%zu available=%zu) — allocating fresh array",
                     slotIdx, slot.ident.opName.c_str(), requiredBytes, availableBytes);
            discardCachedSlotArray(slotIdx, cached, "cached-reuse-too-small");
            cached = nullptr;
          } else {
            outputs[i] = cached;
            writeOutputSlot(slotIdx, cached, "cached-reuse");
            DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                                cachedDb != nullptr ? cachedDb->getLenInBytes() : 0,
                                stream, "cached-reuse");
            continue;
          }
        }
      }
    }
    if (cached != nullptr) {
        // Check if this slot TRANSITIVELY depends on any external input or
        // isDynamicShape/outputShapeDependsOnInputValues slot. A one-hop check
        // misses chains like rms_norm <- add <- ext[N] because during warmup
        // (execCount 0) all outputSlots_ are nullptr so isDynamicShape is never
        // set on intermediate slots. BFS through the full dependency graph.
        bool hasDynamicUpstream = hasTransitiveDynamicUpstream(
            slots_, numSlots_, stepIdx);

        if (executeCount_ <= 1 || hasDynamicUpstream) {
          // During warmup (execCount 0 = prefill, 1 = first decode) OR for
          // dynamic-shape slots: shape drift is expected. The prefill→decode
          // transition changes external placeholder shapes (seq_len=N → 1),
          // which propagates through all downstream slots. Allow reassignment
          // during this window without marking slots dynamic (they stabilize
          // after the first decode step).
          DSP_DIAG_SLOT(SHAPE, slotIdx,
              "step3-shape-reassign: slot %d (%s) shape changed (execCount=%d dynamicUpstream=%d) — replacing cached array",
              slotIdx, slot.ident.opName.c_str(), executeCount_, hasDynamicUpstream ? 1 : 0);
          // Use discardCachedSlotArray instead of raw delete — raw delete
          // leaves a ghost pointer in planOwnedArrays_ which causes
          // "corrupted double-linked list" when malloc reuses the freed
          // chunk and planOwnedArrays_ hash table operates on stale entries.
          discardCachedSlotArray(slotIdx, cached, "step3-shape-reassign");
          cached = nullptr;
          // Invalidate slot state so frozen context/shape cache aren't reused
          // with the wrong shape.
          slot.slotPhase.reset();  // PRIMARY
          slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
          slot.shapeCache.cachedOutputShapes.clear();
          // Only mark as permanently dynamic if there's a genuinely dynamic
          // upstream (value-dependent ops like range/fill). Shape changes from
          // the prefill→decode transition are one-time events, not recurring.
          if (hasDynamicUpstream) {
            slot.flags.isDynamicShape = true;
          }
        } else {
          DSP_THROW(SHAPE,
                   "NativeDynamicShapePlan: slot reassignment on shape drift is forbidden "
                   "post-warmup at slot %d (%s) — output shape changed but slot has no "
                   "dynamic upstream (execCount=%d hasDynamicUpstream=%d). This indicates "
                   "heap corruption or a missing isDynamicShape flag. Fix the upstream "
                   "corruption or mark the slot dynamic during warmup.",
                   slotIdx, slot.ident.opName.c_str(), executeCount_,
                   hasDynamicUpstream ? 1 : 0);
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
      NDArray* emptyOut = new NDArray(const_cast<LongType*>(shapeInfo), true);
      outputs[i] = emptyOut;
      writeOutputSlot(slotIdx, emptyOut, "empty-output-alloc");
      DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                          emptyOut->dataBuffer() != nullptr
                              ? emptyOut->dataBuffer()->getLenInBytes()
                              : 0,
                          stream, "empty-output-alloc");
      outputSlots_[slotIdx] = emptyOut;
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
        LongType currentElements = 1;
        for (int d = 0; d < rank; d++) currentElements *= shape[d];
        LongType maxElements = std::max(maxIt->second, currentElements);
        size_t requiredBytes = static_cast<size_t>(currentElements) * DataTypeUtils::sizeOf(dt);
        size_t maxBytes = static_cast<size_t>(maxElements) * DataTypeUtils::sizeOf(dt);

        DSP_DIAG_SLOT(MEMORY, slotIdx, "max-allocating slot %d, current shape=[%lld,%lld,%lld,%lld], currentBytes=%zu maxBytes=%zu",
                  slotIdx, shape[0], rank>1?shape[1]:0, rank>2?shape[2]:0, rank>3?shape[3]:0,
                  requiredBytes, maxBytes);

        NDArray* maxOut = nullptr;
        try {
          if (!slot.flags.isViewCapableOp) {
            std::vector<LongType> contigShape(rank);
            for (int d = 0; d < rank; d++) contigShape[d] = shape[d];
            maxOut = new NDArray(order, contigShape, dt);
          } else {
            maxOut = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
          }
          auto* maxDb = maxOut != nullptr ? maxOut->dataBuffer() : nullptr;
          if (maxDb != nullptr && maxBytes > maxDb->getLenInBytes()) {
            maxDb->expand(maxBytes);
          }
        } catch (const std::exception& e) {
          DSP_DIAG_SLOT(MEMORY, stepIdx, "max-allocation FAILED at slot %d (%s): %s",
                    stepIdx, slot.ident.opName.c_str(), e.what());
          // Fallback to original shape — use shapeInfo-preserving constructor
          maxOut = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
        }

        outputs[i] = maxOut;
        writeOutputSlot(slotIdx, maxOut, "max-alloc-output");
        DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                            maxOut != nullptr && maxOut->dataBuffer() != nullptr
                                ? maxOut->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "max-alloc-output");
        outputSlots_[slotIdx] = maxOut;
        maxAllocatedSlots_.insert(slotIdx);
        continue;
      }
      NDArray* cached2 = validateReusableSlotArray(slotIdx, outputSlots_[slotIdx],
                                                   "maxalloc-reuse-validate");
      if (cached2 != nullptr) {
        outputs[i] = cached2;
        writeOutputSlot(slotIdx, cached2, "cached2-reuse");
        DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                            cached2->dataBuffer() != nullptr
                                ? cached2->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "cached2-reuse");
        continue;
      }
    }

    NDArray* out = nullptr;
    try {
      // Non-view ops get a fresh C-contiguous buffer. Force C-contiguous shape
      // info so strides match the physical layout (legacy COPY_SHAPE ops can
      // inherit non-contiguous strides from permuted inputs, which would be
      // wrong on a fresh buffer). View ops keep original strides/offsets.
      if (!slot.flags.isViewCapableOp) {
        std::vector<LongType> contigShape(rank);
        for (int d = 0; d < rank; d++) contigShape[d] = shapeInfo[d + 1];
        out = new NDArray(order, contigShape, dt);
      } else {
        // Use the shapeInfo-preserving constructor to match the standard path
        // (prepareOutputs: new NDArray(out, true, ...)). This preserves the
        // exact strides, ordering, and EWS from calculateOutputShape.
        out = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
      }
    } catch (const std::exception& e) {
      DSP_DIAG_SLOT(MEMORY, stepIdx, "output ALLOC EXCEPTION at slot %d (%s) output[%d]: %s",
                stepIdx, slot.ident.opName.c_str(), i, e.what());
      return Status::KERNEL_FAILURE;
    }

    outputs[i] = out;
    writeOutputSlot(slotIdx, out, "normal-alloc-output");
    DSP_DIAG_SLOT_WRITE(slotIdx, slot.ident.opName.c_str(),
                        out != nullptr && out->dataBuffer() != nullptr
                            ? out->dataBuffer()->getLenInBytes()
                            : 0,
                        stream, "normal-alloc-output");
    outputSlots_[slotIdx] = out;
  }

  // ── Step 4: Execute ──

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
      bool isViewOp = slot.flags.isViewCapableOp;
      bool primaryInputNonEmpty = (slot.wiring.numInputs > 0 && inputs[0] != nullptr &&
                                   !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0);
      if (isViewOp && primaryInputNonEmpty) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "view-capable slot %d (%s): outputs empty but input[0] non-empty "
                  "(length=%lld) — forcing execution instead of skip",
                  stepIdx, slot.ident.opName.c_str(), inputs[0]->lengthOf());
        // Invalidate shape cache so next execution re-derives shapes
        slot.slotPhase.reset();  // PRIMARY
        slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
      } else {
        DSP_DIAG_SLOT(EXECUTE, stepIdx, "skipping slot %d (%s): all %d outputs are empty",
                  stepIdx, slot.ident.opName.c_str(), numActualOutputs);
        return Status::OK;
      }
    }
  }

  // ── Step 4: Configure context and execute ────────────────────────────────
  auto& ctx = *contextPool_[stepIdx];

  for (int i = 0; i < slot.wiring.numInputs; i++) {
    ctx.setInputArray(i, inputs[i]);
  }

  for (int i = 0; i < numActualOutputs; i++) {
    ctx.setOutputArray(i, outputs[i]);
  }

  if (slot.args.numIArgs > 0) ctx.setIArguments(slot.args.iArgs, slot.args.numIArgs);
  if (slot.args.numTArgs > 0) ctx.setTArguments(slot.args.tArgs, slot.args.numTArgs);
  if (slot.args.numBArgs > 0) ctx.setBArguments(slot.args.bArgs, slot.args.numBArgs);
  if (slot.args.numDArgs > 0) ctx.setDArguments(slot.args.dArgs, slot.args.numDArgs);
  ctx.getSArguments()->clear();
  if (slot.args.numSArgs > 0) {
    ctx.getSArguments()->insert(ctx.getSArguments()->end(), slot.args.sArgs, slot.args.sArgs + slot.args.numSArgs);
  }

  // Legacy reduce/broadcast ops read reduction dims from block.getAxis(), not iArgs.
  // The Java compiler packs dims into iArgs; mirror them into axis for these op types.
  if (legacyOpReadsAxisFromIArgs(slot.legacy.legacyOpType)) {
    ctx.getAxis()->clear();
    for (int i = 0; i < slot.args.numIArgs; i++) {
      ctx.getAxis()->emplace_back(static_cast<sd::LongType>(slot.args.iArgs[i]));
    }
  }

  ctx.setShapeFunctionOverride(true);

  // Shape-key computation and shape functions may sync control tensors to host
  // in order to read value-dependent shape inputs. Re-sync those inputs before
  // execution so the kernel does not read stale device-side shape values.
  if (slot.flags.isDataDependent || slot.flags.outputShapeDependsOnInputValues) {
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      NDArray* in = inputs[i];
      if (in == nullptr || in->isEmpty()) continue;
      auto* db = in->dataBuffer();
      if (db == nullptr || db->isClosed()) continue;

      const bool syncAllInputs = slot.flags.isDataDependent;
      const bool isControlInput = isSmallIntegralControlArray(in);
      if (!syncAllInputs && !isControlInput) continue;

      if (isControlInput && DSP_DIAG_ENABLED(SHAPE)) {
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "PRE-SYNC-TRACE: slot %d (%s) input[%d] dtype=%d len=%lld pAct=%d sAct=%d "
            "values=[%s]",
            stepIdx, slot.ident.opName.c_str(), i, (int)in->dataType(), (long long)in->lengthOf(),
            db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
            dspDumpHostDeviceValues(in, 16).c_str());
      }

      in->syncToDevice();
    }
  }



  // Log shapes for matmul gap slots to diagnose capture shape mismatches
  if (DSP_DIAG_ENABLED(EXECUTE) && slot.ident.op && slot.ident.op->getOpDescriptor() &&
      slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_MATMUL)) {
    std::string inStr, outStr, cachedStr;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
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
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (inputs[i] != nullptr) {
        auto* db = inputs[i]->dataBuffer();
        DSP_DIAG(EXECUTE, "MATMUL INPUT[%d]: slot %d srcIdx=%d addr=%p special=%p len=%lld "
                  "db=%p closed=%d const=%d",
                  i, stepIdx, slot.wiring.inputSourceIndices[i],
                  (void*)inputs[i], db ? db->special() : nullptr,
                  (long long)inputs[i]->lengthOf(),
                  (void*)db, db ? db->isClosed() : -1,
                  db ? (int)db->isConstant : -1);
      }
    }
  }

  // Set cublasLt epilogue if this matmul has fused bias+activation
  if (slot.flags.ltEpilogueType > 0 && slot.flags.ltEpilogueBiasSourceIdx != -1) {
    NDArray* biasArray = nullptr;
    int biasSrc = slot.flags.ltEpilogueBiasSourceIdx;
    if (biasSrc >= 0) {
      biasArray = outputSlots_[biasSrc];
    } else {
      int extIdx = -(biasSrc + 1);
      if (extIdx < numExt) biasArray = externalArrays[extIdx];
    }
    if (biasArray != nullptr) {
      platformSetLtEpilogue(slot, biasArray);
      DSP_DIAG(FUSION, "set Lt epilogue type=%d bias=%p size=%lld for slot %d (%s)",
                slot.flags.ltEpilogueType, (void*)biasArray,
                (long long)(biasArray->lengthOf() * biasArray->sizeOfT()),
                stepIdx, slot.ident.opName.c_str());
    }
  }

  // ── DSP debug: mirror standard path's isDebugAndVerbose output + slot info (warmup path) ──
  if (Environment::getInstance().isDebugAndVerbose()) {
    DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "DSP_SLOT_WU op=%s numInputs=%d numOutputs=%d",
                  slot.ident.opName.c_str(), slot.wiring.numInputs, numActualOutputs);
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      auto* array = inputs[i];
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (array != nullptr) {
        auto shape = ShapeUtils::shapeAsString(array);
        auto type = DataTypeUtils::asString(array->dataType());
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
                      "  input[%d] shape=%s dtype=%s src=%d",
                      i, shape.c_str(), type.c_str(), srcIdx);
      } else {
        DSP_DIAG_SLOT(EXECUTE, stepIdx, "  input[%d] NULL src=%d", i, srcIdx);
      }
    }
  }

  // Ensure CUDA host↔device coherency (warmup path).
  // PERFORMANCE: In frozen steady-state (executeCount_ >= 2), all data is already
  // device-resident and actuality flags are correct. Skipping eliminates ~5486
  // syncToDevice calls per decode step.
  // forceSync_: Pre-capture warmup forces sync at exec=2+ (same rationale as above).
  bool needsSync = forceSync_ || planLifecycle_.isSlotBySlot() || executeCount_ < 2;
  if (needsSync) {
    NDArray::prepareSpecialUse(ctx.fastpath_out(), ctx.fastpath_in());
  }
  if (!needsSync && DspDiagnostics::getInstance().isEnabled(DSP_DIAG_MEMORY)) {
    auto& fpin = ctx.fastpath_in();
    for (size_t ii = 0; ii < fpin.size(); ii++) {
      NDArray* a = fpin[ii];
      if (a == nullptr) continue;
      DataBuffer* db = a->dataBuffer();
      if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
      bool pAct = db->isPrimaryActual();
      bool sAct = db->isSpecialActual();
      if (pAct && !sAct) {
        DSP_DIAG(MEMORY,
                 "SYNC_SKIP_ANOMALY_IN(warmup): slot=%d op=%s inIdx=%zu db=%p len=%lld "
                 "pAct=1 sAct=0 exec=%d — host poisoned device-authoritative buffer",
                 stepIdx, slot.ident.opName.c_str(), ii, (void*)db,
                 (long long)db->getLenInBytes(), executeCount_);
      }
    }
  }

  Status status;
  if (shapeOnlyMode_) {
    DSP_DIAG(EXECUTE,
             "SHAPE_ONLY: skipping op execution for slot %d (%s) [warmup path]",
             stepIdx, slot.ident.opName.c_str());
    status = Status::OK;
  } else {
  // Pre-execute corruption scan (warmup path)
  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      if (!validateSlotShapeBufferAlignment(outputs[i],
              slot.wiring.outputSlotIndices[i < slot.wiring.numOutputs ? i : 0],
              "BEFORE_warmup_op_execute", executeCount_)) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "CORRUPTION_BEFORE_EXEC_WU: slot %d (%s) output[%d] "
                 "already corrupt BEFORE warmup op->execute. execCount=%d",
                 stepIdx, slot.ident.opName.c_str(), i, executeCount_);
        THROW_EXCEPTION(msg);
      }
    }
  }
  prezeroThisSlot();
  try {
    status = slot.ident.op->execute(&ctx);
  } catch (const std::exception& e) {
    // Reshape failures and other exceptions land here — log with DSP_DIAG
    std::string inputShapes, outputShapesStr, iArgsStr;
    dspFormatSlotExecContext(slot, inputs.data(), slot.wiring.numInputs,
                             outputs.data(), numActualOutputs,
                             inputShapes, outputShapesStr, iArgsStr);
    DSP_DIAG(EXECUTE, "SLOT EXEC EXCEPTION: slot %d (%s) exception='%s', inputs=[%s], outputs=[%s], "
              "iArgs=[%s], cacheHit=%d executeCount=%d shapesFrozen=%d",
              stepIdx, slot.ident.opName.c_str(), e.what(),
              inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
              (int)cacheHit, executeCount_, (int)shapesFrozen_);
    // Propagate error detail to Java via errorReference
    dspPropagateSlotError(stepIdx, slot,
                           (std::string("exec exception: ") + e.what()).c_str(),
                           inputShapes, outputShapesStr, iArgsStr);
    // Clear epilogue state even on exception
    if (slot.flags.ltEpilogueType > 0) platformClearLtEpilogue();
    return Status::KERNEL_FAILURE;
  }
  }  // end else (!shapeOnlyMode_)

  // Post-execute corruption scan (warmup path)
  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      int si = i < slot.wiring.numOutputs ? slot.wiring.outputSlotIndices[i] : -1;
      if (!validateSlotShapeBufferAlignment(outputs[i], si,
              "AFTER_warmup_op_execute", executeCount_)) {
        sd_printf("CORRUPTION_AFTER_EXEC_WU: slot %d (%s) output[%d] outSlot=%d "
                  "corrupted BY warmup op->execute. execCount=%d\n",
                  stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
        fflush(stdout);
      }
    }
  }

  // Full slot scan after warmup execute — catch cross-slot heap overruns
  if (sd::Environment::getInstance().isDebug()) {
    char wuLabel[256];
    snprintf(wuLabel, sizeof(wuLabel),
             "AFTER_warmup_exec_step%d_%s", stepIdx, slot.ident.opName.c_str());
    scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
        wuLabel, executeCount_);
  }

  // Clear epilogue state after matmul execution
  if (!shapeOnlyMode_ && slot.flags.ltEpilogueType > 0) platformClearLtEpilogue();

  if (status != Status::OK) {
    std::string inputShapes, outputShapesStr, iArgsStr;
    dspFormatSlotExecContext(slot, inputs.data(), slot.wiring.numInputs,
                             outputs.data(), numActualOutputs,
                             inputShapes, outputShapesStr, iArgsStr);
    // Propagate error detail to Java via errorReference
    dspPropagateSlotError(stepIdx, slot,
                           (std::string("exec failed status=") + std::to_string(static_cast<int>(status))).c_str(),
                           inputShapes, outputShapesStr, iArgsStr);
    DSP_DIAG(EXECUTE, "SLOT EXEC FAIL: slot %d (%s) status=%d, inputs=[%s], outputs=[%s], iArgs=[%s], "
              "cacheHit=%d executeCount=%d shapesFrozen=%d",
              stepIdx, slot.ident.opName.c_str(), static_cast<int>(status),
              inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
              (int)cacheHit, executeCount_, (int)shapesFrozen_);
  }

  // Register device writes (warmup path).
  // PERFORMANCE: Skip in frozen steady-state to eliminate ~5486 sync calls/step.
  if (needsSync) {
    NDArray::registerSpecialUse(ctx.fastpath_out(), ctx.fastpath_in());
  } else if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_MEMORY)) {
    auto& fpout = ctx.fastpath_out();
    for (size_t oi = 0; oi < fpout.size(); oi++) {
      NDArray* a = fpout[oi];
      if (a == nullptr) continue;
      DataBuffer* db = a->dataBuffer();
      if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
      bool pAct = db->isPrimaryActual();
      bool sAct = db->isSpecialActual();
      if (!sAct) {
        DSP_DIAG(MEMORY,
                 "SYNC_SKIP_ANOMALY_OUT(warmup): slot=%d op=%s outIdx=%zu db=%p len=%lld "
                 "pAct=%d sAct=0 exec=%d — op did not tick device-write counter",
                 stepIdx, slot.ident.opName.c_str(), oi, (void*)db,
                 (long long)db->getLenInBytes(), (int)pAct, executeCount_);
      }
    }
  }

  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      // Post-execution output integrity check (warmup/normal path)
      auto* postSib = outputs[i]->shapeInfoConstBuffer();
      if (postSib != nullptr) {
        uintptr_t postSibAddr = reinterpret_cast<uintptr_t>(postSib);
        if (postSibAddr % alignof(ConstantShapeBuffer) != 0) {
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "DSP CORRUPTION DETECTED: slot %d (%s) output[%d] _shapeInfoBuffer=%p "
                   "is misaligned AFTER op execution (alignOffset=%zu). "
                   "The op itself corrupted this output's shape buffer. "
                   "execCount=%d shapesFrozen=%d",
                   stepIdx, slot.ident.opName.c_str(), i, (void*)postSib,
                   static_cast<size_t>(postSibAddr % alignof(ConstantShapeBuffer)),
                   executeCount_, shapesFrozen_ ? 1 : 0);
          THROW_EXCEPTION(msg);
        }
      }
      platformReconcileOutputActuality("normal-op-exec", stepIdx, slot, outputs[i]);
    }
  }

  if (status == Status::OK) {
    NDArray* tracedInput0 = slot.wiring.numInputs > 0 && !inputs.empty() ? inputs[0] : nullptr;
    NDArray* tracedOutput0 = numActualOutputs > 0 ? outputs[0] : nullptr;
    traceSmallControlSlotIO("normal-op-exec", stepIdx, slot,
                            tracedInput0, tracedOutput0,
                            executeCount_, planLifecycle_.displayName());
    traceSmallControlSlotTensors("normal-op-exec", stepIdx, slot,
                                 inputs.data(), slot.wiring.numInputs,
                                 outputs.data(), numActualOutputs,
                                 executeCount_, planLifecycle_.displayName());
  }

  // ── Step 5: View producer handling ────────────────────────────────────────
  {
    auto& ctxOutputs = ctx.fastpath_out();
    for (int i = 0; i < numActualOutputs && i < static_cast<int>(ctxOutputs.size()); i++) {
      int si = (i < slot.wiring.numOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
      if (si < 0) continue;
      if (ctxOutputs[i] == nullptr) continue;

      if (!outputWrapperMatchesExpectedShape(ctxOutputs[i], outputShapes[i])) {
        DSP_DIAG(SHAPE,
                 "CTX_OUTPUT_REJECT(view-producer): slot=%d op=%s outIdx=%d expected=%s actual=%s",
                 stepIdx, slot.ident.opName.c_str(), i,
                 ShapeUtils::shapeAsString(outputShapes[i]).c_str(),
                 ShapeUtils::shapeAsString(ctxOutputs[i]).c_str());
        continue;
      }

      if (!viewProducerDetectionDone_) {
        if (ctxOutputs[i] != outputs[i]) {
          slotIsViewProducer_[si] = true;
          // writeOutputSlot handles safe deletion of old arrays
          writeOutputSlot(si, ctxOutputs[i], "view-producer-detect");
          DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                              ctxOutputs[i] != nullptr && ctxOutputs[i]->dataBuffer() != nullptr
                                  ? ctxOutputs[i]->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "view-producer-detect");
          outputSlots_[si] = ctxOutputs[i];
        }
      } else if (slotIsViewProducer_[si]) {
        // writeOutputSlot handles safe deletion of old arrays
        writeOutputSlot(si, ctxOutputs[i], "view-producer-update");
        DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                            ctxOutputs[i] != nullptr && ctxOutputs[i]->dataBuffer() != nullptr
                                ? ctxOutputs[i]->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "view-producer-update");
        outputSlots_[si] = ctxOutputs[i];
      }
    }
  }

  // ── DSP debug: post-execute output fingerprint (warmup path) ──
  if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose() && status == Status::OK) {
    for (int i = 0; i < numActualOutputs; i++) {
      auto* array = outputs[i];
      if (array != nullptr && !array->isEmpty() && array->lengthOf() > 0) {
        array->syncToHost();
        double sum = 0.0;
        double firstVal = array->e<double>(0);
        sd::LongType len = array->lengthOf();
        sd::LongType fpLen = sd::math::sd_min(64LL, len);
        for (sd::LongType fi = 0; fi < fpLen; fi++) {
          sum += array->e<double>(fi);
        }
        DSP_DIAG(VERIFY,
                 "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld first=%.8g sum64=%.8g",
                 executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                 ShapeUtils::shapeAsString(array).c_str(),
                 DataTypeUtils::asString(array->dataType()).c_str(),
                 (long long)len, firstVal, sum);
      }
    }
  }



  // Promote slots to FROZEN on the first frozen execution so subsequent
  // steps can use the frozen fast path (frozenContextReady requires FROZEN).
  if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 0 && status == Status::OK) {
    if (!slot.slotPhase.isSealed()) {
      slot.slotPhase.seal(false);  // PRIMARY: freeze (not constant)
      slot.state_ = NativeSlot::SlotState::FROZEN;  // legacy sync
    }
  }

  if (status == Status::OK) {
    platformLogSlotOutput(stepIdx, slot.ident.opName.c_str(), "OP_EXEC",
                          slot.wiring.outputSlotIndices, slot.wiring.numOutputs);
  }

  // Slot-write site: normal (non-fast-frozen) heavy path end. Covers all
  // alloc/reuse → kernel execute → view-producer handling paths above. Bump
  // only on success so failure paths don't create spurious generation drift.
  if (status == Status::OK) {
    slot.bumpGeneration();
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      int si = slot.wiring.outputSlotIndices[i];
      if (si >= 0 && si < totalOutputSlots_) dirtySlotGenerations_[si] = currentDirtyGeneration_;
    }
  }

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

}  // namespace graph
}  // namespace sd
