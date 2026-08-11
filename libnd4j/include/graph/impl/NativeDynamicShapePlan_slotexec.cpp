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
#include <graph/DspThreadState.h>
#include <graph/gpu/DspCudaDispatch.h>
#include <graph/DspHashUtils.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/FusionPass.h>
#include <graph/LegacyOpTypeCodes.h>
#include <graph/PlanExecutionContext.h>
#include <ops/declarable/OpDescriptor.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBuilders.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace sd {
namespace graph {

// ── SlotSyncGuard: RAII wrapper for prepareSpecialUse / registerSpecialUse ───
// Consolidates the scattered needsSync() → prepare → execute → register pattern
// into a single guard. Construction prepares; destruction registers.
// ONE definition, used by all three slot execution paths (gap ops, frozen ctx,
// warmup). Anomaly detection for skipped sync is included in the destructor.
struct SlotSyncGuard {
  std::vector<NDArray*>& outputs;
  std::vector<NDArray*>& inputs;
  const bool doSync;
  const int stepIdx;
  const char* opName;
  const int execCount;

  SlotSyncGuard(bool sync, std::vector<NDArray*>& outs, std::vector<NDArray*>& ins,
                int step, const char* op, int exec)
      : outputs(outs), inputs(ins), doSync(sync), stepIdx(step), opName(op), execCount(exec) {
    // Record pre-prepare state for all inputs
    for (size_t ii = 0; ii < inputs.size(); ii++) {
      DSP_LIFECYCLE_EVENT(exec, step, doSync ? "PREPARE_IN" : "PREPARE_IN_SKIP", inputs[ii]);
    }
    for (size_t oi = 0; oi < outputs.size(); oi++) {
      DSP_LIFECYCLE_EVENT(exec, step, doSync ? "PREPARE_OUT" : "PREPARE_OUT_SKIP", outputs[oi]);
    }
    if (doSync) {
      NDArray::prepareSpecialUse(outputs, inputs);
      // Record post-prepare state
      for (size_t ii = 0; ii < inputs.size(); ii++) {
        DSP_LIFECYCLE_EVENT(exec, step, "POST_PREPARE_IN", inputs[ii]);
      }
      for (size_t oi = 0; oi < outputs.size(); oi++) {
        DSP_LIFECYCLE_EVENT(exec, step, "POST_PREPARE_OUT", outputs[oi]);
      }
    }
  }

  ~SlotSyncGuard() {
    if (doSync) {
      // Record pre-register state
      for (size_t oi = 0; oi < outputs.size(); oi++) {
        DSP_LIFECYCLE_EVENT(execCount, stepIdx, "PRE_REGISTER_OUT", outputs[oi]);
      }
      NDArray::registerSpecialUse(outputs, inputs);
      // Record post-register state
      for (size_t oi = 0; oi < outputs.size(); oi++) {
        DSP_LIFECYCLE_EVENT(execCount, stepIdx, "POST_REGISTER_OUT", outputs[oi]);
      }
    } else if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_MEMORY)) {
      // Anomaly detection: if sync was skipped, verify outputs are device-actual.
      for (size_t oi = 0; oi < outputs.size(); oi++) {
        NDArray* a = outputs[oi];
        if (a == nullptr) continue;
        DataBuffer* db = a->dataBuffer();
        if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
        if (!db->isSpecialActual()) {
          DSP_DIAG(MEMORY,
                   "SYNC_SKIP_ANOMALY_OUT: slot=%d op=%s outIdx=%zu db=%p len=%lld "
                   "sAct=0 exec=%d — op did not tick device-write counter",
                   stepIdx, opName, oi, (void*)db,
                   (long long)db->getLenInBytes(), execCount);
        }
      }
      for (size_t ii = 0; ii < inputs.size(); ii++) {
        NDArray* a = inputs[ii];
        if (a == nullptr) continue;
        DataBuffer* db = a->dataBuffer();
        if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
        if (db->isPrimaryActual() && !db->isSpecialActual()) {
          DSP_DIAG(MEMORY,
                   "SYNC_SKIP_ANOMALY_IN: slot=%d op=%s inIdx=%zu db=%p len=%lld "
                   "pAct=1 sAct=0 exec=%d — host poisoned device-authoritative buffer",
                   stepIdx, opName, ii, (void*)db,
                   (long long)db->getLenInBytes(), execCount);
        }
      }
    }
  }
};

// Verify helpers now in DspVerifyUtils.h (dspLogSlotOutput, dspDumpSlotValues, etc.)

/**
 * Ensure a shape to be cached does not carry UNKNOWN dtype.
 *
 * View ops (permute, reshape, squeeze, expand_dims) preserve their input's
 * dtype, so their output shape must carry a concrete type.  When a stale
 * or corrupted view leaks UNKNOWN into its shapeInfo extras, every
 * downstream consumer (refreshStaleViewWrappersInSegment, frozen fast path,
 * inline view re-creation, and matmul shape inference via max(dtypeX, dtypeY))
 * inherits the poison.
 *
 * Fix: if the shape has UNKNOWN dtype and the slot has a resolvable first
 * input, re-create the shape with the input's dtype via ConstantShapeHelper.
 * This keeps the cache clean so all consumers see valid dtypes.
 */
static const LongType* sanitizeCachedShapeDtype(
    const LongType* shape, const NativeSlot& slot,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalArrays = nullptr, int numExt = 0) {
  if (shape == nullptr) return nullptr;
  if (ArrayOptions::dataType(shape) != DataType::UNKNOWN) return shape;

  // Resolve the first input to derive the correct dtype.
  DataType fallbackDt = DataType::UNKNOWN;
  for (int ii = 0; ii < slot.wiring.numInputs && fallbackDt == DataType::UNKNOWN; ii++) {
    int srcIdx = slot.wiring.inputSourceIndices[ii];
    NDArray* resolved = nullptr;
    if (srcIdx >= 0 && srcIdx < totalOutputSlots) {
      resolved = outputSlots[srcIdx];
    } else if (srcIdx < 0 && externalArrays != nullptr) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) resolved = externalArrays[extIdx];
    }
    if (resolved != nullptr) {
      fallbackDt = resolved->dataType();
    }
  }

  if (fallbackDt == DataType::UNKNOWN) return shape;  // nothing we can do

  // Re-create with correct dtype.  ConstantShapeHelper returns a
  // cache-managed pointer so the caller does not own it.
  return ConstantShapeHelper::getInstance().createShapeInfo(
      fallbackDt, shape::order(shape),
      static_cast<int>(shape::rank(shape)),
      shape::shapeOf(const_cast<LongType*>(shape)));
}

/**
 * Apply sanitizeCachedShapeDtype to every entry in a cachedOutputShapes
 * vector, fixing any UNKNOWN-dtype entries in place.
 */
static void sanitizeCachedOutputShapes(
    NativeSlot& slot, NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalArrays = nullptr, int numExt = 0) {
  for (size_t i = 0; i < slot.shapeCache.cachedOutputShapes.size(); i++) {
    slot.shapeCache.cachedOutputShapes[i] = sanitizeCachedShapeDtype(
        slot.shapeCache.cachedOutputShapes[i], slot,
        outputSlots, totalOutputSlots, externalArrays, numExt);
  }
}

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
      auto& shapeHelper = ConstantShapeHelper::getInstance();
      // Preserve ARRAY_IS_VIEW for trait-declared view outputs: logical copy/dup
      // dispatch depends on it when strides are non-contiguous.
      auto cached = o == 0 && slot.isViewCapableOp()
                        ? shapeHelper.bufferForShapeInfoWithView(shapePtr)->primary()
                        : shapeHelper.createFromExisting(shapePtr);
      slot.shapeCache.cachedOutputShapes.push_back(cached);
    }
  }
  // Sanitize: ensure no UNKNOWN dtype leaked into the cache from stale views.
  sanitizeCachedOutputShapes(slot, outputSlots, totalOutputSlots);
  if (!slot.shapeCache.cachedOutputShapes.empty() && !slot.slotPhase.shapeCacheValid) {
    slot.slotPhase.markShapeCached();  // PRIMARY
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

// View producers (permute, transpose, slice, cast-to-view...) legitimately emit an
// output wrapper whose STRIDES differ from the canonical (contiguous) strides the op's
// DECLARE_SHAPE_FN reports: e.g. permute's evalPermShapeInfo(setContigStrides=true)
// returns C-contiguous strides, but the runtime op produces a permuted-strides VIEW.
// For the view-producer install (Step 5) we must accept the op's actual output on
// dtype+shape match WITHOUT requiring strideEquals — the permuted strides ARE the
// correct result. Requiring strideEquals there rejected valid views and left the slot's
// unwritten (zero) pre-allocated array installed, zeroing GQA K/V across all 30 layers
// (decode garbage, firstToken=11126). The strideEquals gate is preserved for the
// frozen / refresh / cache-validity checks, which detect a STALE cached wrapper.
static bool outputWrapperShapeAndTypeMatch(NDArray* array, const LongType* expectedShape) {
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
         shape::shapeEquals(actualShape, expectedShape);
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
    DSP_DIAG(MEMORY,
             "CORRUPTION_PINPOINT: slot=%d tag=%s arr=%p "
             "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d",
             slotIdx, tag, (void*)arr, (void*)sib,
             static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
             execCount);
    return false;
  }
  // Also check the actual shape data for corruption (rank validity).
  // This catches cases where the ConstantShapeBuffer itself is fine but
  // the underlying shapeCopy data has been overwritten by float/garbage.
  auto* si = arr->shapeInfo();
  if (si != nullptr) {
    LongType rank = si[0];
    if (rank < 0 || rank > SD_MAX_RANK) {
      DSP_DIAG(MEMORY,
               "SHAPE_DATA_CORRUPTION: slot=%d tag=%s arr=%p shapeInfo=%p "
               "rank=%lld (0x%llx) execCount=%d",
               slotIdx, tag, (void*)arr, (void*)si,
               (long long)rank, (unsigned long long)rank, execCount);
      return false;
    }
  }
  return true;
}

/**
 * Scan ALL output slots for _shapeInfoBuffer alignment AND data corruption.
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
      DSP_DIAG(MEMORY,
               "CORRUPTION_SCAN_HIT: checkpoint=%s slot=%d arr=%p "
               "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d",
               checkpoint, i, (void*)outputSlots[i], (void*)sib,
               static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
               execCount);
      return;  // report first hit only
    }
    // Check shape data integrity — catches float overwrite of shape buffer
    auto* si = outputSlots[i]->shapeInfo();
    if (si != nullptr) {
      LongType rank = si[0];
      if (rank < 0 || rank > SD_MAX_RANK) {
        DSP_DIAG(MEMORY,
                 "SHAPE_DATA_SCAN_HIT: checkpoint=%s slot=%d arr=%p "
                 "shapeInfo=%p rank=%lld (0x%llx) execCount=%d",
                 checkpoint, i, (void*)outputSlots[i], (void*)si,
                 (long long)rank, (unsigned long long)rank, execCount);
        return;  // report first hit only
      }
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
 * outputSlots_, slots_[].slotPhase.isViewProducer, and context — this helper only
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
  if (input0->dataBuffer() == nullptr || !input0->dataBuffer()->isValid() ||
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

  // View-producing ops describe their exact view layout in their shape
  // functions. The executor consumes that descriptor generically.

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
  // Guard against closed/invalid DataBuffer (use-after-free: Java GC zeroed the memory,
  // making _dataType = 0 = INHERIT). getNumElements() now returns 0 for closed/INHERIT
  // buffers, but check isValid() here for an early-exit diagnostic.
  if (!input0->dataBuffer()->isValid()) {
    return VIEW_NOT_POSSIBLE;
  }
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
    if (DSP_DIAG_ENABLED(MEMORY)) {
      auto* mintDb = input0->dataBuffer();
      DSP_DIAG_SLOT(MEMORY, stepIdx,
          "VIEW_MINT_SRC: slot=%d op=%s input0=%p db=%p dbValid=%d dbClosed=%d lenBytes=%zu",
          stepIdx, slot.ident.opName.c_str(), (void*)input0, (void*)mintDb,
          mintDb != nullptr && mintDb->isValid() ? 1 : 0,
          mintDb != nullptr && mintDb->isClosed() ? 1 : 0,
          mintDb != nullptr ? mintDb->getLenInBytes() : 0);
    }
    *outView = new NDArray(input0->dataBuffer(),
                           const_cast<LongType*>(outShapeInfo),
                           LaunchContext::defaultContext(),
                           absoluteOffset);
    *outViewOffset = absoluteOffset;
    if (DSP_DIAG_ENABLED(SHAPE)) {
      std::string viewStrides;
      const auto viewRank = (*outView)->rankOf();
      const auto* strides = shape::stride((*outView)->shapeInfo());
      for (int d = 0; d < viewRank; d++) {
        if (d > 0) viewStrides += ",";
        viewStrides += std::to_string(strides[d]);
      }
      DSP_DIAG_SLOT(SHAPE, stepIdx,
          "VIEW_LAYOUT: slot=%d op=%s inferredOrder=%c viewOrder=%c strides=[%s]",
          stepIdx, slot.ident.opName.c_str(), shape::order(outShapeInfo),
          (*outView)->ordering(), viewStrides.c_str());
    }
    if (traceSmallControl && DSP_DIAG_ENABLED(SHAPE)) {
      DSP_DIAG(SHAPE,
               "VIEW_TRACE_OK: slot %d (%s) zero-copy view created "
               "inShape=%s outShape=%s inLen=%lld outLen=%lld offset=%lld",
               stepIdx, slot.ident.opName.c_str(),
               ShapeUtils::shapeAsString(input0).c_str(),
               ShapeUtils::shapeAsString(outShapeInfo).c_str(),
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
               ShapeUtils::shapeAsString(outShapeInfo).c_str(),
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
             ShapeUtils::shapeAsString(outShapeInfo).c_str(),
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
// and triggers bumpArgGeneration() so refreshArgTablesForReplay picks up
// the new device-side addresses on the next replay.
//
// Iteration order [startSlot..endSlot] ensures a parent view-producer slot
// is refreshed before a child view that reads from it (e.g., squeeze(squeeze(x))).
int NativeDynamicShapePlan::refreshStaleViewWrappersInSegment(
    GraphSegment& seg, NDArray** externalArrays, int numExt) {
  if (outputSlots_ == nullptr) {
    return 0;
  }

  int refreshedCount = 0;
  // Retire old view wrappers into the plan-owned queue. They must survive not
  // only this refresh loop but the rest of the plan traversal; the queue drains
  // at the established post-execution boundary.

  for (int stepIdx = seg.def.startSlot;
       stepIdx <= seg.def.endSlot && stepIdx < numSlots_; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    if (!slot.isViewCapableOp()) {
      continue;
    }
    if (slot.wiring.numOutputs <= 0) continue;

    static thread_local std::vector<NDArray*> viewInputs;
    viewInputs.resize(slot.wiring.numInputs);
    NDArray* input0 = nullptr;
    for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
      int srcIdx = slot.wiring.inputSourceIndices[ii];
      // Use resolveViewInput to prefer staging buffers for external inputs.
      // Views must alias the staging buffer (stable address) so that cuBLAS
      // kernels baked into the CUDA graph see consistent addresses on replay.
      NDArray* resolved = resolveViewInput(srcIdx, externalArrays, numExt);
      // Probe BEFORE any deref: on a plan-cache reuse the previous borrower's
      // teardown can leave a destructed NDArray in outputSlots_ (internal
      // srcIdx) — reading ->dataBuffer() from it is the bench-250 SIGSEGV at
      // prepareBundledMtp's freeze-time refresh. A dead input nulls out here
      // and the existing refresh-input-invalid demote path repairs the slot.
      if (resolved != nullptr && !safeHasValidShapeInfo(resolved)) {
        DSP_DIAG_SLOT(MEMORY, stepIdx,
            "VIEW_REFRESH_INPUT_DEAD: step %d (%s) input[%d] srcIdx=%d arr=%p "
            "failed shape-info probe — treating as invalid input",
            stepIdx, slot.ident.opName.c_str(), ii, srcIdx, (void*)resolved);
        resolved = nullptr;
      }
      viewInputs[ii] = resolved;
      if (ii == 0) input0 = resolved;
    }

    for (int outIdx = 0; outIdx < slot.wiring.numOutputs; outIdx++) {
      int outSi = slot.wiring.outputSlotIndices[outIdx];
      if (outSi < 0 || outSi >= totalOutputSlots_) continue;
      if (!slot.slotPhase.isViewProducer) continue;

      NDArray* cached = outputSlots_[outSi];
      bool cachedValid = safeHasValidShapeInfo(cached);
      // A new borrower may deliberately invalidate an external-fed parent view
      // by nulling its output slot. Re-mint that null parent from the cached
      // shape so downstream views can be refreshed in topological order.
      if (cachedValid && cached->isEmpty()) continue;

      if (input0 == nullptr || input0->dataBuffer() == nullptr
          || !input0->dataBuffer()->isValid()) {
        // Input DataBuffer was freed (placeholder closed between calls).
        // demoteViewProducer inspects the slot and clears it if stale.
        demoteViewProducer(stepIdx, outSi, "refresh-input-invalid");
        refreshedCount++;
        continue;
      }

      const LongType* cachedOutShape = nullptr;
      if (slot.shapeCacheValid() &&
          outIdx < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) &&
          slot.shapeCache.cachedOutputShapes[outIdx] != nullptr) {
        cachedOutShape = slot.shapeCache.cachedOutputShapes[outIdx];
      } else if (cachedValid) {
        // Shape cache unavailable — EMULATED_REPLAY segments never freeze, so
        // shapeCache is never built for them; demoting here nulls the slot and
        // the REPLAYING-phase all-slots-populated visibility check then fails
        // (longViewChain/EMULATED_REPLAY frozen_replay_5, task #52). The stale
        // view's own shapeInfo is ConstantShapeHelper-owned and remains valid
        // even after its parent DataBuffer died — reuse it as the re-mint
        // target shape so the view is recreated over the CURRENT input instead
        // of being demoted to a null slot.
        cachedOutShape = cached->shapeInfo();
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "VIEW_REFRESH_NO_CACHE_FALLBACK: step %d (%s) outputSlot=%d — "
            "re-minting from the stale view's own shapeInfo (segment never froze)",
            stepIdx, slot.ident.opName.c_str(), outSi);
      }
      if (cachedOutShape == nullptr) {
        // No shape source at all — demoteViewProducer inspects + cleans up.
        demoteViewProducer(stepIdx, outSi, "refresh-no-shape-cache");
        refreshedCount++;
        continue;
      }

      // Validate cached shape dtype before creating a view from it.
      // View ops preserve input dtype, so a cached shape with UNKNOWN or INHERIT dtype
      // is corrupt — fix it from the input before it poisons downstream ops.
      if ((ArrayOptions::dataType(cachedOutShape) == DataType::UNKNOWN ||
           ArrayOptions::dataType(cachedOutShape) == DataType::INHERIT) && input0 != nullptr) {
        DataType inputDt = input0->dataType();
        if (inputDt != DataType::UNKNOWN && inputDt != DataType::INHERIT) {
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "VIEW_DTYPE_FIX: step %d (%s) cached shape had UNKNOWN dtype, "
              "correcting to %s from input",
              stepIdx, slot.ident.opName.c_str(),
              DataTypeUtils::asString(inputDt).c_str());
          cachedOutShape = ConstantShapeHelper::getInstance().createShapeInfo(
              inputDt, shape::order(cachedOutShape),
              static_cast<int>(shape::rank(cachedOutShape)),
              shape::shapeOf(const_cast<LongType*>(cachedOutShape)));
          // Write-back only when the cache entry actually exists — on the
          // no-cache fallback path (never-frozen EMULATED_REPLAY segments)
          // cachedOutputShapes may be empty and this index would be OOB.
          if (slot.shapeCacheValid() &&
              outIdx < static_cast<int>(slot.shapeCache.cachedOutputShapes.size())) {
            slot.shapeCache.cachedOutputShapes[outIdx] = cachedOutShape;
          }
        }
      }

      NDArray* newView = nullptr;
      LongType viewOffset = 0;
      ViewCreateResult vcr = tryCreateViewForSlot(
          stepIdx, slot, input0, cachedOutShape,
          viewInputs.data(), slot.wiring.numInputs,
          &newView, &viewOffset);

      if (vcr != VIEW_CREATED || newView == nullptr) {
        if (newView != nullptr) deferredSlotDeletes_.push_back(newView);  // Defer: inline delete during loop causes heap corruption

        // VIEW_NOT_POSSIBLE means the input is no longer contiguous enough for a
        // zero-copy view (e.g., a permute upstream changed strides).  Demote this
        // slot: clear the view-producer flag so future refreshes skip it, and
        // null out the stale cached output so the normal slot-by-slot execution
        // path recomputes the output with its own allocation.  Without this, the
        // cached NDArray keeps wrapping a freed DataBuffer whose shape metadata
        // decays to UNKNOWN dtype, poisoning downstream ops.
        if (vcr == VIEW_NOT_POSSIBLE) {
          demoteViewProducer(stepIdx, outSi, "refresh-view-not-possible");
          refreshedCount++;
          continue;
        }

        // VIEW_STALE_EMPTY_SHAPE or VIEW_STRIDED_SLICE_FAIL
        demoteViewProducer(stepIdx, outSi, "refresh-view-create-fail");
        refreshedCount++;
        continue;
      }

      auto* cachedDb = cachedValid ? cached->dataBuffer() : nullptr;
      if (cachedValid && cachedDb != nullptr && cachedDb->isValid() &&
          safeHasValidShapeInfo(newView) &&
          newView->dataBuffer() == cachedDb &&
          newView->offset() == cached->offset() &&
          newView->dataType() == cached->dataType() &&
          shape::shapeEquals(newView->shapeInfo(), cached->shapeInfo()) &&
          shape::strideEquals(newView->shapeInfo(), cached->shapeInfo())) {
        deferredSlotDeletes_.push_back(newView);  // Defer: inline delete during loop causes heap corruption (freed chunk reused by next iteration's tryCreateViewForSlot)
        continue;
      }

      // Remove old cached array and defer its deletion until after the loop.
      // The old code deleted inline here, but in deep view chains (8+ sequential
      // permute/reshape ops) the freed memory from `delete cached` can be
      // immediately reused by the `new NDArray(...)` in tryCreateViewForSlot,
      // and heap metadata corruption can propagate to adjacent allocations.
      // Deferring all deletes until after all new views are installed prevents
      // this interleaving.
      outputSlots_[outSi] = nullptr;
      planOwnedArrays_.erase(cached);
      // Guard: only defer-delete if the DataBuffer is still live.
      // cachedValid checks only _shapeInfo validity (safeHasValidShapeInfo).
      // cachedDb->isValid() additionally checks the DataBuffer magic number and
      // closed flag. A GC-destroyed DataBuffer has MAGIC_DESTROYED magic, so
      // isValid()=false even when closed reads as false (garbage from freed memory).
      // Without this guard, a view with a destroyed DataBuffer would reach the
      // post-execution delete queue and corrupt the heap when its destructor
      // touches the already-destroyed DataBuffer.
      if (cachedValid && cachedDb != nullptr && cachedDb->isValid()) {
        deferredSlotDeletes_.push_back(cached);
      }
      // If !cachedValid or DataBuffer is invalid/destroyed, skip delete —
      // the destructor would SIGSEGV reading corrupt _shapeInfo or DataBuffer.

      writeOutputSlot(outSi, newView, "ff-view-install");
      // writeOutputSlot handles slot assignment and plan-owned array tracking.

      // Post-install dtype guard: if the new view ended up with UNKNOWN or INHERIT dtype
      // (heap corruption from `delete cached`, stale ConstantShapeHelper entry,
      // or transient shapeInfo zero-init), fix it in-place from the input's dtype.
      // Without this, the UNKNOWN/INHERIT propagates down the view chain and eventually
      // reaches sizeOfElement() in mmul/matmul shape inference, causing a throw.
      if ((newView->dataType() == DataType::UNKNOWN || newView->dataType() == DataType::INHERIT) &&
          input0 != nullptr &&
          input0->dataType() != DataType::UNKNOWN && input0->dataType() != DataType::INHERIT) {
        const LongType* fixedShape = ConstantShapeHelper::getInstance().createShapeInfo(
            input0->dataType(), shape::order(newView->shapeInfo()),
            static_cast<int>(shape::rank(newView->shapeInfo())),
            shape::shapeOf(newView->shapeInfo()));
        newView->setShapeInfo(const_cast<LongType*>(fixedShape));
        // Also update the cached shape entry so future fast-path lookups see
        // the corrected dtype.
        if (outIdx < static_cast<int>(slot.shapeCache.cachedOutputShapes.size())) {
          slot.shapeCache.cachedOutputShapes[outIdx] = fixedShape;
        }
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "VIEW_POST_INSTALL_DTYPE_FIX: step %d (%s) outputSlot=%d had UNKNOWN dtype "
            "after view install — corrected to %s from input",
            stepIdx, slot.ident.opName.c_str(), outSi,
            DataTypeUtils::asString(input0->dataType()).c_str());
      }

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

      // Reclassify the refreshed view, including parent ref-count bookkeeping.
      // Updating only dataBuffer leaves stale parent/type metadata when a null
      // external-fed parent is re-minted for a new borrower.
      if (slotOwnership_ != nullptr) {
        classifyAndUpdateOwnership(
            slotOwnership_[outSi], newView, outSi,
            externalArrays, numExt,
            outputSlots_, totalOutputSlots_, slotOwnership_);
      }

      slot.bumpGeneration();
      dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
      // arg-generation bump consolidated into the end-of-function markArgsStale()
      // below (fires once when refreshedCount > 0, covering refresh AND demotion).

      DSP_DIAG_SLOT(MEMORY, stepIdx,
          "REFRESH_VIEW_OK: step %d (%s) outputSlot=%d view refreshed arr=%p db=%p offset=%lld",
          stepIdx, slot.ident.opName.c_str(), outSi,
          static_cast<void*>(newView),
          static_cast<void*>(newView->dataBuffer()),
          static_cast<long long>(viewOffset));

      refreshedCount++;
    }
  }

  // SELF-CONTAINED arg invalidation (single source of truth): if we refreshed or
  // demoted ANY view wrapper, the slot's device addresses changed, so the cached
  // arg table and any captured graph node arguments are stale. Mark args stale
  // HERE so EVERY caller re-refreshes correctly. Previously each of the four
  // callers had to remember the bump+reset triplet on a >0 return; two of them
  // (cuda.cu frozen fast path, NativeDynamicShapePlan.cpp execute path) silently
  // ignored it, risking a stale-address graph replay (CUDA error 700).
  if (refreshedCount > 0) {
    seg.exec.markArgsStale();
  }

  return refreshedCount;
}

// ─── Frozen constant detection ──────────────────────────────────────────────
// After the warmup execution (executeCount_ just went from 0 to 1), identify
// Compute per-slot transitive variable dependency: a slot transitively depends
// on a variable ext input if it directly reads from one, or if any of its input
// slots transitively depend on one. Uses a single forward pass since slots are
// topologically ordered (downstream slots have higher indices).
//
// Called from NativePlanCompiler after externalInputIsVariable_ is populated,
// from deserialization, and from markExternalInputVariable() when the variable
// set changes at runtime.
void NativeDynamicShapePlan::computeSlotVariableDependency() {
  slotDependsOnVariableExtInput_.assign(numSlots_, false);
  if (externalInputIsVariable_.empty()) return;

  // Build output-slot-index → step-index mapping for O(1) lookup.
  std::vector<int> outputSlotToStep(totalOutputSlots_, -1);
  for (int s = 0; s < numSlots_; s++) {
    auto& slot = slots_[s];
    for (int oi = 0; oi < slot.wiring.numOutputs; oi++) {
      int outIdx = slot.wiring.outputSlotIndices[oi];
      if (outIdx >= 0 && outIdx < totalOutputSlots_) {
        outputSlotToStep[outIdx] = s;
      }
    }
  }

  // Forward pass: slots are topologically ordered, so upstream deps are already computed.
  for (int s = 0; s < numSlots_; s++) {
    auto& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        // Direct external input
        int extIdx = -(srcIdx + 1);
        if (extIdx < static_cast<int>(externalInputIsVariable_.size()) &&
            externalInputIsVariable_[extIdx]) {
          slotDependsOnVariableExtInput_[s] = true;
          break;
        }
      } else if (srcIdx < totalOutputSlots_) {
        // Input from another slot — propagate transitively via output-slot-to-step map.
        int producerStep = outputSlotToStep[srcIdx];
        if (producerStep >= 0 && slotDependsOnVariableExtInput_[producerStep]) {
          slotDependsOnVariableExtInput_[s] = true;
          break;
        }
      }
    }
  }
}

// slots whose output never changes between decode steps. These slots are
// skipped entirely during subsequent executions (including graph capture),
// removing their kernels, memsets, and memcpys from the captured graph.

void NativeDynamicShapePlan::detectFrozenConstants() {
  // Allow re-detection at any executeCount_ >= 1 (not just == 1).
  // markExternalInputVariable() clears frozenConstantDetectionDone_ when a new
  // variable is registered (e.g. KV cache inputs from autoregressive_decode).
  // With the old executeCount_ != 1 guard, re-detection was blocked after
  // executeCount_ incremented past 1, leaving K/V-preprocessing slots (e.g.
  // RoPE on K, layer-norm on K) incorrectly sealed as FROZEN_CONSTANT.
  // On Triton composite replay, frozen-constant slots are skipped — so the
  // attention island read stale warmup-time K/V outputs and produced wrong
  // tokens starting at the first captured replay (decode step 2 / test step 4).
  // Fix: allow re-detection whenever the done flag was explicitly cleared by
  // markExternalInputVariable(), as long as at least one warmup execution has
  // run (executeCount_ >= 1 guarantees slot outputs are populated).
  if (planLifecycle_.isSlotBySlot() || executeCount_ < 1 || frozenConstantDetectionDone_) return;
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

  // Propagate runtime dependency through the graph (topological order).
  // Tensor-external dependencies and implicit mutable execution state both make
  // an output live. Value-independent shape ops remain constant when shapes freeze.
  int shapeOnlyCount = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];

    const auto* descriptor = sl.ident.op != nullptr ? sl.ident.op->getOpDescriptor() : nullptr;
    if (descriptor != nullptr && descriptor->hasAnyTrait(sd::ops::OP_TRAIT_STATEFUL)) {
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          dependsOnExternal[si] = true;
        }
      }
      continue;
    }

    // Check if this op is value-independent via trait.
    bool isShapeOnly = descriptor != nullptr &&
                       descriptor->hasAnyTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT);
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
        // Safety check: if externalInputIsVariable_ is non-empty but has ZERO true
        // entries, markExternalInputVariable() has not been called for any decode-step
        // inputs (input_ids, position_ids, attention_mask, etc.). This happens when
        // the Java side populates the vector but forgets to call markExternalInputVariable
        // before detectFrozenConstants() runs. Without this guard, every external-input
        // slot is treated as a constant and ALL 2419 slots get sealed as FROZEN_CONSTANT,
        // causing CUDA graph capture to see 0 nodes.
        {
          bool hasAnyVariable = false;
          for (bool v : externalInputIsVariable_) {
            if (v) { hasAnyVariable = true; break; }
          }
          if (!hasAnyVariable) {
            sd_debug("DSP WARN: detectFrozenConstants: externalInputIsVariable_ has %d entries but NONE are true. "
                      "markExternalInputVariable() likely not called for decode inputs. "
                      "Falling back to conservative (all externals variable) to prevent incorrect freezing.\n",
                      (int)externalInputIsVariable_.size());
            anyInputDependsOnExternal = true;
            break;
          }
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

  // ── Single-pass slot classification ────────────────────────────────────────
  //
  // Every slot is classified in ONE forward pass (topological order guarantees
  // all inputs are classified before the slot itself). The decision is IMMUTABLE.
  // No unfreezing, no correction passes, no mutable state.
  //
  // Classification:
  //
  //   FROZEN_CONSTANT — dependsOnExternal is false for ALL of this slot's output
  //     indices AND the op is not dynamicShape. This means every transitive input
  //     is structurally stable: external constants, weights, shape-only ops, or
  //     other frozen constants. The dependsOnExternal propagation (above) already
  //     handled data-dependent ops correctly — if a concat reads a variable
  //     external (KV cache), dependsOnExternal propagates through it. If all its
  //     inputs are frozen constants, dependsOnExternal stays false and the concat
  //     IS safe to freeze (its inputs never change, so neither does its output).
  //
  //     Buffer invariants:
  //       - Populated once during warmup, read-only thereafter
  //       - In-place writes to this buffer are BANNED (enforced below)
  //       - Buffer must remain allocated for plan lifetime
  //
  //   LIVE — anything not frozen. Must execute every step.
  //     Buffer invariants:
  //       - May be overwritten each step
  //       - Sync per needsSync() policy
  //
  // Why no view-alias unfreezing:
  //   View ops share their input's DataBuffer. If the input is from an unstable
  //   source, dependsOnExternal propagates through the view op — it won't be
  //   frozen. If the input is from a frozen source, the shared buffer is read-only.
  //   The only corruption path is in-place writes, which are blocked by the
  //   in-place protection below. No separate view-alias pass needed.
  //
  // Why no value-dep unfreezing:
  //   If a value-dep op reads shape data from a frozen upstream, that data is
  //   correct and stable (frozen = never changes). Unfreezing the upstream is
  //   unnecessary — it would just re-execute to produce the same result.

  int frozenConstCount = 0;
  int valueIndepCount = 0;
  std::unordered_set<int> frozenOutputSlots;

  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];

    // Check: do all output slot fingerprints prove stability?
    // dependsOnExternal[si] == true means a variable external is in the
    // transitive input chain (propagated topologically above).
    bool allOutputsConstant = true;
    for (int o = 0; o < sl.wiring.numOutputs; o++) {
      int si = sl.wiring.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && dependsOnExternal[si]) {
        allOutputsConstant = false;
        break;
      }
    }

    // Freeze decision (immutable — never revisited)
    if (allOutputsConstant && !sl.flags.isDynamicShape) {
      // ── Pre-freeze validation ─────────────────────────────────────────
      // Every output slot MUST have a populated, valid buffer. Freezing a slot
      // with null/closed outputs means executeSlot will skip it and downstream
      // ops read garbage. Throw immediately — this is a lifecycle bug, not a
      // recoverable condition.
      bool outputsValid = true;
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
        if (si < 0 || si >= totalOutputSlots_) continue;
        NDArray* arr = outputSlots_[si];
        if (arr == nullptr) {
          DSP_DIAG(SHAPE,
              "FREEZE_SKIP_NULL_OUTPUT: slot %d (%s) output[%d] (outputSlot=%d) is null at "
              "freeze time (executeCount=%d). Slot will remain LIVE and re-execute each step.",
              s, sl.ident.opName.c_str(), o, si, executeCount_);
          outputsValid = false;
          break;
        }
        auto* db = arr->dataBuffer();
        if (db == nullptr || db->isClosed()) {
          DSP_DIAG(SHAPE,
              "FREEZE_SKIP_INVALID_BUFFER: slot %d (%s) output[%d] (outputSlot=%d) has %s "
              "DataBuffer at freeze time (executeCount=%d). Slot will remain LIVE and "
              "re-execute each step.",
              s, sl.ident.opName.c_str(), o, si,
              db == nullptr ? "null" : "closed", executeCount_);
          outputsValid = false;
          break;
        }
      }
      if (!outputsValid) continue;  // Skip freeze — slot stays LIVE

      sl.slotPhase.seal(true);
      frozenConstCount++;
      if (isValueIndependentSlot[s]) valueIndepCount++;

      // ── Snapshot primary buffer pointers at freeze time ────────────────
      // Stored in frozenOutputPtrs (one entry per output ordinal). On subsequent
      // executions, if a buffer's primary pointer changes (reallocation), the
      // frozen skip falls through to re-execute rather than returning stale data.
      // This handles ops like rank/size whose output buffer may be reallocated
      // between the warmup execution and later decode steps.
      sl.frozenOutputPtrs.resize(sl.wiring.numOutputs, nullptr);
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          frozenOutputSlots.insert(si);
          NDArray* arr = outputSlots_[si];
          if (arr != nullptr && arr->dataBuffer() != nullptr) {
            sl.frozenOutputPtrs[o] = arr->dataBuffer()->primary();
          }
        }
      }
    }
    // else: LIVE — slot state stays default (not sealed, not frozen)
  }

  // ── View-alias unfreezing ──────────────────────────────────────────────
  //
  // View ops (reshape, permute, expand_dims) share their input's DataBuffer.
  // If a frozen slot's output buffer is the SAME DataBuffer as a non-frozen
  // slot's output, the frozen skip becomes unsafe: the non-frozen slot writes
  // to the shared buffer every step, but the frozen slot never re-executes
  // to pick up the new data. This commonly happens with attention mask view
  // chains — the mask is variable (updated per step) but a reshape wrapping
  // it appears constant from the dependency graph (its shape inputs are fixed).
  //
  // Unfreeze any frozen slot whose DataBuffer is shared with a live slot.
  int viewAliasUnfrozen = 0;
  {
    // Collect DataBuffers from all non-frozen output slots
    std::unordered_set<void*> liveBuffers;
    for (int si = 0; si < totalOutputSlots_; si++) {
      if (frozenOutputSlots.count(si)) continue;
      NDArray* arr = outputSlots_[si];
      if (arr != nullptr && arr->dataBuffer() != nullptr) {
        liveBuffers.insert(static_cast<void*>(arr->dataBuffer()));
      }
    }

    // Check frozen slots for shared buffers with live slots
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      if (!sl.frozenConstantSlot()) continue;

      bool sharedWithLive = false;
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
        if (si < 0 || si >= totalOutputSlots_) continue;
        NDArray* arr = outputSlots_[si];
        if (arr != nullptr && arr->dataBuffer() != nullptr &&
            liveBuffers.count(static_cast<void*>(arr->dataBuffer()))) {
          sharedWithLive = true;
          break;
        }
      }

      if (sharedWithLive) {
        sl.slotPhase.unseal();
        sl.frozenOutputPtrs.clear();  // No longer frozen — clear stale snapshot
        frozenConstCount--;
        viewAliasUnfrozen++;
        for (int o = 0; o < sl.wiring.numOutputs; o++) {
          int si = sl.wiring.outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_) {
            frozenOutputSlots.erase(si);
          }
        }
        DSP_DIAG(SHAPE,
            "VIEW_ALIAS_UNFREEZE: slot %d (%s) shares DataBuffer with live slot — unfrozen",
            s, sl.ident.opName.c_str());
      }
    }
  }

  // ── Value-dep upstream unfreezing ──────────────────────────────────────
  //
  // Ops with outputShapeDependsOnInputValues read shape data from their
  // inputs (e.g. reshape_no_copy reads a shape tensor). If the shape tensor
  // comes from a frozen slot, and that frozen slot's cached output carries
  // stale or UNKNOWN dtype, the downstream op's calculateOutputShape will
  // produce wrong shapes or trigger KERNEL_FAILURE. Unfreeze frozen slots
  // that feed value-dep ops.
  int valueDepUnfrozen = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    if (!sl.flags.outputShapeDependsOnInputValues) continue;

    for (int i = 0; i < sl.wiring.numInputs; i++) {
      int srcIdx = sl.wiring.inputSourceIndices[i];
      if (srcIdx < 0 || srcIdx >= totalOutputSlots_) continue;
      if (!frozenOutputSlots.count(srcIdx)) continue;

      // Find the producing slot and unfreeze it
      for (int ps = 0; ps < numSlots_; ps++) {
        auto& psl = slots_[ps];
        if (!psl.frozenConstantSlot()) continue;
        for (int o = 0; o < psl.wiring.numOutputs; o++) {
          if (psl.wiring.outputSlotIndices[o] == srcIdx) {
            psl.slotPhase.unseal();
            psl.frozenOutputPtrs.clear();  // No longer frozen — clear stale snapshot
            frozenConstCount--;
            valueDepUnfrozen++;
            for (int o2 = 0; o2 < psl.wiring.numOutputs; o2++) {
              int si2 = psl.wiring.outputSlotIndices[o2];
              if (si2 >= 0 && si2 < totalOutputSlots_) {
                frozenOutputSlots.erase(si2);
              }
            }
            DSP_DIAG(SHAPE,
                "VALUE_DEP_UNFREEZE: slot %d (%s) feeds value-dep slot %d (%s) — unfrozen",
                ps, psl.ident.opName.c_str(), s, sl.ident.opName.c_str());
            goto next_input;
          }
        }
      }
      next_input:;
    }
  }

  // ── In-place protection (single pass) ────────────────────────────────────
  //
  // In-place fusion writes the op's output directly into its input buffer.
  // This MUST be disabled when the input buffer is protected:
  //
  //   1. Frozen constant buffer — must remain stable for frozen skip
  //   2. Value-dep op input — shape data must survive until the op reads it
  //   3. Multi-consumer buffer — other consumers haven't read it yet
  //
  // These three checks are computed inline. No separate "protected slots" set
  // needed — each check directly inspects the source slot's state.

  // Pre-compute: consumer count per output slot and value-dep input set
  std::vector<int> slotConsumerCount(totalOutputSlots_, 0);
  std::unordered_set<int> valueDepInputSlots;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    for (int i = 0; i < sl.wiring.numInputs; i++) {
      int srcIdx = sl.wiring.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        slotConsumerCount[srcIdx]++;
        if (sl.flags.outputShapeDependsOnInputValues) {
          valueDepInputSlots.insert(srcIdx);
        }
      }
    }
  }

  // Build requested output set — in-place fusion must not overwrite a slot
  // whose value the caller explicitly asked for (e.g. sd.outputAll()).
  std::unordered_set<int> requestedOutputSlotSet;
  if (requestedOutputSlotIndices_ != nullptr) {
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int si = requestedOutputSlotIndices_[i];
      if (si >= 0 && si < totalOutputSlots_) {
        requestedOutputSlotSet.insert(si);
      }
    }
  }

  int disabledInPlace = 0;
  for (int s = 0; s < numSlots_; s++) {
    auto& sl = slots_[s];
    int srcSlot = sl.inPlaceSourceSlot();
    if (srcSlot < 0) continue;

    bool mustDisable =
        frozenOutputSlots.count(srcSlot) ||        // frozen buffer: read-only
        valueDepInputSlots.count(srcSlot) ||        // value-dep input: must survive
        slotConsumerCount[srcSlot] > 1 ||           // multi-consumer: others need it
        requestedOutputSlotSet.count(srcSlot);      // user-requested output: must preserve

    if (mustDisable) {
      sl.disableInPlaceFusion();
      disabledInPlace++;
    }
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────
  int javaManagedCount = 0;
  for (int si = 0; si < totalOutputSlots_; si++) {
    if (!producedByPlanOp[si]) javaManagedCount++;
  }

  DSP_DIAG(SHAPE, "frozen constant detection: %d/%d slots frozen "
            "(%d shape-only-trait, %d value-independent, "
            "%d view-alias-unfrozen, %d value-dep-unfrozen, "
            "%d in-place disabled, %d java-managed output slots)",
            frozenConstCount, numSlots_, shapeOnlyCount, valueIndepCount,
            viewAliasUnfrozen, valueDepUnfrozen,
            disabledInPlace, javaManagedCount);

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
    // Composite/direct replay suppresses generic D2H traffic because ordinary
    // frozen-shape ops only need cached shapes. Value-dependent shape functions
    // are the exception: their small integral control tensors may have just been
    // produced on the DSP stream, and calculateOutputShape reads their literal
    // values on the host. Temporarily lift replay suppression for only those
    // bounded controls; the normal graph-capture guard still forbids D2H while a
    // CUDA graph is actually being captured.
    DspReplayGuard valueShapeHostReadGuard(false);
    for (int i = 0; i < numInputs; i++) {
      if (inputs[i] == nullptr) continue;
      auto dt = inputs[i]->dataType();
      auto len = inputs[i]->lengthOf();
      if ((dt == INT32 || dt == INT64 || dt == BOOL) && len > 0 && len <= 32) {
        std::vector<NDArray*> hostReads{inputs[i]};
        NDArray::preparePrimaryUse({}, hostReads);
        for (LongType j = 0; j < len; j++) {
          if (dt == BOOL) {
            mix(static_cast<LongType>(inputs[i]->e<bool>(j)));
          } else {
            mix(inputs[i]->e<LongType>(j));
          }
        }
        NDArray::registerPrimaryUse({}, hostReads);
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
  if (DSP_DIAG_ENABLED(MEMORY)) {
    // Count and log which output slots will be zeroed
    int zeroCount = 0;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < numSlots_; s++) {
      const NativeSlot& sl = slots_[s];
      if (!sl.needsPrezero()) continue;
      for (int o = 0; o < sl.wiring.numOutputs; o++) {
        int si = sl.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          zeroCount++;
        }
      }
    }
    DSP_DIAG(MEMORY,
        "PREZERO_SEGMENT: segStart=%d segEnd=%d zeroCount=%d executeCount=%d",
        seg.def.startSlot, seg.def.endSlot, zeroCount, executeCount_);
  }
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
    if (!slot.needsPrezero()) return;
    GraphSegment zeroSeg;
    zeroSeg.def.startSlot = stepIdx;
    zeroSeg.def.endSlot = stepIdx;
    prezeroSegmentOutputs(zeroSeg, stream);
  };

  // Trace slot entry during warmup or when sync override is active (post-markVariable).
  // The syncOverrideDepth_ > 0 check catches post-markVariable warmup where
  // executeCount_ is high but segment-level executionCount has been reset.
  if ((executeCount_ < 2 || syncOverrideDepth_ > 0) && DSP_DIAG_ENABLED(SHAPE)) {
    DSP_DIAG_SLOT(SHAPE, stepIdx,
        "EXEC_SLOT_ENTRY: slot %d (%s) state=%s frozenContextReady=%d shapeValid=%d "
        "isViewCapable=%d isDataDep=%d outputShapeDep=%d execCount=%d shapesFrozen=%d "
        "syncOverride=%d",
        stepIdx, slot.ident.opName.c_str(),
        slot.slotPhase.displayName(),
        slot.frozenContextReady() ? 1 : 0,
        slot.shapeCacheValid() ? 1 : 0,
        slot.isViewCapableOp() ? 1 : 0,
        slot.isDataDependent() ? 1 : 0,
        slot.flags.outputShapeDependsOnInputValues ? 1 : 0,
        executeCount_,
        planLifecycle_.isShapesFrozen() ? 1 : 0,
        syncOverrideDepth_);
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
                 executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0, planLifecycle_.displayName());
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
                   executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0, planLifecycle_.displayName());
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

  // ── Closed-buffer guard for slot-sourced (variable) inputs ───────────────
  // The full input validation above is skipped for executeCount_ >= 3 to
  // avoid ~7500 pointer checks per decode step. However, a weight variable
  // that is closed (freed) between replay calls produces a CLOSED DataBuffer
  // in outputSlots_[] — the frozen fast-path would read it and trigger CUDA
  // error 700 (illegal memory access) in the downstream kernel (e.g., CUTLASS
  // GEMM). Detect this here, outside both the skip gate and the frozen path,
  // so it fires on EVERY execution count for slot-sourced inputs only.
  // Cost: one null/isClosed() check per slot input per execute — negligible.
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
      NDArray* slotArr = outputSlots_[srcIdx];
      if (slotArr != nullptr) {
        DataBuffer* db = slotArr->dataBuffer();
        if (db != nullptr && db->isClosed()) {
          const char* closedPhaseStr = "UNKNOWN";
          if (planLifecycle_.isSlotBySlot()) {
            closedPhaseStr = "SLOT_BY_SLOT";
          } else if (planLifecycle_.isShapesFrozen()) {
            closedPhaseStr = (executeCount_ == 0) ? "FROZEN_WARMUP" :
                             (executeCount_ < 3)  ? "FROZEN_CAPTURE" : "FROZEN_REPLAY";
          } else {
            closedPhaseStr = "UNFROZEN_WARMUP";
          }
          DSP_THROW(EXECUTE,
                   "NativeDynamicShapePlan: CLOSED_SLOT_BUFFER: slot %d (%s) input[%d] "
                   "from outputSlots_[%d] has a CLOSED DataBuffer during %s phase "
                   "(execCount=%d). The weight variable was freed between replay calls "
                   "without invalidating/recapturing the plan. "
                   "Use associateArrayWithVariable with a live replacement BEFORE closing "
                   "the prior weight array, and ensure C++ slot is updated before replay.",
                   stepIdx, slot.ident.opName.c_str(), i, srcIdx,
                   closedPhaseStr, executeCount_);
          return Status::BAD_INPUT;  // unreachable, but keeps return-type analysis clean
        }
      }
    } else if (srcIdx < 0 && externalArrays != nullptr) {
      // Also check external-input variable arrays for closed DataBuffers.
      // Weight variables arrive as external inputs (srcIdx < 0), not via outputSlots_.
      // Without this check, a weight closed between replay calls is silently forwarded
      // to the CUDA graph as a stale/freed device pointer → err700 → context poison.
      // The guard at srcIdx >= 0 above only covers slot-sourced inputs; this covers
      // variable external inputs (externalInputIsVariable_[extIdx] == true).
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt && extIdx < (int)externalInputIsVariable_.size()
          && externalInputIsVariable_[extIdx]) {
        NDArray* extArr = externalArrays[extIdx];
        if (extArr != nullptr) {
          DataBuffer* db = extArr->dataBuffer();
          if (db != nullptr && db->isClosed()) {
            const char* closedPhaseStr = "UNKNOWN";
            if (planLifecycle_.isSlotBySlot()) {
              closedPhaseStr = "SLOT_BY_SLOT";
            } else if (planLifecycle_.isShapesFrozen()) {
              closedPhaseStr = (executeCount_ == 0) ? "FROZEN_WARMUP" :
                               (executeCount_ < 3)  ? "FROZEN_CAPTURE" : "FROZEN_REPLAY";
            } else {
              closedPhaseStr = "UNFROZEN_WARMUP";
            }
            DSP_THROW(EXECUTE,
                     "NativeDynamicShapePlan: CLOSED_EXT_VARIABLE_BUFFER: slot %d (%s) input[%d] "
                     "from external variable extIdx=%d has a CLOSED DataBuffer during %s phase "
                     "(execCount=%d). A weight variable DataBuffer was freed between replay calls "
                     "while the frozen plan still references its GPU address. "
                     "Call associateArrayWithVariable with a live replacement BEFORE closing "
                     "the old weight array so the plan receives a valid pointer on next execute.",
                     stepIdx, slot.ident.opName.c_str(), i, extIdx,
                     closedPhaseStr, executeCount_);
            return Status::BAD_INPUT;  // unreachable, but keeps return-type analysis clean
          }
        }
      }
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
        !dspGetReplayActive() &&
        contextPool_[stepIdx] != nullptr && slot.frozenContextReady() &&
        !slot.isIdentityOp() &&
        !slot.frozenConstantSlot() &&
        !slot.fusedChain.isFusedChainHead &&
        !slot.fusedChain.isFusedChainTail &&
        !slot.flags.isDynamicShape)) {
      // Diagnostic: log why the frozen fast-path was not taken on first eligible execution
      if (DSP_DIAG_ENABLED(EXECUTE) && executeCount_ >= 4 && executeCount_ <= 5) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
            "FF_FAST_PATH_SKIP: slot=%d op=%s frozenContextReady=%d "
            "frozenConst=%d isDynamic=%d isSlotBySlot=%d frozenSlotBySlot=%d "
            "isFusedHead=%d isFusedTail=%d isIdentity=%d executeCount=%d "
            "syncOverrideDepth=%d replayActive=%d",
            stepIdx, slot.ident.opName.c_str(),
            slot.frozenContextReady() ? 1 : 0,
            slot.frozenConstantSlot() ? 1 : 0,
            slot.flags.isDynamicShape ? 1 : 0,
            planLifecycle_.isSlotBySlot() ? 1 : 0,
            frozenSlotBySlot ? 1 : 0,
            slot.fusedChain.isFusedChainHead ? 1 : 0,
            slot.fusedChain.isFusedChainTail ? 1 : 0,
            slot.isIdentityOp() ? 1 : 0,
            executeCount_, syncOverrideDepth_,
            dspGetReplayActive() ? 1 : 0);
      }
      break;
    }

    // Log entry into super fast frozen path on first eligible execution
    if (DSP_DIAG_ENABLED(EXECUTE) && executeCount_ == 4) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
          "FF_FAST_PATH_ENTER: slot=%d op=%s phase=%s frozenContextReady=%d "
          "executeCount=%d syncOverrideDepth=%d isViewCapable=%d",
          stepIdx, slot.ident.opName.c_str(),
          slot.slotPhase.displayName(),
          slot.frozenContextReady() ? 1 : 0,
          executeCount_, syncOverrideDepth_,
          slot.isViewCapableOp() ? 1 : 0);
    }

    // Skip frozen fast-path for slots that DIRECTLY read from variable ext inputs.
    // Variable ext inputs (KV caches, embeddings, masks) change every decode step,
    // so the cached frozen output from the previous step is stale.
    //
    // NOTE: We check DIRECT ext input reads only, NOT transitive dependencies.
    // In composite replay, downstream slots get updated values through the slot chain
    // naturally (gap ops execute live, island replays use updated arg tables).
    // Transitive checking marks virtually ALL slots as variable-dependent in a
    // transformer model (everything is downstream of KV cache / mask), which
    // disables the frozen fast-path entirely and causes a 2.4x perf regression.
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
      if (hasVariableExtInput && !slot.isViewCapableOp()) {
        // Var-ext skip is about kernel-arg stability and does NOT apply to
        // view-capable slots: a view over a variable external MUST reach the
        // view-verify/re-mint section below every exec. Breaking out here left
        // the slot holding the PREVIOUS exec's view over a since-closed input
        // (test closes inputs per exec) — a dead view surviving into REPLAYING
        // that reads as length-0/"null slot" and feeds downstream consumers
        // stale memory (the [cast]-failure close-stress family).
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
            "FF_FAST_PATH_SKIP_VAR_EXT: slot=%d op=%s has variable ext input — falling through to normal path executeCount=%d",
            stepIdx, slot.ident.opName.c_str(), executeCount_);
        break;
      }
    }

    // View-capable ops: verify the installed view still shares a buffer with
    // the current input. If the upstream gap op replaced the input array with
    // a new allocation, the view is stale and we must fall through.
    if (slot.isViewCapableOp() && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
      int outSi = slot.wiring.outputSlotIndices[0];
      if (outSi < 0 || outSi >= totalOutputSlots_) break;

      NDArray* currentOut = outputSlots_[outSi];
      // Resolve input0 via resolveViewInput — prefer staging buffer for
      // external inputs so views alias stable plan-owned addresses.
      int srcIdx = slot.wiring.inputSourceIndices[0];
      NDArray* input0 = resolveViewInput(srcIdx, externalArrays, numExt);
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
          ffViewInputs[ii] = resolveViewInput(iiSrc, externalArrays, numExt);
        }

        // Fix UNKNOWN dtype in cached shape before view creation (frozen fast path)
        const LongType* ffCachedShape = slot.shapeCache.cachedOutputShapes[0];
        if (ffCachedShape != nullptr &&
            ArrayOptions::dataType(ffCachedShape) == DataType::UNKNOWN &&
            input0 != nullptr && input0->dataType() != DataType::UNKNOWN) {
          ffCachedShape = ConstantShapeHelper::getInstance().createShapeInfo(
              input0->dataType(), shape::order(ffCachedShape),
              static_cast<int>(shape::rank(ffCachedShape)),
              shape::shapeOf(const_cast<LongType*>(ffCachedShape)));
          slot.shapeCache.cachedOutputShapes[0] = ffCachedShape;
        }

        NDArray* ffNewView = nullptr;
        LongType ffViewOffset = 0;
        ViewCreateResult ffVcr = tryCreateViewForSlot(
            stepIdx, slot, input0, ffCachedShape,
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
            deferredSlotDeletes_.push_back(ffNewView);  // Defer: inline delete during slot execution causes heap corruption
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
          markViewProducer(stepIdx, "ff-view-install");
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
          }
          break;  // fall through to normal path
        } else if (ffVcr == VIEW_STALE_EMPTY_SHAPE) {
          slot.slotPhase.reset();  // PRIMARY
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
                     executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
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
                       executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
              THROW_EXCEPTION(msg);
            }
          }
        }
      }

      bool ffClosedBuffer = false;
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
            ffCtx.setInputArray(i, externalArrays[extIdx]);
          }
        } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          if (outputSlots_[srcIdx] != nullptr) {
            NDArray* slotArr = outputSlots_[srcIdx];
            DataBuffer* db = slotArr->dataBuffer();
            // If this slot-sourced input's DataBuffer was closed (e.g., weight
            // rebound + freed between calls via associateArrayWithVariable +
            // close()), the frozen fast-path cannot be used safely — the cached
            // NDArray pointer carries a freed GPU buffer, which causes CUDA
            // error 700 (illegal memory access) in the downstream kernel.
            // Break out of the do-while to fall through to the normal execution
            // path, which will re-resolve inputs from fresh external arrays.
            if (db != nullptr && db->isClosed()) {
              DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "FF_FAST_PATH_CLOSED_BUFFER: slot=%d op=%s input[%d] "
                  "srcIdx=%d has CLOSED DataBuffer — weight freed between "
                  "replays; falling through to normal path. execCount=%d",
                  stepIdx, slot.ident.opName.c_str(), i, srcIdx, executeCount_);
              ffClosedBuffer = true;
              break;
            }
            ffCtx.setInputArray(i, slotArr);
          }
        }
      }
      if (ffClosedBuffer) break;  // exit do-while → fall through to normal path

      // Composite replay gap ops: graph replay writes device memory directly
      // without updating actuality flags (no registerSpecialUse in CUDA graph
      // replay). SlotSyncGuard handles prepare (here) + register (on scope exit).
      bool gapSync = needsSync();
      if ((executeCount_ <= 2 || executeCount_ == 4) && DSP_DIAG_ENABLED(STREAM_SYNC)) {
        DSP_DIAG_SLOT(STREAM_SYNC, stepIdx,
                      "FF_FAST_PATH_SYNC: slot=%d op=%s sync=%s reason=%s exec=%d overrideDepth=%d",
                      stepIdx, slot.ident.opName.c_str(),
                      gapSync ? "YES" : "NO", syncReason(),
                      executeCount_, syncOverrideDepth_);
      }
      SlotSyncGuard gapSyncGuard(gapSync, ffCtx.fastpath_out(), ffCtx.fastpath_in(),
                                  stepIdx, slot.ident.opName.c_str(), executeCount_);

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

      // Pre-execute UNKNOWN dtype guard: if any input has UNKNOWN dtype, fix it
      // from its source's cached shape. This catches residual UNKNOWN from heap
      // corruption in view refresh or stale ConstantShapeHelper entries. Without
      // this, sizeOfElement(UNKNOWN) throws inside op shape inference / prepareOutputs.
      for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
        NDArray* inArr = (ii < static_cast<int>(ffCtx.fastpath_in().size()))
                             ? ffCtx.fastpath_in()[ii] : nullptr;
        if (inArr != nullptr && inArr->dataType() == DataType::UNKNOWN) {
          // Try to derive correct dtype from the slot's cached output shapes
          // of the PRODUCING slot for this input.
          int srcIdx = slot.wiring.inputSourceIndices[ii];
          DataType fixDt = DataType::FLOAT32;  // last-resort default
          if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
            // Find the producing slot for this output index
            int prodStep = dsp::findProducingStepForOutputSlot(slots_, numSlots_, srcIdx);
            if (prodStep >= 0 && !slots_[prodStep].shapeCache.cachedOutputShapes.empty()) {
              for (int oi = 0; oi < slots_[prodStep].wiring.numOutputs; oi++) {
                if (slots_[prodStep].wiring.outputSlotIndices[oi] == srcIdx &&
                    oi < static_cast<int>(slots_[prodStep].shapeCache.cachedOutputShapes.size())) {
                  auto dt = ArrayOptions::dataType(slots_[prodStep].shapeCache.cachedOutputShapes[oi]);
                  if (dt != DataType::UNKNOWN) { fixDt = dt; break; }
                }
              }
            }
          } else if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
              fixDt = externalArrays[extIdx]->dataType();
            }
          }
          const LongType* fixedSI = ConstantShapeHelper::getInstance().createShapeInfo(
              fixDt, shape::order(inArr->shapeInfo()),
              static_cast<int>(shape::rank(inArr->shapeInfo())),
              shape::shapeOf(inArr->shapeInfo()));
          inArr->setShapeInfo(const_cast<LongType*>(fixedSI));
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "INPUT_DTYPE_FIX: slot %d (%s) input[%d] had UNKNOWN dtype — corrected to %s",
              stepIdx, slot.ident.opName.c_str(), ii,
              DataTypeUtils::asString(fixDt).c_str());
        }
      }

      // Pre-execute INPUT corruption scan. The OUTPUT scan above only protects
      // this slot's outputs; an op that reads a freed/corrupt INPUT shapeInfo
      // (e.g. a prematurely-released external-input alias — see resize_nearest_neighbor)
      // otherwise faults DEEP inside the op kernel with a cryptic
      // "ConstantShapeBuffer::primary(): invalid magic" and no provenance.
      // Validate inputs at this clean DSP boundary so the producer is named.
      // Mirrors the output scan: gated on isDebug for steady-state perf, and uses
      // the crash-safe shape check so a corrupt input cannot crash the guard itself.
      if (sd::Environment::getInstance().isDebug()) {
        for (int ii = 0; ii < slot.wiring.numInputs &&
                         ii < static_cast<int>(ffCtx.fastpath_in().size()); ii++) {
          NDArray* inArr = ffCtx.fastpath_in()[ii];
          if (inArr != nullptr && !safeHasValidShapeInfo(inArr)) {
            const int srcIdx = slot.wiring.inputSourceIndices[ii];
            char msg[640];
            snprintf(msg, sizeof(msg),
                     "CORRUPTION_BEFORE_EXEC_INPUT: slot %d (%s) input[%d] has INVALID "
                     "shapeInfo BEFORE op->execute — its producer's output was freed/"
                     "corrupted before this consumer read it. arr=%p db=%s srcIdx=%d (%s) "
                     "execCount=%d",
                     stepIdx, slot.ident.opName.c_str(), ii, (void*)inArr,
                     inArr->dataBuffer() == nullptr ? "null" : "present",
                     srcIdx, srcIdx < 0 ? "EXTERNAL" : "slot-produced", executeCount_);
            THROW_EXCEPTION(msg);
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
        ffStatus = platformExecuteSlot(slot, ffCtx);
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

      // registerSpecialUse handled by gapSyncGuard destructor

      auto& ffOuts = ffCtx.fastpath_out();

      // Targeted per-slot output fingerprint — active when debug+verbose
      // Only fingerprints at exec count >= 2 to avoid flooding during warmup
      if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose()) {
        for (int i = 0; i < slot.wiring.numOutputs && i < static_cast<int>(ffOuts.size()); i++) {
          if (ffOuts[i] != nullptr && !ffOuts[i]->isEmpty() && ffOuts[i]->lengthOf() > 0) {
            DSP_DIAG(VERIFY,
                     "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld asyncValues=true",
                     executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                     ShapeUtils::shapeAsString(ffOuts[i]).c_str(),
                     DataTypeUtils::asString(ffOuts[i]->dataType()).c_str(),
                     (long long)ffOuts[i]->lengthOf());
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
                         executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
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
        // Completion diagnostic for fast-frozen non-view path
        if (DSP_DIAG_ENABLED(EXECUTE) && executeCount_ == 4) {
          std::string outSlotStr;
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            if (i > 0) outSlotStr += ",";
            outSlotStr += std::to_string(slot.wiring.outputSlotIndices[i]);
          }
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
              "FF_FAST_PATH_DONE: slot=%d op=%s status=OK outSlots=[%s] executeCount=%d",
              stepIdx, slot.ident.opName.c_str(), outSlotStr.c_str(), executeCount_);
        }
      } else {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
            "FF_FAST_PATH_FAIL: slot=%d op=%s status=%d executeCount=%d",
            stepIdx, slot.ident.opName.c_str(), static_cast<int>(ffStatus), executeCount_);
      }
      return ffStatus;
    }
  } while (false);

  // Diagnostic: normal (non-frozen-fast) execution path entry.
  // Only log during early executions to avoid per-step overhead at steady state.
  if (DSP_DIAG_ENABLED(EXECUTE) && executeCount_ <= 2) {
    DSP_DIAG_SLOT(EXECUTE, stepIdx,
        "NORMAL_EXEC_ENTRY: slot=%d op=%s phase=%s frozenContextReady=%d "
        "isIdentity=%d frozenConst=%d isDynamic=%d isViewCapable=%d "
        "executeCount=%d shapesFrozen=%d",
        stepIdx, slot.ident.opName.c_str(),
        slot.slotPhase.displayName(),
        slot.frozenContextReady() ? 1 : 0,
        slot.isIdentityOp() ? 1 : 0,
        slot.frozenConstantSlot() ? 1 : 0,
        slot.flags.isDynamicShape ? 1 : 0,
        slot.isViewCapableOp() ? 1 : 0,
        executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
  }

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
      // Defer deletion until after executeSlot completes to prevent heap
      // corruption.  Inline delete here frees the NDArray memory, which the
      // allocator may immediately reuse for a subsequent op's output or
      // workspace allocation.  If the freed chunk is adjacent to a Workspace
      // object, the allocator's free-list metadata update corrupts the
      // Workspace, producing SIGSEGV in Workspace::allocateBytes with a
      // garbage `this` pointer full of ASCII string data.
      deferredSlotDeletes_.push_back(cached);
      DSP_DIAG(MEMORY, "discardCachedSlotArray: slot=%d tag=%s deferred-delete=%p",
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
    // A reusable CUDA output can be structurally valid yet belong to the
    // device used by an earlier segment. Check provenance during warmup/
    // first-decode, when a segment device transition is expected; later
    // executions keep the hot path free of per-slot cudaGetDevice calls.
    bool platformInvalid = false;
    if (!invalid && executeCount_ <= 1 &&
        !platformValidateReusableSlotBuffer(cached)) {
      invalid = true;
      platformInvalid = true;
    }
    // Ext-fed view slots: the cached wrapper is only current while the
    // EXTERNAL it aliases is still the same DataBuffer OBJECT. When the
    // borrower closed + re-supplied that input (close-stress tests, decode
    // array churn), the cached view survives over the dead buffer — reading
    // as a stale/length-0 view in REPLAYING and feeding downstream consumers
    // freed memory. Compare object identity against the CURRENT external:
    // same object → reuse (stable weights/decode unchanged, zero cost);
    // different object → re-mint over the live input. Blanket re-mint is
    // WRONG (breaks frozen baked-address contracts — 9F/19E regression).
    if (!invalid && slot.isViewCapableOp() && slot.wiring.numInputs >= 1 &&
        slot.wiring.inputSourceIndices[0] < 0) {
      int viewExtIdx = -(slot.wiring.inputSourceIndices[0] + 1);
      if (viewExtIdx >= 0 && viewExtIdx < numExt &&
          externalArrays[viewExtIdx] != nullptr) {
        auto* curDb = externalArrays[viewExtIdx]->dataBuffer();
        if (curDb != nullptr && db != curDb) {
          invalid = true;
        }
      }
    }

    if (!invalid) return cached;

    // View-capable ops legitimately recreate their output wrapper on each
    // execution. When the upstream input buffer changes (fresh placeholder feed
    // or an earlier view in the chain being rebuilt), the previously cached
    // wrapper can point at a closed DataBuffer by the time Step 3 reaches it.
    // Treat that as a cache miss and let the view-install path below recreate
    // the wrapper from current inputs.
    //
    // CONST_GEN ops (min_max_datatype, shape_of, ones_as, etc.) produce
    // deterministic constant outputs. During executeCount_==0 (shape-only or
    // first warmup pass), the output NDArray may be allocated with shape info
    // but no DataBuffer (the op body wasn't executed, or the allocation took
    // the empty-scalar path). At executeCount_==1 the cached array has
    // db==nullptr — treat as cache miss so the op re-executes and populates
    // the buffer. This is safe because CONST_GEN outputs are input-independent.
    if (platformInvalid ||
        slot.isViewCapableOp() ||
        slot.flags.outputShapeDependsOnInputValues ||
        slot.hasOpTrait(sd::ops::OP_TRAIT_CONSTANT_GENERATION) ||
        executeCount_ == 0) {
      // View-capable ops recreate their output wrapper each step.
      // Value-dependent ops (Where, NonZero) produce variable-length output
      // that can be empty (null DataBuffer) one step and non-empty the next.
      // CONST_GEN ops may have null DataBuffer from shape-only warmup pass.
      // In all cases, a stale/empty cached output is a normal cache miss —
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
  if (slot.isIdentityOp() && slot.wiring.numInputs == 1 && slot.wiring.numOutputs >= 1) {
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
      // ── Frozen buffer invariant check ─────────────────────────────────
      // A frozen constant's buffer was populated during warmup and must remain
      // valid for the entire plan lifetime. If ANY output has a null or closed
      // DataBuffer, the freeze classification was wrong or something
      // deallocated the buffer. Throw — no fallback.
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        int si = slot.wiring.outputSlotIndices[o];
        if (si < 0 || si >= totalOutputSlots_) continue;
        NDArray* arr = outputSlots_[si];
        if (arr == nullptr) continue;  // shouldn't happen (allOutputsPopulated), belt-and-suspenders
        auto* db = arr->dataBuffer();
        if (db == nullptr || db->isClosed()) {
          DSP_THROW(EXECUTE,
              "FROZEN_BUFFER_INVALID: step=%d (%s) outputSlot=%d has %s DataBuffer. "
              "Frozen constants must have valid buffers for the plan's lifetime. "
              "This means either: (1) something deallocated a frozen buffer, "
              "(2) the freeze classification was wrong, or "
              "(3) an in-place op overwrote the buffer despite protection. "
              "execCount=%d numSlots=%d",
              stepIdx, slot.ident.opName.c_str(), si,
              db == nullptr ? "null" : "closed",
              executeCount_, numSlots_);
        }
      }

      // ── Buffer reallocation detection ─────────────────────────────────
      // If any output's primary buffer pointer changed since freeze time, the
      // buffer was reallocated and the frozen value is gone. Fall through to
      // re-execute the slot so the output is repopulated with correct data.
      //
      // This fixes rank/size/size_at/shape_of returning 0 on CPU: these ops
      // are classified as frozen constants but their output buffers can be
      // reallocated between the warmup execution and subsequent decode steps.
      // Without this check the skip path returns Status::OK with a zeroed buffer.
      bool frozenPtrChanged = false;
      if (!slot.frozenOutputPtrs.empty()) {
        for (int o = 0; o < slot.wiring.numOutputs && o < (int)slot.frozenOutputPtrs.size(); o++) {
          int si = slot.wiring.outputSlotIndices[o];
          if (si < 0 || si >= totalOutputSlots_) continue;
          NDArray* arr = outputSlots_[si];
          if (arr == nullptr || arr->dataBuffer() == nullptr) continue;
          void* currentPtr = arr->dataBuffer()->primary();
          void* frozenPtr  = slot.frozenOutputPtrs[o];
          if (frozenPtr != nullptr && currentPtr != frozenPtr) {
            DSP_DIAG(SHAPE,
                "FROZEN_PTR_CHANGED: step=%d (%s) output[%d] (outputSlot=%d) "
                "primary ptr changed: frozen=%p current=%p — re-executing slot to repopulate. "
                "execCount=%d",
                stepIdx, slot.ident.opName.c_str(), o, si,
                frozenPtr, currentPtr, executeCount_);
            frozenPtrChanged = true;
            break;
          }
        }
        if (frozenPtrChanged) {
          // Update the frozen pointer snapshot so subsequent steps use the new ptr
          for (int o = 0; o < slot.wiring.numOutputs && o < (int)slot.frozenOutputPtrs.size(); o++) {
            int si = slot.wiring.outputSlotIndices[o];
            if (si < 0 || si >= totalOutputSlots_) continue;
            NDArray* arr = outputSlots_[si];
            if (arr != nullptr && arr->dataBuffer() != nullptr) {
              slot.frozenOutputPtrs[o] = arr->dataBuffer()->primary();
            }
          }
          // Fall through to normal execution (do NOT return here)
        }
      }

      if (!frozenPtrChanged) {
        DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s [SKIPPED:frozen-const]", stepIdx, slot.ident.opName.c_str());
        backfillCachedOutputShapes(slot, outputSlots_, totalOutputSlots_);
        return Status::OK;
      }
    } else {
      // Frozen constant with null output — this should never happen post-warmup.
      // detectFrozenConstants runs at executeCount_==1, and warmup populates all
      // outputs. If we get here, the slot was incorrectly classified as frozen
      // before its outputs were populated.
      DSP_THROW(EXECUTE,
          "FROZEN_NULL_OUTPUT: step=%d (%s) is frozenConstantSlot but has null output "
          "at executeCount=%d. detectFrozenConstants must run AFTER warmup populates "
          "all outputs. This indicates a lifecycle ordering bug.",
          stepIdx, slot.ident.opName.c_str(), executeCount_);
    }
    // If we reach here: allOutputsPopulated && frozenPtrChanged — fall through
    // to normal slot execution below (buffer was reallocated, must re-execute).
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

#ifdef SD_CUDA
    // BUF_FP_RING fused-chain output fingerprint. The fused path returns before
    // the common post-execute hook, so record its real tail output here on the
    // exact LaunchContext stream used by fusedElementwiseChain.
    if (fpRingEnabled_ && output != nullptr && output->specialBuffer() != nullptr &&
        !output->isEmpty()) {
      auto* fpStreamPtr = lc != nullptr ? lc->getCudaStream() : nullptr;
      if (fpStreamPtr != nullptr) {
        void* fpSavedCompositeCaptureStream = tl_compositeCaptureStream;
        int fpStep = execCtx != nullptr ? execCtx->fpStep : executeCount_;
        if (fpSavedCompositeCaptureStream != nullptr) {
          tl_compositeCaptureStream = nullptr;
          fpStep += BUF_FP_POST_CAPTURE_STEP_OFFSET;
        }
        int fpTrack = numExternalInputs_ + lastOutputSlotIdx;
        bool fpRequested = numRequestedOutputs_ > 0 && requestedOutputSlotIndices_ != nullptr &&
                           requestedOutputSlotIndices_[0] == lastOutputSlotIdx;
        if (fpRequested) {
          recordBufFingerprintPublic(*fpStreamPtr, fpStep, BUF_FP_REQUESTED_OUTPUT_TRACK,
                                     output->specialBuffer(), dspSafeByteCount(output));
          if (fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag[0] == '\0') {
            snprintf(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag,
                     sizeof(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag), "req.writer");
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].extIdx = -1;
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].groupIdx = -1;
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].whichAB = -1;
          }
        }
        if (fpTrack >= 0 && fpTrack < BUF_FP_TRACE_TRACK) {
          recordBufFingerprintPublic(*fpStreamPtr, fpStep, fpTrack,
                                     output->specialBuffer(), dspSafeByteCount(output));
          if (fpLabels_[fpTrack].tag[0] == '\0') {
            snprintf(fpLabels_[fpTrack].tag, sizeof(fpLabels_[fpTrack].tag),
                     "s%d", lastOutputSlotIdx);
            fpLabels_[fpTrack].extIdx = -1;
            fpLabels_[fpTrack].groupIdx = -1;
            fpLabels_[fpTrack].whichAB = -1;
          }
        }
        DSP_DIAG(MEMORY,
                 "BUF_FP_STREAM plan=%p step=%d slot=%d path=fused writer=%p api=%p dsp=%p lc=%p",
                 (void*)this, fpStep, lastOutputSlotIdx, (void*)*fpStreamPtr, stream,
                 execCtx != nullptr ? execCtx->dspStream : nullptr,
                 execCtx != nullptr ? execCtx->lcDefaultStream : nullptr);
        tl_compositeCaptureStream = fpSavedCompositeCaptureStream;
      }
    }
#endif

    return Status::OK;
  }

  // ── Fast path: frozen context ────────────────────────────────────────────
  // Dynamic-shape slots always bypass this path: their output shape changes each
  // step, so the frozen context's cached shapes/buffers are stale. They must
  // fall through to the normal execution path below.
  if (!frozenSlotBySlot && slot.frozenContextReady() && !slot.flags.isDynamicShape
      && syncOverrideDepth_ <= 0) {

    // ── View-capable fast path (reshape/expand_dims/squeeze/strided_slice) ──
    if (slot.isViewCapableOp() && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
      int si = slot.wiring.outputSlotIndices[0];
      if (si >= 0 && si < totalOutputSlots_) {
        // Resolve input0 from slot source indices.
        // Use resolveViewInput to prefer staging buffers for external inputs.
        // Views must alias the staging buffer (stable address) so that cuBLAS
        // kernels baked into the CUDA graph see consistent addresses on replay.
        int srcIdx = slot.wiring.inputSourceIndices[0];
        NDArray* input0 = resolveViewInput(srcIdx, externalArrays, numExt);

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
            ssInputs[ii] = resolveViewInput(iiSrc, externalArrays, numExt);
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
              deferredSlotDeletes_.push_back(newView);  // Defer: inline delete during slot execution causes heap corruption
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
            markViewProducer(stepIdx, "view-install");
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
    if (slot.isViewCapableOp() && slot.wiring.numInputs >= 1) {
      int srcIdx0 = slot.wiring.inputSourceIndices[0];
      NDArray* inp0 = resolveViewInput(srcIdx0, externalArrays, numExt);
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
          // Device-buffer identity of the slot a consumer actually reads. Pairs with
          // VIEW-INSTALL-DEV to confirm a view-producer's device buffer reaches its consumer.
          if (outputSlots_[srcIdx]->dataBuffer() != nullptr) {
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                "INPUT-BIND-DEV: op=%s in=%d srcSlot=%d srcDev=%p len=%lld exec=%d",
                slot.ident.opName.c_str(), i, srcIdx,
                (void*)outputSlots_[srcIdx]->dataBuffer()->special(),
                (long long)outputSlots_[srcIdx]->lengthOf(), (int)executeCount_);
          }
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

    // ── Slot-identity reconciliation (task #54) ─────────────────────────────
    // A plan checked out from the native cache can carry outputSlots_ wrappers
    // (re-installed by the step-3 cached-reuse path) that differ from this
    // frozen context's output arrays (bound at freeze time). The kernel then
    // writes the ctx array while readback materializes from the wrapper's
    // stale device pointer — the host reads content this exec never wrote
    // (bit-identical, batch-only wrong outputs; the two buffers only diverge
    // after a cross-test plan reuse). Rebind the ctx output to the CURRENT
    // wrapper so kernel-write and readback share one identity.
    {
      auto& fpOutRebind = ctx.fastpath_out();
      for (int oi = 0; oi < slot.wiring.numOutputs && oi < (int)fpOutRebind.size(); oi++) {
        int osi = slot.wiring.outputSlotIndices[oi];
        if (osi < 0 || osi >= totalOutputSlots_) continue;
        NDArray* wrapper = outputSlots_[osi];
        NDArray* ctxArr = fpOutRebind[oi];
        if (wrapper == nullptr || ctxArr == nullptr) continue;
        if (wrapper->dataBuffer() != ctxArr->dataBuffer()) {
          DSP_DIAG(FALLBACK,
                   "FROZEN_CTX_OUTPUT_REBIND: slot=%d op=%s outIdx=%d ctxDb=%p wrapperDb=%p "
                   "— frozen ctx output diverged from installed slot wrapper (cached-plan "
                   "reuse); rebinding ctx to the wrapper so write and readback share one buffer",
                   stepIdx, slot.ident.opName.c_str(), oi,
                   (void*)ctxArr->dataBuffer(), (void*)wrapper->dataBuffer());
          ctx.setOutputArray(oi, wrapper);
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
    // Sync policy: computed from ModeContract + lifecycle + scoped overrides.
    // During warmup (count<2), slot-by-slot, or when contract requires it, sync runs.
    bool syncNeeded = needsSync();
    if (DSP_DIAG_ENABLED(STREAM_SYNC)) {
      DSP_DIAG_SLOT(STREAM_SYNC, stepIdx,
                    "frozenCtx sync=%s reason=%s exec=%d mode=%s frozen=%d overrideDepth=%d",
                    syncNeeded ? "YES" : "NO", syncReason(),
                    executeCount_, ModeContract::modeName(static_cast<int>(graphExecutionMode_)),
                    (int)planLifecycle_.isShapesFrozen(), syncOverrideDepth_);
    }
    // SlotSyncGuard: prepareSpecialUse now, registerSpecialUse on scope exit.
    SlotSyncGuard frozenSyncGuard(syncNeeded, ctx.fastpath_out(), ctx.fastpath_in(),
                                   stepIdx, slot.ident.opName.c_str(), executeCount_);
    if (!syncNeeded) {
      // Assertion 4: Actuality flag consistency when sync is skipped.
      // Only meaningful when the backend tracks a distinct device-memory domain.
      // Host-only isSpecialActual() is false, so exclude it from this assertion.
      if (dspHasDeviceMemory() && executeCount_ < 5) {
        auto& fpin_assert = ctx.fastpath_in();
        for (size_t ii = 0; ii < fpin_assert.size(); ii++) {
          NDArray* a = fpin_assert[ii];
          if (a == nullptr) continue;
          DataBuffer* db = a->dataBuffer();
          if (db == nullptr || db->isClosed() || db->getLenInBytes() == 0) continue;
          bool sAct = db->isSpecialActual();
          if (!sAct) {
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
        status = platformExecuteSlot(slot, ctx);
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

#ifdef SD_CUDA
    // BUF_FP_RING slot-output fingerprints (frozen-context path) — async on
    // the op's own stream. During an outer composite region, native gap outputs
    // use the post-capture step; hiding only the outer marker lets the stream's
    // actual capture state remain the authoritative guard.
    if (fpRingEnabled_ && status == Status::OK && !shapeOnlyMode_) {
      auto* fpLc = ctx.launchContext() != nullptr ? ctx.launchContext()
                                                  : LaunchContext::defaultContext();
      auto* fpStreamPtr = fpLc != nullptr ? fpLc->getCudaStream() : nullptr;
      if (fpStreamPtr != nullptr) {
        void* fpSavedCompositeCaptureStream = tl_compositeCaptureStream;
        int fpStep = execCtx != nullptr ? execCtx->fpStep : executeCount_;
        if (fpSavedCompositeCaptureStream != nullptr) {
          tl_compositeCaptureStream = nullptr;
          fpStep += BUF_FP_POST_CAPTURE_STEP_OFFSET;
        }
        auto& fpCtxOut = ctx.fastpath_out();
        for (int i = 0; i < slot.wiring.numOutputs &&
                        i < static_cast<int>(fpCtxOut.size()); i++) {
          int fpSi = slot.wiring.outputSlotIndices[i];
          NDArray* fpOutArr = fpCtxOut[i];
          int fpTrack = numExternalInputs_ + fpSi;
          bool fpRequested = numRequestedOutputs_ > 0 && requestedOutputSlotIndices_ != nullptr &&
                             requestedOutputSlotIndices_[0] == fpSi;
          if (fpRequested && fpOutArr != nullptr && fpOutArr->specialBuffer() != nullptr &&
              !fpOutArr->isEmpty()) {
            recordBufFingerprintPublic(*fpStreamPtr, fpStep, BUF_FP_REQUESTED_OUTPUT_TRACK,
                                       fpOutArr->specialBuffer(), dspSafeByteCount(fpOutArr));
            if (fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag[0] == '\0') {
              snprintf(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag,
                       sizeof(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag), "req.writer");
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].extIdx = -1;
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].groupIdx = -1;
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].whichAB = -1;
            }
            DSP_DIAG(MEMORY, "BUF_FP_STREAM plan=%p step=%d slot=%d path=frozen writer=%p dsp=%p lc=%p",
                     (void*)this, fpStep, fpSi, (void*)*fpStreamPtr,
                     execCtx != nullptr ? execCtx->dspStream : nullptr,
                     execCtx != nullptr ? execCtx->lcDefaultStream : nullptr);
          }
          if (fpOutArr != nullptr && fpSi >= 0 && fpTrack < BUF_FP_TRACE_TRACK &&
              fpOutArr->specialBuffer() != nullptr && !fpOutArr->isEmpty()) {
            recordBufFingerprintPublic(*fpStreamPtr, fpStep, fpTrack,
                                       fpOutArr->specialBuffer(), dspSafeByteCount(fpOutArr));
            if (fpLabels_[fpTrack].tag[0] == '\0') {
              snprintf(fpLabels_[fpTrack].tag, sizeof(fpLabels_[fpTrack].tag), "s%d", fpSi);
              fpLabels_[fpTrack].extIdx = -1;
              fpLabels_[fpTrack].groupIdx = -1;
              fpLabels_[fpTrack].whichAB = -1;
            }
          }
        }
        tl_compositeCaptureStream = fpSavedCompositeCaptureStream;
      }
    }
#endif

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

    // registerSpecialUse + anomaly detection handled by frozenSyncGuard destructor

    auto& ctxOuts = ctx.fastpath_out();

    // Per-slot output fingerprint — active when debug+verbose at exec >= 2
    if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose()) {
      for (int i = 0; i < slot.wiring.numOutputs && i < (int)ctxOuts.size(); i++) {
        auto* array = ctxOuts[i];
        if (array != nullptr && !array->isEmpty() && array->lengthOf() > 0) {
          DSP_DIAG(VERIFY,
                   "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld asyncValues=true",
                   executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                   ShapeUtils::shapeAsString(array).c_str(),
                   DataTypeUtils::asString(array->dataType()).c_str(),
                   (long long)array->lengthOf());
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
                     executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
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
      DSP_LIFECYCLE_EVENT(executeCount_, stepIdx, "SLOT_INPUT_FROM_SLOT", inputs[i]);
    } else {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) {
        inputs[i] = externalArrays[extIdx];
        DSP_LIFECYCLE_EVENT(executeCount_, stepIdx, "SLOT_INPUT_FROM_EXT", inputs[i]);
      } else {
        inputs[i] = nullptr;
      }
    }

    if (inputs[i] == nullptr) {
      // Phase-aware null input detection — a null input is ALWAYS a bug.
      // Identify the source for diagnostics then throw a hard exception.
      const char* srcOpName = "?";
      int srcStep = -1;
      bool isSelfRef = false;
      bool isForwardRef = false;
      bool isExternal = (srcIdx < 0);
      int extIdx = isExternal ? -(srcIdx + 1) : -1;
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
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
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          if (slot.wiring.outputSlotIndices[o] == srcIdx) {
            isSelfRef = true;
            break;
          }
        }
        isForwardRef = (srcStep > stepIdx);
      }
      // Determine execution phase for the error message
      const char* phaseStr = "UNKNOWN";
      if (planLifecycle_.isSlotBySlot()) {
        phaseStr = "SLOT_BY_SLOT";
      } else if (planLifecycle_.isShapesFrozen()) {
        phaseStr = (executeCount_ == 0) ? "FROZEN_WARMUP" :
                   (executeCount_ < 3)  ? "FROZEN_CAPTURE" : "FROZEN_REPLAY";
      } else {
        phaseStr = "UNFROZEN_WARMUP";
      }
      DSP_THROW(EXECUTE,
               "NativeDynamicShapePlan: NULL input at slot %d (%s) input[%d] "
               "during %s phase (execCount=%d frozen=%d). "
               "srcIdx=%d %s extIdx=%d producerStep=%d producerOp=%s "
               "selfRef=%d forwardRef=%d effectiveExt=%p stagingBufs=%p "
               "planAddr=%p planFP=0x%016llx. "
               "A null input means the producer slot or external input was never populated. "
               "Fix the source — never work around by guarding the consumer.",
               stepIdx, slot.ident.opName.c_str(), i,
               phaseStr, executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0,
               slot.wiring.inputSourceIndices[i],
               isExternal ? "EXTERNAL" : "SLOT_OUTPUT",
               extIdx, srcStep, srcOpName,
               isSelfRef ? 1 : 0, isForwardRef ? 1 : 0,
               (void*)effectiveExternals_,
               (void*)placeholderStagingBuffers_,
               (void*)this, (unsigned long long)identityFingerprint_);
      return Status::BAD_INPUT;  // unreachable, but keeps return-type analysis clean
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
      const char* closedPhaseStr = "UNKNOWN";
      if (planLifecycle_.isSlotBySlot()) {
        closedPhaseStr = "SLOT_BY_SLOT";
      } else if (planLifecycle_.isShapesFrozen()) {
        closedPhaseStr = (executeCount_ == 0) ? "FROZEN_WARMUP" :
                         (executeCount_ < 3)  ? "FROZEN_CAPTURE" : "FROZEN_REPLAY";
      } else {
        closedPhaseStr = "UNFROZEN_WARMUP";
      }
      DSP_THROW(EXECUTE,
               "NativeDynamicShapePlan: CLOSED DataBuffer at slot %d (%s) input[%d] "
               "during %s phase (execCount=%d frozen=%d). "
               "srcIdx=%d specialBuf=%p isConst=%d. "
               "The buffer was freed while the slot still references it — "
               "this is a lifecycle bug in the producer or caller.",
               stepIdx, slot.ident.opName.c_str(), i,
               closedPhaseStr, executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0,
               slot.wiring.inputSourceIndices[i],
               db->special(), db->isConstant ? 1 : 0);
      return Status::BAD_INPUT;  // unreachable, but keeps return-type analysis clean
    }
    // Validate platform buffer (GPU special buffer or CPU primary buffer)
    if (!platformValidateSlotInputBuffer(stepIdx, slot, i, inputs[i])) {
      const char* bufPhaseStr = "UNKNOWN";
      if (planLifecycle_.isSlotBySlot()) {
        bufPhaseStr = "SLOT_BY_SLOT";
      } else if (planLifecycle_.isShapesFrozen()) {
        bufPhaseStr = (executeCount_ == 0) ? "FROZEN_WARMUP" :
                      (executeCount_ < 3)  ? "FROZEN_CAPTURE" : "FROZEN_REPLAY";
      } else {
        bufPhaseStr = "UNFROZEN_WARMUP";
      }
      DSP_THROW(EXECUTE,
               "NativeDynamicShapePlan: invalid platform buffer at slot %d (%s) input[%d] "
               "during %s phase (execCount=%d frozen=%d). "
               "srcIdx=%d ptr=%p db=%p len=%lld. "
               "The slot's input has a null device/host buffer — "
               "this is a lifecycle bug in the producer or buffer allocation.",
               stepIdx, slot.ident.opName.c_str(), i,
               bufPhaseStr, executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0,
               slot.wiring.inputSourceIndices[i],
               (void*)inputs[i], (void*)inputs[i]->dataBuffer(),
               (long long)inputs[i]->lengthOf());
      return Status::BAD_INPUT;  // unreachable
    }
  }

  // ── Capture probe: after Step 1 (gather inputs) ──────────────────────────
  {
    bool _capInvalid = false;
    void* _capStream = dspGetExecutionStream();
    DSP_CAPTURE_PROBE(_capStream,
                      stepIdx, "AFTER_GATHER_INPUTS",
                      slot.ident.opName.c_str(), _capInvalid);
  }

  // Value-dependent shape functions run on the host, while CUDA replay uses
  // stable staging wrappers whose primary buffers may intentionally be stale
  // after an asynchronous D2D refresh. For small placeholder controls, use the
  // current call's host-authoritative wrapper for shape-key calculation and
  // calculateOutputShape only. Step 4 restores the stable staged inputs before
  // executing the op, so captured device pointers remain unchanged.
  NDArray** shapeValueInputs = inputs.data();
  static thread_local std::vector<NDArray*> hostShapeInputs;
  if (slot.flags.outputShapeDependsOnInputValues && lastExternalInputs_ != nullptr) {
    hostShapeInputs.assign(inputs.begin(), inputs.end());
    bool replacedShapeInput = false;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      const int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx >= 0) continue;

      const int extIdx = -(srcIdx + 1);
      if (extIdx < 0 || extIdx >= lastNumExternalInputs_ ||
          !isExternalInputPlaceholder(extIdx)) {
        continue;
      }

      NDArray* currentInput = lastExternalInputs_[extIdx];
      NDArray* stagedInput = inputs[i];
      if (currentInput == nullptr || stagedInput == nullptr ||
          currentInput == stagedInput ||
          !isSmallIntegralControlArray(currentInput) ||
          currentInput->dataType() != stagedInput->dataType() ||
          currentInput->lengthOf() != stagedInput->lengthOf()) {
        continue;
      }

      auto* currentBuffer = currentInput->dataBuffer();
      if (currentBuffer == nullptr || currentBuffer->isClosed() ||
          currentBuffer->primary() == nullptr || !currentBuffer->isPrimaryActual()) {
        continue;
      }

      hostShapeInputs[i] = currentInput;
      replacedShapeInput = true;
      DSP_DIAG_SLOT(SHAPE, stepIdx,
          "VALUE_SHAPE_ORIGINAL_EXT: slot=%d op=%s input=%d ext=%d "
          "staged=%p current=%p currentHost=%p dtype=%d len=%lld exec=%d",
          stepIdx, slot.ident.opName.c_str(), i, extIdx,
          (void*)stagedInput, (void*)currentInput, currentBuffer->primary(),
          (int)currentInput->dataType(), (long long)currentInput->lengthOf(),
          executeCount_);
    }
    if (replacedShapeInput) {
      shapeValueInputs = hostShapeInputs.data();
    }
  }

  // A value-dependent shape op can follow a captured segment whose integral
  // control output was produced asynchronously on the plan-owned stream. The
  // subsequent host D2H in computeShapeKey must not race that producer. Actuality
  // counters tell us whether a plan-internal control has a newer device value;
  // synchronize the producer stream once at that precise graph-to-host boundary.
  // External host-authoritative controls and graph-capture execution do not need
  // (and during capture must not perform) this synchronization.
  if (slot.flags.outputShapeDependsOnInputValues &&
      !dspStreamIsCapturing(dspGetExecutionStream())) {
    bool hasPendingInternalControl = false;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (slot.wiring.inputSourceIndices[i] < 0 ||
          !isSmallIntegralControlArray(shapeValueInputs[i])) {
        continue;
      }
      auto* controlBuffer = shapeValueInputs[i]->dataBuffer();
      if (controlBuffer != nullptr && !controlBuffer->isClosed() &&
          controlBuffer->isSpecialActual() && !controlBuffer->isPrimaryActual()) {
        hasPendingInternalControl = true;
        break;
      }
    }
    if (hasPendingInternalControl) {
      void* executionStream = dspGetExecutionStream();
      if (executionStream != nullptr && !dspSynchronizeStream(executionStream)) {
        DSP_DIAG_SLOT(STREAM_SYNC, stepIdx,
            "VALUE_SHAPE_PRODUCER_SYNC_FAILED: slot=%d op=%s stream=%p exec=%d",
            stepIdx, slot.ident.opName.c_str(), executionStream, executeCount_);
        return Status::KERNEL_FAILURE;
      }
      DSP_DIAG_SLOT(STREAM_SYNC, stepIdx,
          "VALUE_SHAPE_PRODUCER_SYNC: slot=%d op=%s stream=%p exec=%d",
          stepIdx, slot.ident.opName.c_str(), executionStream, executeCount_);
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
    shapeKey = computeShapeKey(slot, shapeValueInputs, slot.wiring.numInputs);
    cacheHit = (slot.shapeCache.cachedShapeKey == shapeKey);

    // View-capable ops: check for stale empty cached shapes.
    // During Step 0 (first execution), KV cache inputs are empty, so shape
    // inference produces empty output shapes that get cached. On subsequent
    // executions the inputs become non-empty (concat grows KV), but
    // planLifecycle_.isShapesFrozen() prevents re-inference. Detect this and force a cache miss.
    if (cacheHit && slot.isViewCapableOp() && !slot.shapeCache.cachedOutputShapes.empty()) {
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
    shapeKey = computeShapeKey(slot, shapeValueInputs, slot.wiring.numInputs);
    cacheHit = slot.shapeCacheValid() && (slot.shapeCache.cachedShapeKey == shapeKey);

    // View-capable ops: even with matching shape key, check for stale empty
    // cached shapes. computeShapeKey does NOT include the ARRAY_EMPTY flag,
    // so an input that transitions from empty (ARRAY_EMPTY) to non-empty with
    // the same dimensions produces an identical key but needs re-inference.
    if (cacheHit && slot.isViewCapableOp() && !slot.shapeCache.cachedOutputShapes.empty()) {
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
    // Validate cached output shapes: UNKNOWN/INHERIT dtype means the cache
    // was poisoned (view-chain use-after-free with zeroed DataBuffer).
    // Force a cache miss so shape re-inference runs and fixes the dtype.
    for (int i = 0; i < static_cast<int>(outputShapes.size()); i++) {
      if (outputShapes[i] != nullptr) {
        DataType dt = ArrayOptions::dataType(outputShapes[i]);
        if (dt == DataType::UNKNOWN || dt == DataType::INHERIT) {
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "CACHE_HIT_DTYPE_POISON: slot %d (%s) cached output[%d] has UNKNOWN/INHERIT dtype "
              "— forcing cache miss for re-inference",
              stepIdx, slot.ident.opName.c_str(), i);
          cacheHit = false;
          break;
        }
      }
    }
    if (cacheHit) {
      // Cache hit is clean — keep outputShapes as assigned
    } else {
      outputShapes.clear();
    }
  }
  if (!cacheHit) {
    // A value-dependent op may reveal its recurring output-shape behavior only
    // on the first SHAPES_FROZEN execution (the first value change after the
    // slot-by-slot pass).  Step 3 below owns wrapper replacement and marks the
    // slot permanently dynamic.  Permit exactly that pre-replay observation;
    // replay-time shape drift remains a hard lifecycle error.
    const bool allowWarmupValueShapeClassification =
        planLifecycle_.isShapesFrozen() && executeCount_ == 1 &&
        slot.shapeCacheValid() && slot.flags.outputShapeDependsOnInputValues &&
        !slot.flags.isDynamicShape;
    bool allowRestabilization =
        allowWarmupValueShapeClassification ||
        (!planLifecycle_.isSlotBySlot() && executeCount_ > 0 && slot.shapeCacheValid() &&
         allowFrozenShapeRestabilization(stepIdx, slot, getPlanPhase(),
                                         slots_, numSlots_,
                                         outputSlots_, totalOutputSlots_,
                                         externalArrays, numExt));

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
      ctx.setInputArray(i, shapeValueInputs[i]);
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
      if (shapeValueInputs[i] != nullptr) {
        inputShapes.push_back(shapeValueInputs[i]->shapeInfo());
      }
    }

    // Validate: no input shape may carry UNKNOWN or INHERIT dtype into calculateOutputShape.
    // UNKNOWN (255) is numerically larger than all real types, so ops that infer
    // output dtype via max(dtypeX, dtypeY) (e.g. matmul) silently inherit the
    // poison.  INHERIT (0) is the zero-init value of freed memory and indicates a
    // use-after-free of a DataBuffer whose memory was reclaimed by Java GC.
    // Catching it here gives a clear diagnostic instead of a cryptic
    // "Unknown data type requested DataTypeUtils" from elementSize().
    for (int i = 0; i < static_cast<int>(inputShapes.size()); i++) {
      const LongType* inShape = inputShapes.at(i);
      if (inShape != nullptr &&
          (ArrayOptions::dataType(inShape) == DataType::UNKNOWN ||
           ArrayOptions::dataType(inShape) == DataType::INHERIT)) {
        int srcIdx = (i < slot.wiring.numInputs) ? slot.wiring.inputSourceIndices[i] : -9999;
        int producerStep = -1;
        std::string producerName = "?";
        if (srcIdx >= 0) {
          producerStep = findProducingStepForOutputSlot(slots_, numSlots_, srcIdx);
          if (producerStep >= 0) producerName = slots_[producerStep].ident.opName;
        } else if (srcIdx < 0 && srcIdx != -9999) {
          producerName = "ext[" + std::to_string(-(srcIdx + 1)) + "]";
        }
        std::string msg = "UNKNOWN dtype on input[" + std::to_string(i)
            + "] of slot " + std::to_string(stepIdx) + " (" + slot.ident.opName
            + "). Producer: step " + std::to_string(producerStep)
            + " (" + producerName + "), srcIdx=" + std::to_string(srcIdx)
            + ". Shape: " + ShapeUtils::shapeAsString(inShape);
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "UNKNOWN_DTYPE_INPUT: %s", msg.c_str());
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
            static_cast<int>(Status::KERNEL_FAILURE));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(msg.c_str());
        return Status::KERNEL_FAILURE;
      }
    }

    ShapeList* shapeList = nullptr;
    try {
      if (!slot.shapeCache.staticOutputShapeInfos.empty()) {
        shapeList = new ShapeList();
        for (auto& shapeInfo : slot.shapeCache.staticOutputShapeInfos) {
          shapeList->push_back(shapeInfo.data());
        }
      } else {
        shapeList = slot.ident.op->calculateOutputShape(&inputShapes, ctx);
      }

      // ── Diagnostic: log shape inference results for reduce ops ──────────
      // This catches concurrency bugs where keepDims=true produces wrong rank.
      if (shapeList != nullptr && shapeList->size() > 0 && DSP_DIAG_ENABLED(SHAPE)) {
        std::string outShapeStr;
        for (int si = 0; si < static_cast<int>(shapeList->size()); si++) {
          if (si > 0) outShapeStr += ", ";
          if (shapeList->at(si) != nullptr) {
            outShapeStr += ShapeUtils::shapeAsString(shapeList->at(si));
            outShapeStr += "{order=";
            outShapeStr.push_back(shape::order(shapeList->at(si)));
            outShapeStr += ",strides=[";
            const auto rank = shape::rank(shapeList->at(si));
            const auto* strides = shape::stride(shapeList->at(si));
            for (int d = 0; d < rank; d++) {
              if (d > 0) outShapeStr += ",";
              outShapeStr += std::to_string(strides[d]);
            }
            outShapeStr += "]}";
          } else {
            outShapeStr += "null";
          }
        }
        std::string inShapeStr;
        for (int si = 0; si < static_cast<int>(inputShapes.size()); si++) {
          if (si > 0) inShapeStr += ", ";
          inShapeStr += inputShapes.at(si) ? ShapeUtils::shapeAsString(inputShapes.at(si)) : "null";
        }
        DSP_DIAG_SLOT(SHAPE, stepIdx,
            "SHAPE_INFER_RESULT: slot %d (%s) bArgs=%d iArgs=%d width=%lu "
            "inputs=[%s] -> outputs=[%s] tid=%lu exec=%d",
            stepIdx, slot.ident.opName.c_str(),
            slot.args.numBArgs, slot.args.numIArgs,
            (unsigned long)ctx.width(),
            inShapeStr.c_str(), outShapeStr.c_str(),
            (unsigned long)std::hash<std::thread::id>{}(std::this_thread::get_id()),
            executeCount_);
      }
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
      // Compose via std::string: the previous fixed snprintf buffer truncated
      // exactly the src=/vals= fields needed to diagnose value-dependent
      // shape failures (observed cut mid-token at "src").
      std::string errMsg = "slot " + std::to_string(stepIdx) + " (" +
          slot.ident.opName + ") shape inference failed: " + e.what() +
          " | inputs=[" + inputShapeStr + "] iArgs=" +
          std::to_string(slot.args.numIArgs) +
          " frozenConst=" + std::to_string(slot.frozenConstantSlot() ? 1 : 0) +
          " state=" + slot.slotPhase.displayName() +
          " execCount=" + std::to_string(executeCount_) +
          " src=[" + srcIdxSummary + "]";
      // Append actual iArgs values for reshape/shape debugging
      if (slot.args.numIArgs > 0) {
        errMsg += " iArgValues=[";
        for (int ia = 0; ia < slot.args.numIArgs; ia++) {
          if (ia > 0) errMsg += ",";
          errMsg += std::to_string(slot.args.iArgs[ia]);
        }
        errMsg += "]";
      }
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
      // Mirror the fields into the diagnostics stream as separate lines so
      // they survive any downstream error-message plumbing limits.
      DSP_DIAG_SLOT(SHAPE, stepIdx, "SHAPE_FAIL_SRC: slot=%d src=[%s]",
                    stepIdx, srcIdxSummary.c_str());
      DSP_DIAG_SLOT(SHAPE, stepIdx, "SHAPE_FAIL_VALS: slot=%d %s",
                    stepIdx, smallInputValueStr.c_str());
      DSP_DIAG_SLOT(SHAPE, stepIdx, "SHAPE_FAIL_UPSTREAM: slot=%d %s",
                    stepIdx, upstreamTraceStr.c_str());
      // Post-mortem deep probe (cold error path only): for each small control
      // input dump the parent buffer's full host contents before and after a
      // forced D2H through the normal lifecycle sync, plus the wrapper's view
      // offset. Separates device-side staleness from host staleness from
      // wrapper-offset corruption without any hot-path cost.
      if (DSP_DIAG_ENABLED(SHAPE)) {
        for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
          NDArray* in = inputs[ii];
          if (in == nullptr || in->dataBuffer() == nullptr) continue;
          const auto dt = in->dataType();
          if (dt != INT32 && dt != INT64 && dt != BOOL) continue;
          auto* db = in->dataBuffer();
          const auto dbElems = db->getNumElements();
          if (dbElems <= 0 || dbElems > 64) continue;
          const int dumpN = static_cast<int>(dbElems < 4 ? dbElems : 4);
          float beforeVals[16];
          float afterVals[16];
          const int nb = dspReadHostValues(db, dt, dbElems, beforeVals, dumpN);
          db->syncToPrimary(LaunchContext::defaultContext(), /*forceSync=*/true);
          const int na = dspReadHostValues(db, dt, dbElems, afterVals, dumpN);
          // The exact pointer + value NDArray::e() consumes (buffer() applies
          // the wrapper offset), vs the DataBuffer-level base pointer, to
          // catch a wrapper whose live pointer diverges from db->primary().
          const void* eReadPtr = in->buffer();
          long long eReadVal = 0;
          if (eReadPtr != nullptr && dt == INT64) {
            eReadVal = static_cast<long long>(
                reinterpret_cast<const sd::LongType*>(eReadPtr)[0]);
          } else if (eReadPtr != nullptr && dt == INT32) {
            eReadVal = static_cast<long long>(
                reinterpret_cast<const int*>(eReadPtr)[0]);
          }
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "SHAPE_FAIL_DEEP: slot=%d input=%d wrapperOffset=%lld dbElems=%lld "
              "hostBefore=%s hostAfterForceD2H=%s pAct=%d sAct=%d "
              "eReadPtr=%p eReadVal=%lld dbPrimary=%p special=%p",
              stepIdx, ii, static_cast<long long>(in->offset()),
              static_cast<long long>(dbElems),
              dspFormatValues(beforeVals, nb).c_str(),
              dspFormatValues(afterVals, na).c_str(),
              db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
              eReadPtr, eReadVal, db->primary(), db->special());
        }
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
        auto& shapeHelper = ConstantShapeHelper::getInstance();
        auto* rawShape = const_cast<LongType*>(shapeList->at(i));
        // Preserve ARRAY_IS_VIEW for trait-declared view outputs so later
        // logical materialization honors this shape's strides.
        auto cached = i == 0 && slot.isViewCapableOp()
                          ? shapeHelper.bufferForShapeInfoWithView(rawShape)->primary()
                          : shapeHelper.createFromExisting(rawShape);
        outputShapes[i] = cached;
      } catch (const std::exception& e) {
        DSP_DIAG_SLOT(SHAPE, stepIdx, "shape cache interning EXCEPTION at slot %d (%s) output[%d]: %s",
                  stepIdx, slot.ident.opName.c_str(), i, e.what());
        delete shapeList;
        return Status::KERNEL_FAILURE;
      }
    }

    // Validate output shapes: UNKNOWN dtype in any output is a bug.
    // The pre-calculateOutputShape input validation above should have caught
    // UNKNOWN inputs before they poison shape inference.  If we still see
    // UNKNOWN here, a new code path is leaking it — fail hard so we find it.
    for (int i = 0; i < static_cast<int>(outputShapes.size()); i++) {
      if (ArrayOptions::dataType(outputShapes[i]) == DataType::UNKNOWN) {
        std::string msg = "calculateOutputShape produced UNKNOWN dtype for output["
            + std::to_string(i) + "] of slot " + std::to_string(stepIdx)
            + " (" + slot.ident.opName + "). Input dtypes:";
        for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
          msg += " i" + std::to_string(ii) + "=";
          msg += inputs[ii] ? DataTypeUtils::asString(inputs[ii]->dataType()) : "null";
        }
        DSP_DIAG_SLOT(SHAPE, stepIdx, "UNKNOWN_DTYPE_OUTPUT: %s", msg.c_str());
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
            static_cast<int>(Status::KERNEL_FAILURE));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(msg.c_str());
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
          if (allowWarmupValueShapeClassification) {
            DSP_DIAG(SHAPE,
                     "VALUE_DEP_WARMUP_SHAPE_DRIFT: slot %d (%s) output shape changed "
                     "on first SHAPES_FROZEN execution (execCount=%d phase=%s). "
                     "Updating cached shapes so Step 3 can reclassify the slot as dynamic.",
                     stepIdx, slot.ident.opName.c_str(), executeCount_,
                     planLifecycle_.displayName());
          } else {
            DSP_DIAG(SHAPE,
                     "FROZEN_RESTABILIZE: slot %d (%s) output shape changed during SHAPES_FROZEN "
                     "but the controlling value-shape chain is plan-internal and "
                     "planPhase=%s (< SEALED). Updating cached shapes.",
                     stepIdx, slot.ident.opName.c_str(), planLifecycle_.displayName());
          }
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
        sanitizeCachedOutputShapes(slot, outputSlots_, totalOutputSlots_,
                                   externalArrays, numExt);
        if (!slot.slotPhase.shapeCacheValid) {
          slot.slotPhase.markShapeCached();  // PRIMARY
        }
      } else {
        slot.shapeCache.cachedShapeKey = shapeKey;
        outputShapes = slot.shapeCache.cachedOutputShapes;
      }
    } else {
      slot.shapeCache.cachedShapeKey = shapeKey;
      slot.shapeCache.cachedOutputShapes = outputShapes;
      sanitizeCachedOutputShapes(slot, outputSlots_, totalOutputSlots_,
                                 externalArrays, numExt);
      if (!slot.slotPhase.shapeCacheValid) {
        slot.slotPhase.markShapeCached();  // PRIMARY
      }
    }

    delete shapeList;
  }

  // ── Capture probe: after Step 2 (shape inference) ────────────────────────
  {
    bool _capInvalid = false;
    void* _capStream = dspGetExecutionStream();
    DSP_CAPTURE_PROBE(_capStream,
                      stepIdx, "AFTER_SHAPE_INFERENCE",
                      slot.ident.opName.c_str(), _capInvalid);
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
  // Some ops (notably inference-only flash attention) keep their logical output
  // positions for ABI compatibility while explicitly opting out of materialization.
  // Optional masks/cache placeholders are runtime values, so re-check them after
  // input binding instead of treating every extra input position as live.
  auto hasRuntimeData = [](NDArray* array) {
    return array != nullptr && !array->isEmpty() && array->lengthOf() > 0;
  };
  auto hasRuntimeValue = [&hasRuntimeData](NDArray* array) {
    return hasRuntimeData(array) && array->rankOf() > 0;
  };
  auto inputAt = [](int index) -> NDArray* {
    return index >= 0 && index < static_cast<int>(inputs.size()) ? inputs[index] : nullptr;
  };
  bool omitAttentionAuxOutputs = slot.ident.opName == "dot_product_attention_v2";
  if (omitAttentionAuxOutputs) {
    // qMask/vMask are inputs 3/4. A non-empty mask forces the fallback path,
    // which requires scores/logits for its softmax and dropout bookkeeping.
    if (hasRuntimeValue(inputAt(3)) || hasRuntimeValue(inputAt(4))) {
      omitAttentionAuxOutputs = false;
    }
    // Inputs 5/6/7 are key cache, value cache, and cache position. The fused
    // cache-write/decode path also requires the auxiliary outputs to remain live.
    if (hasRuntimeValue(inputAt(5)) && hasRuntimeValue(inputAt(6)) &&
        hasRuntimeData(inputAt(7))) {
      omitAttentionAuxOutputs = false;
    }
  }
  auto isOptionalOutput = [&slot, omitAttentionAuxOutputs](int outputIndex) {
    return omitAttentionAuxOutputs && outputIndex >= 0 && outputIndex < slot.wiring.numOutputs &&
           slot.wiring.optionalOutputMask != nullptr &&
           slot.wiring.optionalOutputMask[outputIndex] != 0;
  };
  auto makeOptionalOutputPlaceholder = [&](int outputIndex) -> NDArray* {
    const LongType* expectedShape = outputShapes[outputIndex];
    int outputSlot = (outputIndex < numWiredOutputs)
                         ? slot.wiring.outputSlotIndices[outputIndex]
                         : -1;
    if (outputSlot >= 0 && outputSlot < totalOutputSlots_) {
      NDArray* existing = outputSlots_[outputSlot];
      if (existing != nullptr && existing->hasValidShapeInfo() &&
          existing->isEmpty() &&
          shape::equalsSoft(existing->shapeInfo(), expectedShape) &&
          ArrayOptions::dataType(existing->shapeInfo()) ==
              ArrayOptions::dataType(expectedShape)) {
        return existing;
      }
    }
    // Keep the logical output shape and dtype visible to DeclarableOp while
    // marking the array empty. This lets fastpath validation and tracing run
    // without allocating the quadratic scores/logits buffers.
    LongType* emptyShape = ShapeBuilders::copyShapeInfo(expectedShape, true, nullptr);
    ArrayOptions::setPropertyBit(emptyShape, ARRAY_EMPTY);
    NDArray* placeholder =
        new NDArray(nullptr, emptyShape, LaunchContext::defaultContext(), false, 0);
    delete[] emptyShape;
    if (outputSlot >= 0 && outputSlot < totalOutputSlots_) {
      writeOutputSlot(outputSlot, placeholder, "optional-output-placeholder");
      outputSlots_[outputSlot] = placeholder;
      DSP_DIAG_SLOT_WRITE(outputSlot, slot.ident.opName.c_str(), 0, stream,
                          "optional-output-placeholder");
    }
    return placeholder;
  };

  // ── In-place fusion: reuse input buffer as output ──
  // Guard on the input *index* (always non-negative for valid in-place ops),
  // NOT inPlaceSourceSlot() which returns negative for external-sourced inputs.
  // External inputs are valid in-place sources — they're in the inputs[] array.
  if (slot.isInPlaceFused() && slot.inPlaceFusedInputIdx() >= 0 &&
      slot.inPlaceFusedInputIdx() < slot.wiring.numInputs && numActualOutputs >= 1) {
    NDArray* inPlaceBuffer = inputs[slot.inPlaceFusedInputIdx()];
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
          if (isOptionalOutput(i)) {
            outputs[i] = makeOptionalOutputPlaceholder(i);
            continue;
          }
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
              deferredSlotDeletes_.push_back(cached);
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
  if (!outputsReady && slot.isViewCapableOp() && slot.wiring.numInputs >= 1 && numActualOutputs >= 1) {
    NDArray* input0 = inputs[0];
    if (input0 != nullptr) {
      // VIEW-INPUT0-DEV: the device buffer this view op reads (input0) vs the current array at its
      // source slot. slotMatch=0 means the resolved input diverged from the producer's slot install
      // (the GQA K/V dead-attention signature: consumer reads a fresh/zero buffer, not the producer view).
      if (input0->dataBuffer() != nullptr) {
        int __src0 = slot.wiring.inputSourceIndices[0];
        NDArray* __slotArr = (__src0 >= 0 && __src0 < totalOutputSlots_) ? outputSlots_[__src0] : nullptr;
        DSP_DIAG_SLOT(MEMORY, stepIdx,
            "VIEW-INPUT0-DEV: op=%s src0=%d input0Dev=%p slotArrDev=%p slotMatch=%d len=%lld",
            slot.ident.opName.c_str(), __src0,
            (void*)input0->dataBuffer()->special(),
            (void*)(__slotArr != nullptr && __slotArr->dataBuffer() != nullptr ? __slotArr->dataBuffer()->special() : nullptr),
            (input0 == __slotArr) ? 1 : 0, (long long)input0->lengthOf());
      }
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
                const int srcProducerStep =
                    dsp::findProducingStepForOutputSlot(slots_, numSlots_, srcIdx0);
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-INPUT: producerSlot=%d (%s) from outputSlot=%d producerSlot=%d (%s) "
                    "srcDb=%p oldDb=%p newDb=%p exec=%d phase=%s dbMatch=%d",
                    stepIdx, slot.ident.opName.c_str(), srcIdx0, srcProducerStep,
                    srcProducerStep >= 0 ? slots_[srcProducerStep].ident.opName.c_str() : "?",
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
                const int srcProducerStep =
                    dsp::findProducingStepForOutputSlot(slots_, numSlots_, srcIdx0);
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                    "VIEW-BUF-CHANGE: producerSlot=%d (%s) input from outputSlot=%d "
                    "producerSlot=%d (%s) srcDb=%p oldDb=%p newDb=%p exec=%d phase=%s",
                    stepIdx, slot.ident.opName.c_str(), srcIdx0, srcProducerStep,
                    srcProducerStep >= 0 ? slots_[srcProducerStep].ident.opName.c_str() : "?",
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
              // ── Stale-CUDA-graph guard ───────────────────────────────────
              // A SEALED:VIEW slot's underlying DataBuffer changed between
              // replays (e.g. external placeholder 'x' was replaced with a
              // new NDArray). Any capturable segment that was graph-captured
              // with this view's device address baked into its node arguments
              // will silently read freed/garbage GPU memory on replay (→ err700).
              //
              // Bump argTableGeneration on every capturable segment so that
              // needsArgRefresh() returns true. platformTryFrozenFastPath
              // checks needsArgRefresh() before replayMonolithicGraph and
              // returns MAYBE when it is set, causing the caller to fall back
              // to the normal execute() path which invalidates and recaptures
              // the segment with the correct (stable) device addresses.
              //
              // Performance: we only enter this branch when oldDb != newDb
              // (i.e. the DataBuffer pointer actually changed), which is rare
              // in steady-state execution with staging buffers in place.  In
              // normal frozen replay, the staging buffer address is stable so
              // oldDb == newDb and this block never executes.
              for (size_t _bumpSi = 0; _bumpSi < segments_.size(); _bumpSi++) {
                GraphSegment& _bumpSeg = segments_[_bumpSi];
                if (_bumpSeg.def.isCapturable && !isTerminalOutcome(_bumpSeg.exec.outcome)) {
                  if (!_bumpSeg.exec.needsArgRefresh()) {
                    _bumpSeg.exec.bumpArgGeneration();
                    DSP_DIAG_SLOT(MEMORY, stepIdx,
                        "VIEW-BUF-CHANGE-BUMP: seg[%d-%d] argGen bumped (capturable, stale baked addr guard)",
                        _bumpSeg.def.startSlot, _bumpSeg.def.endSlot);
                  }
                }
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
            deferredSlotDeletes_.push_back(view);  // Defer: inline delete during slot execution causes heap corruption
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
            markViewProducer(stepIdx, "view-op-reuse");
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
            markViewProducer(stepIdx, "view-op-install");
            // Device-buffer identity of the installed view, keyed by output slot + its source
            // slot. Pairs with INPUT-BIND-DEV below to trace whether a view-producer's device
            // buffer actually reaches its downstream consumer's input slot (the GQA K/V chain).
            if (view != nullptr && view->dataBuffer() != nullptr) {
              DSP_DIAG_SLOT(MEMORY, stepIdx,
                  "VIEW-INSTALL-DEV: outSlot=%d op=%s srcSlot0=%d viewDev=%p len=%lld exec=%d",
                  slotIdx, slot.ident.opName.c_str(), slot.wiring.inputSourceIndices[0],
                  (void*)view->dataBuffer()->special(), (long long)view->lengthOf(), (int)executeCount_);
            }
          }
        }

        for (int i = 1; i < numActualOutputs; i++) {
          if (isOptionalOutput(i)) {
            outputs[i] = makeOptionalOutputPlaceholder(i);
            continue;
          }
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
              deferredSlotDeletes_.push_back(untrackedOutputCache_[cacheIdx]);  // Defer: inline delete during slot execution causes heap corruption
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
    if (isOptionalOutput(i)) {
      outputs[i] = makeOptionalOutputPlaceholder(i);
      continue;
    }
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
            deferredSlotDeletes_.push_back(cached);
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
            deferredSlotDeletes_.push_back(cached);
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
            // Slot-identity reconciliation (task #54): if this step has a
            // pooled/frozen op context whose output binding differs from the
            // wrapper we just installed, rebind the context to the wrapper.
            // Otherwise the kernel writes the context's array while readback
            // materializes from this wrapper's (possibly stale, cross-test
            // recycled) device pointer — host reads content this exec never
            // wrote. Covers all exec flavors that reuse pooled contexts,
            // including frozenSlotBySlot (where the frozen-op-exec install
            // loop is skipped).
            if (contextPool_ != nullptr && contextPool_[stepIdx] != nullptr) {
              auto& poolCtxOuts = contextPool_[stepIdx]->fastpath_out();
              if (i < (int)poolCtxOuts.size() && poolCtxOuts[i] != nullptr &&
                  poolCtxOuts[i]->dataBuffer() != cached->dataBuffer()) {
                DSP_DIAG(FALLBACK,
                         "CACHED_REUSE_CTX_REBIND: slot=%d op=%s outIdx=%d "
                         "pooled-ctx out db=%p != reused wrapper db=%p — "
                         "rebinding ctx so kernel-write and readback share one buffer",
                         stepIdx, slot.ident.opName.c_str(), i,
                         (void*)poolCtxOuts[i]->dataBuffer(),
                         (void*)cached->dataBuffer());
                contextPool_[stepIdx]->setOutputArray(i, cached);
              }
            }
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
          slot.shapeCache.cachedOutputShapes.clear();
          // Only mark as permanently dynamic if there's a genuinely dynamic
          // upstream (value-dependent ops like range/fill). Shape changes from
          // the prefill→decode transition are one-time events, not recurring.
          if (hasDynamicUpstream && !slot.flags.isDynamicShape) {
            slot.flags.isDynamicShape = true;
            DSP_DIAG_SLOT(SHAPE, slotIdx,
                "DYNAMIC_SHAPE_CLASSIFIED: slot %d (%s) became dynamic after "
                "observed output-shape drift (execCount=%d)",
                slotIdx, slot.ident.opName.c_str(), executeCount_);
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
          if (rank == 0) {
            maxOut = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
          } else if (!slot.isViewCapableOp()) {
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
      if (rank == 0) {
        // Scalar output: use shapeInfo-preserving constructor so the exact
        // cached shape info (dtype, ews, order from calculateOutputShape) is kept.
        // The empty-vector NDArray constructor creates a fresh scalar shape via
        // scalarDescriptor which may not match the cached shapeInfo exactly.
        out = new NDArray(const_cast<LongType*>(shapeInfo), dt, true);
      } else if (!slot.isViewCapableOp()) {
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

  // ── Capture probe: after Step 3 (output allocation) ──────────────────────
  {
    bool _capInvalid = false;
    void* _capStream = dspGetExecutionStream();
    DSP_CAPTURE_PROBE(_capStream,
                      stepIdx, "AFTER_OUTPUT_ALLOC",
                      slot.ident.opName.c_str(), _capInvalid);
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
      bool isViewOp = slot.isViewCapableOp();
      bool primaryInputNonEmpty = (slot.wiring.numInputs > 0 && inputs[0] != nullptr &&
                                   !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0);
      if (isViewOp && primaryInputNonEmpty) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "view-capable slot %d (%s): outputs empty but input[0] non-empty "
                  "(length=%lld) — forcing execution instead of skip",
                  stepIdx, slot.ident.opName.c_str(), inputs[0]->lengthOf());
        // Invalidate shape cache so next execution re-derives shapes
        slot.slotPhase.reset();  // PRIMARY
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

  // All output entries remain non-null: optional attention auxiliaries use
  // shape-preserving empty placeholders so the normal context validation and
  // tracing paths stay intact.
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
  if (slot.isDataDependent() || slot.flags.outputShapeDependsOnInputValues) {
    std::vector<NDArray*> deviceReads;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      NDArray* in = inputs[i];
      if (in == nullptr || in->isEmpty()) continue;
      auto* db = in->dataBuffer();
      if (db == nullptr || db->isClosed()) continue;

      const bool syncAllInputs = slot.isDataDependent();
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

      deviceReads.push_back(in);
    }
    if (!deviceReads.empty()) {
      NDArray::prepareSpecialUse({}, deviceReads);
      NDArray::registerSpecialUse({}, deviceReads);
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
              (int)cacheHit, (int)planLifecycle_.isShapesFrozen(), executeCount_);
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (inputs[i] != nullptr) {
        auto* db = inputs[i]->dataBuffer();
        DSP_DIAG(EXECUTE, "MATMUL INPUT[%d]: slot %d srcIdx=%d addr=%p special=%p len=%lld "
                  "db=%p closed=%d const=%d pAct=%d sAct=%d",
                  i, stepIdx, slot.wiring.inputSourceIndices[i],
                  (void*)inputs[i], db ? db->special() : nullptr,
                  (long long)inputs[i]->lengthOf(),
                  (void*)db, db ? db->isClosed() : -1,
                  db ? (int)db->isConstant : -1,
                  db ? (int)db->isPrimaryActual() : -1,
                  db ? (int)db->isSpecialActual() : -1);
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
  // Sync policy is contract-driven via needsSync().
  // SlotSyncGuard: prepareSpecialUse now, registerSpecialUse on scope exit.
  bool syncNeeded2 = needsSync();
  if (DSP_DIAG_ENABLED(STREAM_SYNC)) {
    DSP_DIAG_SLOT(STREAM_SYNC, stepIdx,
                  "warmupSlot sync=%s reason=%s exec=%d mode=%s",
                  syncNeeded2 ? "YES" : "NO", syncReason(),
                  executeCount_, ModeContract::modeName(static_cast<int>(graphExecutionMode_)));
  }
  SlotSyncGuard warmupSyncGuard(syncNeeded2, ctx.fastpath_out(), ctx.fastpath_in(),
                                 stepIdx, slot.ident.opName.c_str(), executeCount_);

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

  // ── Capture probe: immediately before op->execute() ──────────────────────
  {
    bool _capInvalid = false;
    void* _capStream = dspGetExecutionStream();
    DSP_CAPTURE_PROBE(_capStream,
                      stepIdx, "BEFORE_OP_EXECUTE",
                      slot.ident.opName.c_str(), _capInvalid);
  }

  // PRE-EXEC input metadata for variable-input slots (exec 2-4 only to limit noise).
  if (DSP_DIAG_ENABLED(VERIFY) && executeCount_ >= 2 && executeCount_ <= 4 && !tl_graphExecutionActive) {
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {  // external input
        NDArray* inp = ctx.fastpath_in().size() > (size_t)i ? ctx.fastpath_in()[i] : nullptr;
        if (inp != nullptr && inp->dataBuffer() != nullptr && inp->specialBuffer() != nullptr) {
          size_t bytes = dspSafeByteCount(inp);
          DSP_DIAG(VERIFY, "PRE_EXEC_INPUT: slot=%d op=%s input[%d] extIdx=%d "
                   "specialBuf=%p bytes=%zu dtype=%s pAct=%d sAct=%d exec=%d",
                   stepIdx, slot.ident.opName.c_str(), i, -(srcIdx + 1),
                   inp->specialBuffer(), bytes,
                   DataTypeUtils::asString(inp->dataType()).c_str(),
                   inp->dataBuffer()->isPrimaryActual() ? 1 : 0,
                   inp->dataBuffer()->isSpecialActual() ? 1 : 0,
                   executeCount_);
        }
      }
    }
  }

  try {
    status = platformExecuteSlot(slot, ctx);
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
              (int)cacheHit, executeCount_, (int)planLifecycle_.isShapesFrozen());
    // Propagate error detail to Java via errorReference
    dspPropagateSlotError(stepIdx, slot,
                           (std::string("exec exception: ") + e.what()).c_str(),
                           inputShapes, outputShapesStr, iArgsStr);
    // Clear epilogue state even on exception
    if (slot.flags.ltEpilogueType > 0) platformClearLtEpilogue();
    return Status::KERNEL_FAILURE;
  }
  }  // end else (!shapeOnlyMode_)

  // POST-EXEC output metadata for debugging stale output (exec 2-4 only).
  if (DSP_DIAG_ENABLED(VERIFY) && executeCount_ >= 2 && executeCount_ <= 4 && !tl_graphExecutionActive) {
    for (int i = 0; i < numActualOutputs; i++) {
      NDArray* out = ctx.fastpath_out().size() > (size_t)i ? ctx.fastpath_out()[i] : nullptr;
      if (out != nullptr && out->dataBuffer() != nullptr && out->specialBuffer() != nullptr) {
        size_t bytes = dspSafeByteCount(out);
        int si = i < slot.wiring.numOutputs ? slot.wiring.outputSlotIndices[i] : -1;
        float __fv = -999.0f;
        if (out->dataType() == DataType::FLOAT32 && !out->isEmpty() && out->lengthOf() > 0)
          __fv = out->e<float>(0);
        DSP_DIAG(VERIFY, "POST_EXEC_OUTPUT: slot=%d op=%s output[%d] outSlot=%d "
                 "specialBuf=%p bytes=%zu dtype=%s first=%.5f exec=%d",
                 stepIdx, slot.ident.opName.c_str(), i, si,
                 out->specialBuffer(), bytes,
                 DataTypeUtils::asString(out->dataType()).c_str(), (double)__fv, executeCount_);
      }
    }
  }

  // ── Capture probe: after Step 4 (op execution) ──────────────────────────
  {
    bool _capInvalid = false;
    void* _capStream = dspGetExecutionStream();
    DSP_CAPTURE_PROBE(_capStream,
                      stepIdx, "AFTER_OP_EXECUTE",
                      slot.ident.opName.c_str(), _capInvalid);
  }

#ifdef SD_CUDA
  // BUF_FP_RING slot-output fingerprints: async XOR fingerprint of each wired
  // output, enqueued on the SAME stream the op's kernels used. Native outputs
  // produced inside the outer composite region use the post-capture step; the
  // actual CUDA stream capture check still suppresses graph-baked probes.
  // Env-gated (BUF_FP_RING=1); dumped at releaseGpuIntermediates/teardown.
  if (fpRingEnabled_ && status == Status::OK) {
    auto* fpLc = ctx.launchContext() != nullptr ? ctx.launchContext()
                                                : LaunchContext::defaultContext();
    auto* fpStreamPtr = fpLc != nullptr ? fpLc->getCudaStream() : nullptr;
    if (fpStreamPtr != nullptr) {
      void* fpSavedCompositeCaptureStream = tl_compositeCaptureStream;
      int fpStep = execCtx != nullptr ? execCtx->fpStep : executeCount_;
      if (fpSavedCompositeCaptureStream != nullptr) {
        tl_compositeCaptureStream = nullptr;
        fpStep += BUF_FP_POST_CAPTURE_STEP_OFFSET;
      }
      for (int i = 0; i < numActualOutputs && i < slot.wiring.numOutputs; i++) {
        int fpSi = slot.wiring.outputSlotIndices[i];
        NDArray* fpOut = outputs[i];
        int fpTrack = numExternalInputs_ + fpSi;
        bool fpRequested = numRequestedOutputs_ > 0 && requestedOutputSlotIndices_ != nullptr &&
                           requestedOutputSlotIndices_[0] == fpSi;
        if (fpRequested && fpOut != nullptr && fpOut->specialBuffer() != nullptr &&
            !fpOut->isEmpty()) {
          recordBufFingerprintPublic(*fpStreamPtr, fpStep, BUF_FP_REQUESTED_OUTPUT_TRACK,
                                     fpOut->specialBuffer(), dspSafeByteCount(fpOut));
          if (fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag[0] == '\0') {
            snprintf(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag,
                     sizeof(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag), "req.writer");
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].extIdx = -1;
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].groupIdx = -1;
            fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].whichAB = -1;
          }
          DSP_DIAG(MEMORY, "BUF_FP_STREAM plan=%p step=%d slot=%d path=warmup writer=%p dsp=%p lc=%p",
                   (void*)this, fpStep, fpSi, (void*)*fpStreamPtr,
                   execCtx != nullptr ? execCtx->dspStream : nullptr,
                   execCtx != nullptr ? execCtx->lcDefaultStream : nullptr);
        }
        if (fpOut != nullptr && fpSi >= 0 && fpTrack < BUF_FP_TRACE_TRACK &&
            fpOut->specialBuffer() != nullptr && !fpOut->isEmpty()) {
          recordBufFingerprintPublic(*fpStreamPtr, fpStep, fpTrack,
                                     fpOut->specialBuffer(), dspSafeByteCount(fpOut));
          if (fpLabels_[fpTrack].tag[0] == '\0') {
            snprintf(fpLabels_[fpTrack].tag, sizeof(fpLabels_[fpTrack].tag), "s%d", fpSi);
            fpLabels_[fpTrack].extIdx = -1;
            fpLabels_[fpTrack].groupIdx = -1;
            fpLabels_[fpTrack].whichAB = -1;
          }
        }
      }
      tl_compositeCaptureStream = fpSavedCompositeCaptureStream;
    }
  }
#endif

  // Post-execute corruption scan (warmup path)
  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      int si = i < slot.wiring.numOutputs ? slot.wiring.outputSlotIndices[i] : -1;
      if (!validateSlotShapeBufferAlignment(outputs[i], si,
              "AFTER_warmup_op_execute", executeCount_)) {
        DSP_DIAG(MEMORY,
                 "CORRUPTION_AFTER_EXEC_WU: slot %d (%s) output[%d] outSlot=%d "
                 "corrupted BY warmup op->execute. execCount=%d",
                 stepIdx, slot.ident.opName.c_str(), i, si, executeCount_);
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
              (int)cacheHit, executeCount_, (int)planLifecycle_.isShapesFrozen());
  }

  // registerSpecialUse + anomaly detection handled by warmupSyncGuard destructor

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
                   executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
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

      // Accept the view-producer's actual output on dtype+shape match WITHOUT requiring
      // strideEquals: permute/transpose/slice legitimately emit a VIEW whose strides
      // differ from the contiguous strides reported by the op's shape-fn. Rejecting on a
      // stride mismatch left this slot's unwritten (zero) array installed -> GQA K/V=0 ->
      // dead attention -> decode garbage (firstToken=11126). Only a true dtype/shape
      // mismatch is rejected here; the strideEquals gate stays on the frozen/refresh paths.
      if (!outputWrapperShapeAndTypeMatch(ctxOutputs[i], outputShapes[i])) {
        DSP_DIAG(SHAPE,
                 "CTX_OUTPUT_REJECT(view-producer-shape/dtype): slot=%d op=%s outIdx=%d expected=%s actual=%s",
                 stepIdx, slot.ident.opName.c_str(), i,
                 ShapeUtils::shapeAsString(outputShapes[i]).c_str(),
                 ShapeUtils::shapeAsString(ctxOutputs[i]).c_str());
        continue;
      }

      if (!viewProducerDetectionDone_) {
        if (ctxOutputs[i] != outputs[i]) {
          markViewProducer(stepIdx, "view-producer-detect");
          // writeOutputSlot handles safe deletion of old arrays
          writeOutputSlot(si, ctxOutputs[i], "view-producer-detect");
          DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                              ctxOutputs[i] != nullptr && ctxOutputs[i]->dataBuffer() != nullptr
                                  ? ctxOutputs[i]->dataBuffer()->getLenInBytes()
                                  : 0,
                              stream, "view-producer-detect");
          outputSlots_[si] = ctxOutputs[i];
        }
      } else if (slot.slotPhase.isViewProducer) {
        // writeOutputSlot handles safe deletion of old arrays
        writeOutputSlot(si, ctxOutputs[i], "view-producer-update");
        DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                            ctxOutputs[i] != nullptr && ctxOutputs[i]->dataBuffer() != nullptr
                                ? ctxOutputs[i]->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "view-producer-update");
        outputSlots_[si] = ctxOutputs[i];
      } else if (ctxOutputs[i] != outputs[i]) {
        // Late-detection: slot is in a later segment (viewProducerDetectionDone_=true)
        // and was never detected in segment 0, OR was demoted by refreshStaleViewWrappers
        // (isViewProducer=false) but the op still produced a view this execution.
        // Without this branch the view is ORPHANED: outputSlots_[si] keeps a stale/zero
        // pre-allocated array while the actual view lives in ctxOutputs[i].
        // This is the root of GQA K/V=0 → dead attention → firstToken=11126 garbage.
        markViewProducer(stepIdx, "view-producer-detect-late");
        writeOutputSlot(si, ctxOutputs[i], "view-producer-detect-late");
        DSP_DIAG_SLOT_WRITE(si, slot.ident.opName.c_str(),
                            ctxOutputs[i] != nullptr && ctxOutputs[i]->dataBuffer() != nullptr
                                ? ctxOutputs[i]->dataBuffer()->getLenInBytes()
                                : 0,
                            stream, "view-producer-detect-late");
        outputSlots_[si] = ctxOutputs[i];
      }
    }
  }

  // ── DSP debug: post-execute output fingerprint (warmup path) ──
  if (executeCount_ >= 2 && Environment::getInstance().isDebugAndVerbose() && status == Status::OK) {
    for (int i = 0; i < numActualOutputs; i++) {
      auto* array = outputs[i];
      if (array != nullptr && !array->isEmpty() && array->lengthOf() > 0) {
        DSP_DIAG(VERIFY,
                 "DSP_FINGERPRINT exec=%d slot=%d out=%d op=%s shape=%s dtype=%s len=%lld asyncValues=true",
                 executeCount_, stepIdx, i, slot.ident.opName.c_str(),
                 ShapeUtils::shapeAsString(array).c_str(),
                 DataTypeUtils::asString(array->dataType()).c_str(),
                 (long long)array->lengthOf());
      }
    }
  }



  // Promote slots to FROZEN on the first frozen execution so subsequent
  // steps can use the frozen fast path (frozenContextReady requires FROZEN).
  // Skip sealing when syncOverride is active (post-markVariable warmup):
  // the segment has been invalidated and needs fresh execution, not frozen
  // context. Re-sealing during warmup causes the frozen path to fire with
  // stale cached inputs on the next step.
  if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 0 && status == Status::OK
      && syncOverrideDepth_ <= 0) {
    if (!slot.slotPhase.isSealed()) {
      slot.slotPhase.seal(false);  // PRIMARY: freeze (not constant)
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
//   Status::EQ_TRUE — NextIteration, caller should jump back to loopBackTarget
//   Negative int via KERNEL_FAILURE — error
//
// Note: This method is called from executeSegmentSlotBySlot (in _segments.cpp)
// which has been updated to handle backward jumps for loops.

// ─── Gap fast-path execution ─────────────────────────────────────────────────
// Ultra-fast slot execution for steady-state cached gap path.
// Preconditions (caller must guarantee):
//   - executeCount_ >= 5 (well past warmup)
//   - planLifecycle_.pointersStable() (no pointer changes)
//   - slot is classified as EXECUTE in gap cache (not identity/view/batched/constant)
//   - needsSync() == false (no H2D/D2H needed)
//
// Bypasses: DeferredDeleteGuard, input validation, frozen fast-path attempt,
//   view checks, sync, prezero, corruption scans, diagnostics, output reconciliation.
// This eliminates ~20µs of per-slot overhead in the gap execution hot loop.

Status NativeDynamicShapePlan::executeSlotGapFast(
    int slotIdx, NDArray** externalArrays, int numExt) {
  NativeSlot& slot = slots_[slotIdx];

  if (contextPool_[slotIdx] == nullptr) {
    // Context not initialized — fall back to full executeSlot
    DSP_DIAG_SLOT(EXECUTE, slotIdx,
        "GAP_FAST_FALLBACK: slot=%d op=%s no contextPool entry — using full executeSlot path executeCount=%d",
        slotIdx, slot.ident.opName.c_str(), executeCount_);
    return executeSlot(slotIdx, externalArrays, numExt, nullptr);
  }
  auto& ctx = *contextPool_[slotIdx];

  // Direct input pointer assignment — bypasses Context::setInputArray validation.
  // Safe because: (1) executeCount_ >= 5 means all slots are validated,
  // (2) pointers are stable so no null/stale sources, (3) vector already sized
  // from prior executions. This eliminates ~2000 null checks + dtype checks per step.
  auto& fpIn = ctx.fastpath_in();
  if (static_cast<int>(fpIn.size()) < slot.wiring.numInputs) {
    // Vector not sized from prior executions — fall back to full path
    DSP_DIAG_SLOT(EXECUTE, slotIdx,
        "GAP_FAST_FALLBACK: slot=%d op=%s fpIn.size()=%zu < numInputs=%d — using full executeSlot path executeCount=%d",
        slotIdx, slot.ident.opName.c_str(), fpIn.size(), slot.wiring.numInputs, executeCount_);
    return executeSlot(slotIdx, externalArrays, numExt, nullptr);
  }
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExt) {
        fpIn[i] = externalArrays[extIdx];
      }
    } else if (srcIdx < totalOutputSlots_) {
      fpIn[i] = outputSlots_[srcIdx];
    }
  }

  // Shape override already set from prior frozen-context path (executeCount_ >= 3).
  // Prezero: handled by batch prezero pass in compositeReplay before gap loop.
  // Sync not needed: device-resident in steady state.

  // ── View-op fast path ──────────────────────────────────────────────────
  // For view-capable ops (reshape, reshape_no_copy, permute, etc.) in steady state,
  // if the output already shares the input's data buffer, the op is a metadata-only
  // no-op. Advance logical dependency generations without claiming a device write:
  // the view aliases data produced by its input and this branch launches no kernel.
  // The first executeSlotGapFast call goes through execute() which establishes
  // the view; all subsequent calls hit this fast path.
  if (slot.isViewCapableOp() && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
    NDArray* input0 = fpIn[0];
    int outSi = slot.wiring.outputSlotIndices[0];
    if (input0 != nullptr && outSi >= 0 && outSi < totalOutputSlots_) {
      NDArray* output = outputSlots_[outSi];
      if (output != nullptr && input0->dataBuffer() != nullptr &&
          output->dataBuffer() == input0->dataBuffer()) {
        dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
        slot.bumpGeneration();
        DSP_DIAG_SLOT(EXECUTE, slotIdx,
            "GAP_FAST_VIEW_REUSE: slot=%d op=%s outSi=%d buffer already shared with input — metadata reuse executeCount=%d",
            slotIdx, slot.ident.opName.c_str(), outSi, executeCount_);
        return Status::OK;
      }
      // View buffer mismatch means this slot may produce a fresh view/wrapper.
      // The fast path cannot safely adopt replacement wrappers; use the normal
      // executeSlot path so view-producer detection and output ownership run.
      DSP_DIAG_SLOT(EXECUTE, slotIdx,
          "GAP_FAST_VIEW_MISMATCH: slot=%d op=%s outSi=%d "
          "outDb=%p inDb=%p — using full executeSlot path executeCount=%d",
          slotIdx, slot.ident.opName.c_str(), outSi,
          output != nullptr ? (void*)output->dataBuffer() : nullptr,
          input0->dataBuffer() != nullptr ? (void*)input0->dataBuffer() : nullptr,
          executeCount_);
      return executeSlot(slotIdx, externalArrays, numExt, nullptr);
    }
  }

  // Execute the op directly.
  Status result = platformExecuteSlot(slot, ctx);

  if (DSP_DIAG_ENABLED(EXECUTE) && result != Status::OK) {
    DSP_DIAG_SLOT(EXECUTE, slotIdx,
        "GAP_FAST_EXEC_FAIL: slot=%d op=%s status=%d executeCount=%d",
        slotIdx, slot.ident.opName.c_str(), static_cast<int>(result), executeCount_);
  }
  if (result == Status::OK) {
    auto& fpOut = ctx.fastpath_out();
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      int si = slot.wiring.outputSlotIndices[i];
      if (si < 0 || si >= totalOutputSlots_) continue;
      NDArray* out = (i < static_cast<int>(fpOut.size()) && fpOut[i] != nullptr)
                         ? fpOut[i]
                         : outputSlots_[si];
      if (out != nullptr) {
        platformReconcileOutputActuality("gap-fast", slotIdx, slot, out);
#ifdef SD_CUDA
        // BUF_FP_RING slot-output fingerprint (gap-fast path); native composite
        // gaps land in the shared post-capture step.
        if (fpRingEnabled_) {
          auto* fpLc = ctx.launchContext() != nullptr ? ctx.launchContext()
                                                      : LaunchContext::defaultContext();
          auto* fpStreamPtr = fpLc != nullptr ? fpLc->getCudaStream() : nullptr;
          void* fpSavedCompositeCaptureStream = tl_compositeCaptureStream;
          int fpStep = executeCount_;
          if (fpStreamPtr != nullptr && fpSavedCompositeCaptureStream != nullptr) {
            tl_compositeCaptureStream = nullptr;
            fpStep += BUF_FP_POST_CAPTURE_STEP_OFFSET;
          }
          int fpTrack = numExternalInputs_ + si;
          bool fpRequested = numRequestedOutputs_ > 0 && requestedOutputSlotIndices_ != nullptr &&
                             requestedOutputSlotIndices_[0] == si;
          if (fpRequested && fpStreamPtr != nullptr && out->specialBuffer() != nullptr &&
              !out->isEmpty()) {
            recordBufFingerprintPublic(*fpStreamPtr, fpStep, BUF_FP_REQUESTED_OUTPUT_TRACK,
                                       out->specialBuffer(), dspSafeByteCount(out));
            if (fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag[0] == '\0') {
              snprintf(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag,
                       sizeof(fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].tag), "req.writer");
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].extIdx = -1;
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].groupIdx = -1;
              fpLabels_[BUF_FP_REQUESTED_OUTPUT_TRACK].whichAB = -1;
            }
            auto* fpExecCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
            DSP_DIAG(MEMORY, "BUF_FP_STREAM plan=%p step=%d slot=%d path=gap-fast writer=%p dsp=%p lc=%p",
                     (void*)this, fpStep, si, (void*)*fpStreamPtr,
                     fpExecCtx != nullptr ? fpExecCtx->dspStream : nullptr,
                     fpExecCtx != nullptr ? fpExecCtx->lcDefaultStream : nullptr);
          }
          if (fpStreamPtr != nullptr && fpTrack < BUF_FP_TRACE_TRACK &&
              out->specialBuffer() != nullptr && !out->isEmpty()) {
            recordBufFingerprintPublic(*fpStreamPtr, fpStep, fpTrack,
                                       out->specialBuffer(), dspSafeByteCount(out));
            if (fpLabels_[fpTrack].tag[0] == '\0') {
              snprintf(fpLabels_[fpTrack].tag, sizeof(fpLabels_[fpTrack].tag), "s%d", si);
              fpLabels_[fpTrack].extIdx = -1;
              fpLabels_[fpTrack].groupIdx = -1;
              fpLabels_[fpTrack].whichAB = -1;
            }
          }
          tl_compositeCaptureStream = fpSavedCompositeCaptureStream;
        }
#endif
      }
      dirtySlotGenerations_[si] = currentDirtyGeneration_;
    }
    slot.bumpGeneration();
  }
  return result;
}

}  // namespace graph
}  // namespace sd
