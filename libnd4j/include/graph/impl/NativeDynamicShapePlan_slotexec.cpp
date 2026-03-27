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
#include <system/Environment.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/OpRegistrator.h>

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

namespace {
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
    auto normalized = normalizeOpName_slotexec(sl.opName);
    if (VALUE_INDEPENDENT_OPS.count(normalized) > 0) {
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
      sl.frozenConstantSlot = true;
      frozenConstCount++;
      if (isValueIndependentSlot[s]) valueIndepCount++;
    }
  }
  // Collect frozen output slot indices for quick lookup
  std::unordered_set<int> frozenOutputSlots;
  for (int s = 0; s < numSlots_; s++) {
    if (slots_[s].frozenConstantSlot) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        frozenOutputSlots.insert(slots_[s].outputSlotIndices[o]);
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

  DSP_DIAG(SHAPE, "frozen constant detection: %d/%d slots are frozen constants (%d value-independent, %d in-place disabled)",
            frozenConstCount, numSlots_, valueIndepCount, disabledInPlace);
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
          outputSlots_[si] = input;
        }
      }
#ifdef SD_CUDA
      if (Environment::getInstance().tritonVerifyKernels()) {
        dspLogSlotOutput(stepIdx, slot.opName.c_str(), "IDENTITY",
                            outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
      }
#endif
      return Status::OK;
    }
  }

  // ── Frozen constant optimization ──────────────────────────────────────────
  // Only skip when shapesFrozen_ && executeCount_ > 0, because that's when
  // outputSlots_ is populated from slotArrayCache_ (line 921-922).  When
  // shapesFrozen_=false, outputSlots_ is zeroed — skipping a slot would
  // leave a NULL entry and downstream slots would get NULL inputs.
  if (slot.frozenConstantSlot && shapesFrozen_ && executeCount_ > 0) {
#ifdef SD_CUDA
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s [SKIPPED:frozen-const]", stepIdx, slot.opName.c_str());
    }
#endif
    return Status::OK;
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
      if (primaryInput == nullptr && primarySrcIdx < totalOutputSlots_ && slotArrayCache_[primarySrcIdx] != nullptr) {
        primaryInput = slotArrayCache_[primarySrcIdx];
        outputSlots_[primarySrcIdx] = slotArrayCache_[primarySrcIdx];
      }
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
          if (secondaryInputs[ci] == nullptr && secSrc < totalOutputSlots_ && slotArrayCache_[secSrc] != nullptr) {
            secondaryInputs[ci] = slotArrayCache_[secSrc];
            outputSlots_[secSrc] = slotArrayCache_[secSrc];
          }
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
          output = nullptr;
        }
      }
      if (output == nullptr) {
        output = new NDArray(const_cast<LongType*>(outputShapeInfo), true, LaunchContext::defaultContext());
        slotArrayCache_[lastOutputSlotIdx] = output;
      }
      if (!isBatchZeroActive()) {
        if (isBatchZeroRegistering() && output->specialBuffer() != nullptr) {
          registerBatchZeroBuffer(output->specialBuffer(),
                                  output->dataBuffer()->getLenInBytes());
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
    outputSlots_[lastOutputSlotIdx] = output;
    for (int ci = 0; ci < slot.fusedChainLength - 1; ci++) {
      int chainSlotIdx = slot.fusedChainSlots[ci];
      int chainOutputSlotIdx = slots_[chainSlotIdx].outputSlotIndices[0];
      if (chainOutputSlotIdx >= 0 && chainOutputSlotIdx < totalOutputSlots_) {
        outputSlots_[chainOutputSlotIdx] = output;
      }
    }

#ifdef SD_CUDA
    if (Environment::getInstance().tritonVerifyKernels()) {
      int outIndices[1] = { lastOutputSlotIdx };
      dspLogSlotOutput(stepIdx, slot.opName.c_str(), "FUSED_HEAD",
                          outputSlots_, outIndices, 1, totalOutputSlots_);
    }
#endif

    return Status::OK;
  }

  // ── Fast path: frozen context ────────────────────────────────────────────
  if (slot.frozenContextReady) {

    // ── View-capable fast path (reshape/expand_dims/squeeze/strided_slice) ──
    if (slot.isViewCapableOp && slot.numInputs >= 1 && slot.numOutputs >= 1) {
      int si = slot.outputSlotIndices[0];
      if (si >= 0 && si < totalOutputSlots_) {
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
          // Compute view offset: 0 for reshape/squeeze/expand_dims/permute,
          // computed from begin indices for strided_slice.
          LongType viewOffset = 0;
          bool isStridedSlice = (slot.opName == "strided_slice");
          if (isStridedSlice) {
            // Gather begin/end/strides inputs for offset computation
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
            viewOffset = computeStridedSliceViewOffset(slot, input0, ssInputs.data(), slot.numInputs);
            if (viewOffset < 0) {
              // Can't create view (non-unit strides or newAxisMask) — fall through to normal execution
              slot.frozenContextReady = false;
              goto normalExecution;
            }
          }

          NDArray* currentOut = outputSlots_[si];
          if (currentOut != nullptr && currentOut->dataBuffer() == input0->dataBuffer()
              && currentOut->offset() == viewOffset) {
            return Status::OK;
          }
          if (slot.shapeCacheValid && !slot.cachedOutputShapes.empty()) {
            const LongType* outShapeInfo = slot.cachedOutputShapes[0];
            LongType outLen = shape::length(outShapeInfo);
            LongType inLen = input0->lengthOf();
            if (outLen > 0 && outLen <= inLen) {
              NDArray* newView = new NDArray(input0->dataBuffer(),
                                             const_cast<LongType*>(outShapeInfo),
                                             LaunchContext::defaultContext(),
                                             viewOffset);
              outputSlots_[si] = newView;
              slotIsViewProducer_[si] = true;
              auto& ctx2 = *contextPool_[stepIdx];
              ctx2.setOutputArray(0, newView);
              ctx2.setInputArray(0, input0);
              NDArray* old = slotArrayCache_[si];
              if (old != nullptr && old != newView) {
                pendingClose_.push_back(old);
              }
              slotArrayCache_[si] = newView;
              return Status::OK;
            } else if (outLen == 0 && inLen > 0) {
              // Stale empty shape from Step 0 — input is now non-empty.
              // Invalidate frozen state and fall through to normal path
              // for shape re-inference with actual input shapes.
              DSP_DIAG_SLOT(SHAPE, stepIdx,
                  "view-capable slot %d (%s): frozen shape empty but input "
                  "non-empty (len=%lld) — re-inferring via normal path",
                  stepIdx, slot.opName.c_str(), inLen);
              slot.frozenContextReady = false;
              slot.shapeCacheValid = false;
              goto normalExecution;
            }
          }
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
          DSP_DIAG_SLOT(SHAPE, stepIdx,
              "view-capable slot %d (%s): frozen outputs empty but input non-empty "
              "(len=%lld) — re-inferring via normal path (fallback)",
              stepIdx, slot.opName.c_str(), inp0->lengthOf());
          slot.frozenContextReady = false;
          slot.shapeCacheValid = false;
          goto normalExecution;
        }
      }
    }

    auto& ctx = *contextPool_[stepIdx];

    // Refresh inputs that change each decode step
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
          ctx.setInputArray(i, externalArrays[extIdx]);
        }
      } else if (srcIdx < totalOutputSlots_ && slotIsViewProducer_[srcIdx]) {
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

        if (isBatchZeroRegistering() && ctxOuts[i]->specialBuffer() != nullptr) {
          registerBatchZeroBuffer(ctxOuts[i]->specialBuffer(),
                                  ctxOuts[i]->dataBuffer()->getLenInBytes());
        }
        ctxOuts[i]->nullify();
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
        ctxOuts[i]->tickWriteDevice();
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots_) {
          NDArray* oldCached = slotArrayCache_[si];
          if (oldCached != nullptr && oldCached != ctxOuts[i]) {
            pendingClose_.push_back(oldCached);
          }
          outputSlots_[si] = ctxOuts[i];
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
      if (inputs[i] == nullptr && srcIdx < totalOutputSlots_ && slotArrayCache_[srcIdx] != nullptr) {
        inputs[i] = slotArrayCache_[srcIdx];
        outputSlots_[srcIdx] = slotArrayCache_[srcIdx];
      }
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
  }

#ifdef SD_CUDA
  // Debug: dump native slot 354 (pow) input to compare with Triton path
  if (sd::Environment::getInstance().isDebug() && stepIdx == 354
      && slot.numInputs > 0 && inputs[0] != nullptr) {
    auto* inp = inputs[0];
    DSP_DIAG(VERIFY, "NATIVE_SLOT354: exec=%d srcIdx=%d len=%lld dtype=%d addr=%p pAct=%d sAct=%d",
             executeCount_, slot.inputSourceIndices[0], (long long)inp->lengthOf(),
             (int)inp->dataType(), inp->specialBuffer(),
             inp->dataBuffer() ? (inp->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
             inp->dataBuffer() ? (inp->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
    if (inp->specialBuffer() != nullptr && inp->lengthOf() > 0 && inp->dataType() == FLOAT32) {
      int dc = std::min((int)inp->lengthOf(), 8);
      std::vector<float> hb(dc);
      cudaDeviceSynchronize();
      cudaMemcpy(hb.data(), inp->specialBuffer(), dc * 4, cudaMemcpyDeviceToHost);
      std::string vs;
      for (int v = 0; v < dc; v++) {
        if (v > 0) vs += ",";
        char buf[32]; snprintf(buf, sizeof(buf), "%.6f", hb[v]); vs += buf;
      }
      DSP_DIAG(VERIFY, "NATIVE_SLOT354: exec=%d values: %s", executeCount_, vs.c_str());
    }
  }
#endif

  // ── Step 2: Shape inference ──────────────────────────────────────────────
  LongType shapeKey = 0;
  bool cacheHit;
  if (shapesFrozen_ && executeCount_ > 0 && slot.shapeCacheValid &&
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
    cacheHit = slot.shapeCacheValid && (slot.cachedShapeKey == shapeKey);

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
      return Status::KERNEL_FAILURE;
    }
    if (shapeList == nullptr || shapeList->size() == 0) {
      DSP_DIAG_SLOT(SHAPE, stepIdx, "shape inference returned null for slot %d (%s)",
                stepIdx, slot.opName.c_str());
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
          outputSlots_[slotIdx] = inPlaceBuffer;
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
          std::vector<LongType> shape(rank);
          for (int d = 0; d < rank; d++) shape[d] = shapeInfo[d + 1];
          outputs[i] = new NDArray(order, shape, dt);
          outputSlots_[si] = outputs[i];
          slotArrayCache_[si] = outputs[i];
        }

        goto step4_execute;
      }
    }
  }

  // ── View-capable ops: share input 0's DataBuffer for output 0 ──────────
  if (slot.isViewCapableOp && slot.numInputs >= 1 && numActualOutputs >= 1) {
    NDArray* input0 = inputs[0];
    if (input0 != nullptr && input0->dataBuffer() != nullptr &&
        input0->ews() == 1 && input0->ordering() == 'c') {
      const LongType* outShapeInfo = outputShapes[0];
      LongType outLen = shape::length(outShapeInfo);
      LongType inLen = input0->lengthOf();

      // Compute view offset: 0 for reshape/squeeze/expand_dims/permute,
      // computed from begin indices for strided_slice.
      LongType viewOffset = 0;
      if (slot.opName == "strided_slice") {
        viewOffset = computeStridedSliceViewOffset(slot, input0, inputs.data(), slot.numInputs);
        if (viewOffset < 0) {
          // Non-unit strides or newAxisMask — can't create contiguous view, fall through
          goto step3_allocate;
        }
      }

      if (outLen > 0 && outLen <= inLen) {
        int slotIdx = slot.outputSlotIndices[0];
        NDArray* view = new NDArray(input0->dataBuffer(),
                                     const_cast<LongType*>(outShapeInfo),
                                     LaunchContext::defaultContext(),
                                     viewOffset);
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

        for (int i = 1; i < numActualOutputs; i++) {
          int si = slot.outputSlotIndices[i];
          const LongType* shapeInfo = outputShapes[i];
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
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

    NDArray* cached = slotArrayCache_[slotIdx];
    if (cached != nullptr) {
      const LongType* cachedShape = cached->shapeInfo();
      if (shape::equalsSoft(cachedShape, shapeInfo) &&
          ArrayOptions::dataType(cachedShape) == dt) {
        if (!isBatchZeroActive()) {
          if (isBatchZeroRegistering() && cached->specialBuffer() != nullptr) {
            registerBatchZeroBuffer(cached->specialBuffer(),
                                    cached->dataBuffer()->getLenInBytes());
          }
          cached->nullify();
        }
        outputs[i] = cached;
        outputSlots_[slotIdx] = cached;
        continue;
      } else {
        pendingCloseBytes_ += cached->lengthOf() * cached->sizeOfT();
        pendingClose_.push_back(cached);
        slotArrayCache_[slotIdx] = nullptr;
      }
    }

    // Empty arrays (ARRAY_EMPTY flag): use full-shapeInfo constructor to
    // preserve the flag.  NDArray(order, shape, dt) rebuilds shapeInfo from
    // dimensions alone, losing ARRAY_EMPTY and crashing concat/etc.
    if (shape::isEmpty(const_cast<LongType*>(shapeInfo))) {
      NDArray* emptyOut = new NDArray(const_cast<LongType*>(shapeInfo), true);
      outputs[i] = emptyOut;
      outputSlots_[slotIdx] = emptyOut;
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
          maxOut = new NDArray(order, maxShape, dt);
          if (!isBatchZeroActive()) {
            if (isBatchZeroRegistering() && maxOut->specialBuffer() != nullptr) {
              registerBatchZeroBuffer(maxOut->specialBuffer(),
                                      maxOut->dataBuffer()->getLenInBytes());
            }
            maxOut->nullify();
          }
        } catch (const std::exception& e) {
          DSP_DIAG_SLOT(MEMORY, stepIdx, "max-allocation FAILED at slot %d (%s): %s",
                    stepIdx, slot.opName.c_str(), e.what());
          maxOut = new NDArray(order, shape, dt);
          if (slot.needsZeroedOutput && !isBatchZeroActive()) {
            if (isBatchZeroRegistering() && maxOut->specialBuffer() != nullptr) {
              registerBatchZeroBuffer(maxOut->specialBuffer(),
                                      maxOut->dataBuffer()->getLenInBytes());
            }
            maxOut->nullify();
          }
        }

        outputs[i] = maxOut;
        outputSlots_[slotIdx] = maxOut;
        slotArrayCache_[slotIdx] = maxOut;
        maxAllocatedSlots_.insert(slotIdx);
        continue;
      }
      NDArray* cached2 = slotArrayCache_[slotIdx];
      if (cached2 != nullptr) {
        outputs[i] = cached2;
        outputSlots_[slotIdx] = cached2;
        continue;
      }
    }

    NDArray* out = nullptr;
    try {
      out = new NDArray(order, shape, dt);
      if (slot.needsZeroedOutput && !isBatchZeroActive()) {
        if (isBatchZeroRegistering() && out->specialBuffer() != nullptr) {
          registerBatchZeroBuffer(out->specialBuffer(),
                                  out->dataBuffer()->getLenInBytes());
        }
        out->nullify();
      }
    } catch (const std::exception& e) {
      DSP_DIAG_SLOT(MEMORY, stepIdx, "output ALLOC EXCEPTION at slot %d (%s) output[%d]: %s",
                stepIdx, slot.opName.c_str(), i, e.what());
      return Status::KERNEL_FAILURE;
    }

    outputs[i] = out;
    outputSlots_[slotIdx] = out;
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
      bool isViewOp = slot.isViewCapableOp ||
                      (normalizeOpName_slotexec(slot.opName) == "permute");
      bool primaryInputNonEmpty = (slot.numInputs > 0 && inputs[0] != nullptr &&
                                   !inputs[0]->isEmpty() && inputs[0]->lengthOf() > 0);
      if (isViewOp && primaryInputNonEmpty) {
        DSP_DIAG_SLOT(EXECUTE, stepIdx,
                  "view-capable slot %d (%s): outputs empty but input[0] non-empty "
                  "(length=%lld) — forcing execution instead of skip",
                  stepIdx, slot.opName.c_str(), inputs[0]->lengthOf());
        // Invalidate shape cache so next execution re-derives shapes
        slot.shapeCacheValid = false;
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

  // Log shapes for matmul gap slots to diagnose capture shape mismatches
  if (normalizeOpName_slotexec(slot.opName) == "matmul") {
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
    return Status::KERNEL_FAILURE;
  }

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
    DSP_DIAG(EXECUTE, "SLOT EXEC FAIL: slot %d (%s) status=%d, inputs=[%s], outputs=[%s], iArgs=[%s], "
              "cacheHit=%d executeCount=%d shapesFrozen=%d",
              stepIdx, slot.opName.c_str(), static_cast<int>(status),
              inputShapes.c_str(), outputShapesStr.c_str(), iArgsStr.c_str(),
              (int)cacheHit, executeCount_, (int)shapesFrozen_);
  }

  for (int i = 0; i < numActualOutputs; i++) {
    if (outputs[i] != nullptr) {
      outputs[i]->tickWriteDevice();
    }
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
          if (oldCached != nullptr && oldCached != ctxOutputs[i]) {
            pendingClose_.push_back(oldCached);
          }
          outputSlots_[si] = ctxOutputs[i];
          slotArrayCache_[si] = ctxOutputs[i];
        }
      } else if (slotIsViewProducer_[si]) {
        NDArray* oldCached = slotArrayCache_[si];
        if (oldCached != nullptr && oldCached != ctxOutputs[i]) {
          pendingClose_.push_back(oldCached);
        }
        outputSlots_[si] = ctxOutputs[i];
        slotArrayCache_[si] = ctxOutputs[i];
      }
    }
  }

  if (shapesFrozen_ && executeCount_ > 0 && status == Status::OK) {
    slot.frozenContextReady = true;
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
