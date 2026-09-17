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

//
// Shared execution / cache logic for JIT graph backends (NVRTC, PTX).
//


#include <graph/gpu/JitGraphBackendCommon.h>
#include <graph/DspDiagnostics.h>
#include <execution/LaunchContext.h>
#include <system/common.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace sd {
namespace graph {

bool jitCanFuseSegment(NativeSlot* slots, int start, int end) {
  // Any op that is not NVRTC-jittable falls through to an identity pass-through
  // in generateCudaSource() / generatePtx(). View-producing and data-movement ops
  // (reshape, permute, squeeze, expand_dims, slice, strided_slice, gather, concat,
  // etc.) are in SHAPE_MANIPULATION / DATA_MOVEMENT categories. In a flat 1D
  // element-wise kernel every thread reads position [idx] from inputs and writes
  // to outputs[idx] — there is no way to express a reshape (which remaps indices),
  // a permute (which reorders axes), or any gather/scatter. Fusing a segment that
  // contains such ops silently produces wrong data: the non-jittable slots emit
  // the identity of their input instead of their actual computation.
  //
  // Rule: if ANY slot in the range is not NVRTC-jittable, reject the whole segment.
  // The only exception is IDENTITY/CAST (already in isNvrtcJittable) and UNSUPPORTED
  // ops that are already gated by isNvrtcJittable returning false here.
  int fusible = 0;
  for (int i = start; i <= end; i++) {
    auto cat = getOpCategoryFromName(slots[i].ident.opName);
    if (!isNvrtcJittable(cat)) {
      // Non-jittable op present — the generated kernel would produce wrong results
      // (identity pass-through instead of the actual op). Reject the entire segment.
      DSP_DIAG(JIT,
               "jitCanFuseSegment: segment [%d-%d] contains non-jittable op '%s' "
               "(category=%d) at slot %d — segment ineligible for JIT fusion",
               start, end, slots[i].ident.opName.c_str(), static_cast<int>(cat), i);
      return false;
    }
    // Ternary (where/select) with an externally-bound condition can never be
    // lowered by a flat FLOAT32 elementwise kernel: external inputs are bound
    // as `const float*` kernel parameters, while a where-condition is a BOOL
    // mask and is frequently broadcast (e.g. mask [1,4,1] vs x [1,4,8]) —
    // neither dtype nor shape mapping is expressible in the per-element
    // [idx] addressing every JIT generator emits. Admitting such a slot
    // guarantees the segment compiles wrong math or fails the concrete
    // lowering contract at seal time. Same structural-honesty pattern as the
    // NNAPI gather rule: reject at admission so buildSegments splits the range
    // and the ternary op runs native (gap-op path, sanctioned for JIT modes).
    // Slot-produced conditions are already float-encoded 0/1 SSA values and
    // remain fusible.
    if (cat == TritonOpCategory::TERNARY &&
        slots[i].wiring.numInputs > 0 &&
        slots[i].wiring.inputSourceIndices != nullptr) {
      const int condSrc = slots[i].wiring.inputSourceIndices[0];
      if (condSrc < 0) {
        DSP_DIAG(JIT,
                 "jitCanFuseSegment: ternary op '%s' at slot %d has an "
                 "externally-bound condition (ext[%d]) — a BOOL/broadcast "
                 "condition binding is not expressible in a flat FLOAT32 "
                 "[idx] kernel; segment must not fuse (ternary runs native)",
                 slots[i].ident.opName.c_str(), i, -(condSrc + 1));
        return false;
      }
    }
    fusible++;
  }
  return fusible >= JIT_MIN_FUSIBLE_OPS;
}

bool jitValidateFloatTensorBindings(
    NativeSlot* slots, int start, int end,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    std::string& reason) {
  reason.clear();
  if (!slots || start < 0 || end < start) {
    reason = "FLOAT32-only JIT requires concrete segment wiring";
    return false;
  }

  auto check = [&](int slot, const char* role, int binding, bool external) {
    NDArray* array = nullptr;
    const int index = external ? -(binding + 1) : binding;
    if (external) {
      if (externalInputs && index >= 0 && index < numExternalInputs)
        array = externalInputs[index];
    } else if (outputSlots && index >= 0 && index < totalOutputSlots) {
      array = outputSlots[index];
    }
    if (array && array->dataType() == DataType::FLOAT32) return true;
    reason = "FLOAT32-only JIT tensor contract rejected: slot=" + std::to_string(slot) +
        ", " + role + " binding=" + std::to_string(binding) +
        (external ? " (external)" : " (output slot)") +
        ", dtype=" + (array ? std::to_string(static_cast<int>(array->dataType()))
                             : std::string("missing")) + "; required FLOAT32";
    return false;
  };

  for (int si = start; si <= end; ++si) {
    const auto& wiring = slots[si].wiring;
    if (wiring.numInputs < 0 || (wiring.numInputs > 0 && !wiring.inputSourceIndices) ||
        wiring.numOutputs <= 0 || !wiring.outputSlotIndices) {
      reason = "FLOAT32-only JIT requires concrete tensor bindings at slot=" + std::to_string(si);
      return false;
    }
    for (int i = 0; i < wiring.numInputs; ++i) {
      const int binding = wiring.inputSourceIndices[i];
      if (!check(si, "input", binding, binding < 0)) return false;
    }
    // Intermediate outputs matter too: a cast/comparison cannot silently become
    // FLOAT32 SSA simply because only the final output is a kernel parameter.
    for (int o = 0; o < wiring.numOutputs; ++o) {
      if (!check(si, "output", wiring.outputSlotIndices[o], false)) return false;
    }
  }
  return true;
}

Status jitExecuteSegment(
    const JitSegmentCacheKey& key,
    std::unordered_map<JitSegmentCacheKey, JitCompiledKernel, JitSegmentCacheHash>& cache,
    std::mutex& cacheMtx,
    const char* backendName,
    NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  auto failSegment = [&](const std::string& reason) {
    const std::string message =
        reason + " [" + backendName + " JIT segment " +
        std::to_string(key.startSlot) + "-" +
        std::to_string(key.endSlot) + ", DSP status=KERNEL_FAILURE (50)]";
    auto* errorReference = LaunchContext::defaultContext()->errorReference();
    errorReference->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
    errorReference->setErrorMessage(message);
    return Status::KERNEL_FAILURE;
  };

  std::string dtypeReason;
  if (!jitValidateFloatTensorBindings(slots, key.startSlot, key.endSlot,
                                     externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots, dtypeReason)) {
    DSP_DIAG(EXECUTE, "%s: %s", backendName, dtypeReason.c_str());
    return failSegment(dtypeReason);
  }

  JitCompiledKernel* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx);
    auto it = cache.find(key);
    if (it == cache.end()) {
      DSP_DIAG(EXECUTE, "%s::executeSegment: no compiled kernel for segment [%d-%d]",
               backendName, key.startSlot, key.endSlot);
      return failSegment("compiled kernel is absent from the shape/device cache");
    }
    compiled = &it->second;
  }

  // Build kernel arguments.
  // cuLaunchKernel expects void** kernelParams where each element POINTS TO
  // the parameter value. For pointer params, we store the pointer values in
  // argValues[] and then argPtrs[i] = &argValues[i].
  std::vector<void*> argValues;
  argValues.reserve(compiled->argMap.size());

  LongType nElements = 0;

  for (auto& am : compiled->argMap) {
    NDArray* arr = nullptr;
    if (am.slotIndex < 0) {
      int extIdx = -(am.slotIndex + 1);
      if (externalInputs && extIdx < numExternalInputs) {
        arr = externalInputs[extIdx];
      }
    } else {
      if (outputSlots && am.slotIndex < totalOutputSlots) {
        arr = outputSlots[am.slotIndex];
      }
    }

    if (!arr) {
      DSP_DIAG(EXECUTE, "%s::executeSegment: null array for arg slot %d", backendName, am.slotIndex);
      return failSegment(
          "kernel argument resolved to a null array: slot=" +
          std::to_string(am.slotIndex) +
          (am.slotIndex < 0
               ? ", externalIndex=" + std::to_string(-(am.slotIndex + 1))
               : std::string()));
    }

    // Validate the cached artifact's actual pointer arguments as well as the
    // current wiring; a stale mapping must never reinterpret non-FLOAT storage.
    if (arr->dataType() != DataType::FLOAT32) {
      const std::string reason = "FLOAT32-only JIT cached argument rejected: binding=" +
          std::to_string(am.slotIndex) + ", dtype=" +
          std::to_string(static_cast<int>(arr->dataType()));
      DSP_DIAG(EXECUTE, "%s: %s", backendName, reason.c_str());
      return failSegment(reason);
    }
    argValues.push_back(arr->specialBuffer());

    if (am.isOutput && nElements == 0) {
      nElements = arr->lengthOf();
    }
  }

  int nElem32 = static_cast<int>(nElements);

  // Build the indirection array: each entry points to the corresponding value
  std::vector<void*> argPtrs;
  argPtrs.reserve(argValues.size() + 1);
  for (size_t i = 0; i < argValues.size(); i++) {
    argPtrs.push_back(&argValues[i]);
  }
  argPtrs.push_back(&nElem32);

  // Launch config
  unsigned int blockSize = 256;
  unsigned int gridSize = (static_cast<unsigned int>(nElements) + blockSize - 1) / blockSize;
  if (gridSize == 0) gridSize = 1;

  // Dereference stream pointer (NativeDynamicShapePlan passes void* to cudaStream_t)
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  bool ok = GpuKernelLauncher::launchKernel(
      compiled->kernelFunction,
      gridSize, 1, 1,
      blockSize, 1, 1,
      0, actualStream,
      argPtrs.data(),
      static_cast<int>(argPtrs.size()));

  if (!ok) {
    DSP_DIAG(EXECUTE, "%s::executeSegment: kernel launch failed for segment [%d-%d]",
             backendName, key.startSlot, key.endSlot);
    return failSegment(
        "GpuKernelLauncher::launchKernel failed: grid=" +
        std::to_string(gridSize) + ", block=" +
        std::to_string(blockSize) + ", args=" +
        std::to_string(argPtrs.size()) + ", elements=" +
        std::to_string(nElements));
  }

  return Status::OK;
}

void jitInvalidateCache(
    std::unordered_map<JitSegmentCacheKey, JitCompiledKernel, JitSegmentCacheHash>& cache,
    std::mutex& cacheMtx,
    std::vector<CompilationAuditEntry>& lastAudit) {
  std::lock_guard<std::mutex> lock(cacheMtx);
  for (auto& entry : cache) {
    if (entry.second.gpuModule) {
      GpuKernelLauncher::unloadModule(entry.second.gpuModule);
    }
  }
  cache.clear();
  lastAudit.clear();
}

}  // namespace graph
}  // namespace sd

