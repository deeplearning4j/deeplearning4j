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
#include <system/common.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
    fusible++;
  }
  return fusible >= JIT_MIN_FUSIBLE_OPS;
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

  JitCompiledKernel* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx);
    auto it = cache.find(key);
    if (it == cache.end()) {
      DSP_DIAG(EXECUTE, "%s::executeSegment: no compiled kernel for segment [%d-%d]",
               backendName, key.startSlot, key.endSlot);
      return Status::KERNEL_FAILURE;
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
      if (extIdx < numExternalInputs) {
        arr = externalInputs[extIdx];
      }
    } else {
      if (am.slotIndex < totalOutputSlots) {
        arr = outputSlots[am.slotIndex];
      }
    }

    if (!arr) {
      DSP_DIAG(EXECUTE, "%s::executeSegment: null array for arg slot %d", backendName, am.slotIndex);
      return Status::KERNEL_FAILURE;
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
    return Status::KERNEL_FAILURE;
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

