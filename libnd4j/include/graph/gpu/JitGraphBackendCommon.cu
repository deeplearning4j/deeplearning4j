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
  int fusible = 0;
  for (int i = start; i <= end; i++) {
    auto cat = getOpCategoryFromName(slots[i].ident.opName);
    if (isNvrtcJittable(cat)) {
      fusible++;
    }
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

  // Build kernel arguments
  std::vector<void*> kernelArgs;
  kernelArgs.reserve(compiled->argMap.size() + 1);

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

    kernelArgs.push_back(arr->specialBuffer());

    if (am.isOutput && nElements == 0) {
      nElements = arr->lengthOf();
    }
  }

  int nElem32 = static_cast<int>(nElements);
  kernelArgs.push_back(&nElem32);

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
      kernelArgs.data(),
      static_cast<int>(kernelArgs.size()));

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

