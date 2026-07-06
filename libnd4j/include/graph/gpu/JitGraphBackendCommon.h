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
// Shared execution / cache / audit logic for JIT graph backends (NVRTC, PTX).
//
// Both backends use identical executeSegment, invalidateCache, canFuseSegment,
// and getLastCompilationAudit implementations. This header + cpp centralizes
// that logic so neither backend has to duplicate it.
//

#ifndef LIBND4J_JIT_GRAPH_BACKEND_COMMON_H
#define LIBND4J_JIT_GRAPH_BACKEND_COMMON_H

#ifdef SD_CUDA

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/GpuKernelLauncher.h>
#include <graph/gpu/OpCategoryTable.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// ─── Shared types for JIT backends ──────────────────────────────────────────

struct JitCompiledKernel {
  void* gpuModule;
  void* kernelFunction;
  int numArgs;  // Only used by NVRTC (total kernel arguments including n)

  struct ArgMapping {
    int slotIndex;
    bool isOutput;
  };
  std::vector<ArgMapping> argMap;
  std::vector<CompilationAuditEntry> audit;

  JitCompiledKernel() : gpuModule(nullptr), kernelFunction(nullptr), numArgs(0) {}
};

// Per-device cache key for JIT backends (NVRTC, PTX).
// GPU driver module handles (CUmodule) are bound to the device context on
// which they were loaded.  Sharing a handle across CUDA devices produces
// CUDA error 700 on cuLaunchKernel.  This key adds a deviceId field so each
// device gets its own compiled module, mirroring TritonGraphBackend's
// per-device SegmentCacheKey.
struct JitSegmentCacheKey {
  int startSlot = 0;
  int endSlot = 0;
  LongType shapeKey = 0;
  int deviceId = -1;
  bool operator==(const JitSegmentCacheKey& o) const {
    return startSlot == o.startSlot && endSlot == o.endSlot &&
           shapeKey == o.shapeKey && deviceId == o.deviceId;
  }
};

struct JitSegmentCacheHash {
  size_t operator()(const JitSegmentCacheKey& k) const {
    size_t h = std::hash<int>()(k.startSlot);
    h ^= std::hash<int>()(k.endSlot) << 1;
    h ^= std::hash<LongType>()(k.shapeKey) << 2;
    h ^= std::hash<int>()(k.deviceId) << 3;
    return h;
  }
};

// Minimum ops to attempt fusion (shared constant)
constexpr int JIT_MIN_FUSIBLE_OPS = 2;

// ─── Shared functions ───────────────────────────────────────────────────────

// Check if a segment has enough fusible ops for JIT compilation.
bool jitCanFuseSegment(NativeSlot* slots, int start, int end);

// Execute a compiled JIT kernel for a segment.
Status jitExecuteSegment(
    const JitSegmentCacheKey& key,
    std::unordered_map<JitSegmentCacheKey, JitCompiledKernel, JitSegmentCacheHash>& cache,
    std::mutex& cacheMtx,
    const char* backendName,
    NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream);

// Invalidate all cached JIT kernels.
void jitInvalidateCache(
    std::unordered_map<JitSegmentCacheKey, JitCompiledKernel, JitSegmentCacheHash>& cache,
    std::mutex& cacheMtx,
    std::vector<CompilationAuditEntry>& lastAudit);

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_JIT_GRAPH_BACKEND_COMMON_H
