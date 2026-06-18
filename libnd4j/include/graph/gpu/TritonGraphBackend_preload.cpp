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
// TritonGraphBackend batched module preload (task #4).
//
// After NativeDynamicShapePlan::platformPrecompileSegments() finishes
// compiling every segment, this file's preloadAllModules() walks the
// per-segment cache_ once, making sure every CompiledKernel has a live
// CUmodule loaded into GPU memory.  This avoids paying lazy-load latency
// during the first replay of each segment and gives us a single, easy-to-
// monitor checkpoint where total module residency is computed and compared
// against env.triton().moduleResidencyBudgetBytes().
//
// This file deliberately lives in its own translation unit (rather than
// being added to TritonGraphBackend_kernel.cu or TritonGraphBackend_cache.cpp)
// so it can evolve independently of the LRU eviction work in those files.
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <system/common.h>

#include <chrono>
#include <cstddef>
#include <mutex>

namespace sd {
namespace graph {

Status TritonGraphBackend::preloadAllModules(int deviceId) {
  // Honor the config flag. Disabled => preload is a no-op (lazy-load on first
  // execution remains the default behavior pre-task-#4).
  auto& env = sd::Environment::getInstance();
  if (!env.triton().batchPreloadModules()) {
    DSP_DIAG(COMPILE,
             "TritonGraphBackend::preloadAllModules: skipped (batchPreloadModules=false)");
    return Status::OK;
  }

  using Clock = std::chrono::steady_clock;
  const auto tStart = Clock::now();

  // Snapshot the cache under the cache mutex. We compute projected bytes and
  // gather pointers to kernels that need a reload before releasing the lock,
  // so we don't hold cacheMtx_ across cuModuleLoadDataEx (which can take ms).
  size_t totalKernels = 0;
  size_t alreadyLoadedKernels = 0;
  size_t alreadyLoadedBytes = 0;
  size_t needReloadKernels = 0;
  size_t needReloadBytesEstimate = 0;
  std::vector<CompiledKernel*> reloadList;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    for (auto& entry : cache_) {
      // Only count entries that were compiled for the device the caller
      // currently has active.  Cross-device modules will be preloaded by a
      // later preloadAllModules() call after their device is selected.
      if (entry.first.deviceId != deviceId) continue;

      auto& seg = entry.second;
      for (auto& k : seg.subKernels) {
        totalKernels++;
        if (k.gpuModule != nullptr) {
          alreadyLoadedKernels++;
          alreadyLoadedBytes += k.estimatedModuleBytes;
        } else {
          needReloadKernels++;
          needReloadBytesEstimate += k.estimatedModuleBytes;
          reloadList.push_back(&k);
        }
      }
    }
  }

  if (totalKernels == 0) {
    DSP_DIAG(COMPILE,
             "TritonGraphBackend::preloadAllModules: no cached kernels for device %d",
             deviceId);
    return Status::OK;
  }

  const size_t projectedBytes = alreadyLoadedBytes + needReloadBytesEstimate;
  const int64_t budget = env.triton().moduleResidencyBudgetBytes();

  DSP_DIAG(COMPILE,
           "TritonGraphBackend::preloadAllModules START device=%d totalKernels=%zu "
           "alreadyLoaded=%zu (%zu bytes) needReload=%zu (~%zu bytes) "
           "projected=%zu bytes budget=%lld bytes",
           deviceId,
           totalKernels, alreadyLoadedKernels, alreadyLoadedBytes,
           needReloadKernels, needReloadBytesEstimate,
           projectedBytes, static_cast<long long>(budget));

  // Loud warning BEFORE doing any work, so the user sees it even if a reload
  // crashes mid-flight.  budget == 0 means unlimited (no warning).
  if (budget > 0 && static_cast<int64_t>(projectedBytes) > budget) {
    sd_printf(
        "[TRITON_PRELOAD] warning: projected total module memory %lld bytes "
        "exceeds residency budget %lld bytes on device %d "
        "(alreadyLoaded=%lld, needReload=%lld, kernels=%lld). "
        "Preload will continue; LRU eviction may unload cold modules afterward.\n",
        static_cast<long long>(projectedBytes),
        static_cast<long long>(budget),
        deviceId,
        static_cast<long long>(alreadyLoadedBytes),
        static_cast<long long>(needReloadBytesEstimate),
        static_cast<long long>(totalKernels));
    DSP_DIAG(MEMORY,
             "TRITON_PRELOAD_OVER_BUDGET device=%d projected=%lld budget=%lld "
             "(alreadyLoaded=%lld needReload=%lld kernels=%lld)",
             deviceId,
             static_cast<long long>(projectedBytes),
             static_cast<long long>(budget),
             static_cast<long long>(alreadyLoadedBytes),
             static_cast<long long>(needReloadBytesEstimate),
             static_cast<long long>(totalKernels));
  }

  // Perform any reloads.  Today nothing is evicted at preload time, so this
  // loop is typically empty — the heavy lifting (cuModuleLoadDataEx) happened
  // during compileSegment().  Once LRU eviction (task #3) lands and starts
  // unloading modules between plan compile and plan execute, this loop will
  // pick them up via reloadModuleIfEvicted().
  //
  // The actual reload call is intentionally left to a follow-up integration
  // patch so that this file links cleanly while task #3's
  // reloadModuleIfEvicted() implementation is still in flight.  Until then,
  // any non-loaded kernel is reported via DSP_DIAG and counted as a reload
  // failure so the surrounding budget bookkeeping stays accurate.
  size_t reloadOk = 0;
  size_t reloadFail = 0;
  size_t reloadedBytes = 0;
  Status finalStatus = Status::OK;

  for (CompiledKernel* k : reloadList) {
    if (k == nullptr) continue;
    // Defensive re-check: another thread (e.g., a concurrent execute path)
    // may have raced ahead and reloaded the kernel between snapshot and now.
    if (k->gpuModule != nullptr) {
      reloadOk++;
      reloadedBytes += k->estimatedModuleBytes;
      continue;
    }

    Status reloadStatus = reloadModuleIfEvicted(k);
    if (reloadStatus == Status::OK && k->gpuModule != nullptr) {
      reloadOk++;
      reloadedBytes += k->estimatedModuleBytes;
      DSP_DIAG(COMPILE,
               "TritonGraphBackend::preloadAllModules: reloaded kernel '%s' "
               "(diskCacheHash=%s, %zu bytes)",
               k->kernelName.c_str(), k->diskCacheHash.c_str(),
               k->estimatedModuleBytes);
    } else {
      DSP_DIAG(COMPILE,
               "TritonGraphBackend::preloadAllModules: failed to reload kernel '%s' "
               "(diskCacheHash=%s); will fall back to lazy launch path",
               k->kernelName.empty() ? "<unnamed>" : k->kernelName.c_str(),
               k->diskCacheHash.c_str());
      reloadFail++;
      finalStatus = Status::KERNEL_FAILURE;
    }
  }

  const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - tStart)
                             .count();
  const size_t finalDeviceBytes = getTritonModuleMemory(deviceId);

  DSP_DIAG(COMPILE,
           "TritonGraphBackend::preloadAllModules DONE device=%d in %lld ms "
           "(alreadyLoaded=%zu, reloadedOk=%zu, reloadFailed=%zu, "
           "reloadedBytes=%zu, totalLoadedBytes=%zu, budget=%lld)",
           deviceId,
           static_cast<long long>(elapsedMs),
           alreadyLoadedKernels, reloadOk, reloadFail,
           reloadedBytes, finalDeviceBytes,
           static_cast<long long>(budget));

  // Always log a one-line summary at sd_printf level so it shows up in build
  // logs even when DSP_DIAG categories are disabled.  This mirrors the style
  // of TRITON_BUDGET reporting in NativeDynamicShapePlan_cuda.cu.
  sd_printf(
      "[TRITON_PRELOAD] device=%d kernels=%lld alreadyLoaded=%lld "
      "reloadedOk=%lld reloadedFail=%lld bytes=%lld budget=%lld elapsed=%lldms\n",
      deviceId,
      static_cast<long long>(totalKernels),
      static_cast<long long>(alreadyLoadedKernels),
      static_cast<long long>(reloadOk),
      static_cast<long long>(reloadFail),
      static_cast<long long>(finalDeviceBytes),
      static_cast<long long>(budget),
      static_cast<long long>(elapsedMs));

  return finalStatus;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
