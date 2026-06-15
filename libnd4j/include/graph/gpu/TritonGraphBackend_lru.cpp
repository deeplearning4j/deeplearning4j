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
// ModuleResidencyCache implementation.
//
// Tracks loaded Triton CUmodules per device and evicts least-recently-used
// modules via cuModuleUnload when the per-device byte budget configured in
// env.triton().moduleResidencyBudgetBytes() is exceeded.  Evicted kernels
// reload from the (atomic) disk cache on next launch.
//
// This file deliberately stays a .cpp (no NVCC required) — all CUDA driver
// calls go through TritonTargetDispatch::loadModule / unloadModule, which is
// already compiled as plain C++.
//

#include <config.h>

#if HAVE_TRITON

#include <graph/DspDiagnostics.h>
#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <limits>
#include <mutex>

namespace sd {
namespace graph {

// ─── LRU touch ──────────────────────────────────────────────────────────────
// Lock-free fast path: bump lruTickCounter_ atomically and store into the
// CompiledKernel.  A torn read here is harmless — eviction picks the lowest
// tick, not the exact lowest, and a kernel that races between update and
// eviction is still a recently-used kernel.
void TritonGraphBackend::touchModule(CompiledKernel* k) {
  if (k == nullptr) return;
  k->lruTick = lruTickCounter_.fetch_add(1, std::memory_order_relaxed);
}

// ─── Register a freshly-loaded kernel ───────────────────────────────────────
// Adds the kernel to the per-device tracking list and triggers an eviction
// sweep if the device is now over budget.  Skips registration if the
// residency budget is disabled (budget == 0) — there is no point paying the
// per-device list overhead when eviction will never run.
void TritonGraphBackend::registerLoadedKernel(CompiledKernel* k, int deviceId) {
  if (k == nullptr) return;
  if (deviceId < 0 || deviceId >= kMaxTritonDevices) return;

  const int64_t budget =
      sd::Environment::getInstance().triton().moduleResidencyBudgetBytes();
  const int64_t warnBytes =
      sd::Environment::getInstance().triton().moduleResidencyWarnBytes();

  // Always seed lruTick so reload bookkeeping works, regardless of budget.
  k->lruTick = lruTickCounter_.fetch_add(1, std::memory_order_relaxed);
  k->loadedDeviceId = deviceId;

  // Per-device tracking list is only needed when eviction is enabled.
  if (budget > 0) {
    {
      std::lock_guard<std::mutex> lock(loadedKernelsMtx_);
      auto& list = loadedKernels_[deviceId];
      // Avoid double-insert (e.g., reload after eviction on the same device).
      if (std::find(list.begin(), list.end(), k) == list.end()) {
        list.push_back(k);
      }
    }

    evictIfOverBudget(deviceId, k);
  }

  // ── Optional warn threshold ──
  // Warning fires independently of the residency budget — getTritonModuleMemory
  // reads the atomic counter that recordModuleAlloc updates on every module
  // load, so it is meaningful even when budget == 0 (eviction disabled).  The
  // flag is latched once per device so repeated loads do not spam stdout;
  // tests that need to re-trigger it should call clearResidencyWarned().
  if (warnBytes > 0) {
    const size_t devBytes = getTritonModuleMemory(deviceId);
    if (static_cast<int64_t>(devBytes) > warnBytes) {
      bool expected = false;
      if (residencyWarned_[deviceId].compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        sd::Environment::getInstance().triton().incrementModuleResidencyWarnFireCount();
        sd_printf("TritonGraphBackend: module residency on device %d (%zu bytes) "
                  "exceeds warn threshold (%lld bytes). Consider raising "
                  "ND4J_TRITON_MODULE_RESIDENCY_BUDGET_BYTES or the warn ceiling.\n",
                  deviceId, devBytes, static_cast<long long>(warnBytes));
        DSP_DIAG(MEMORY,
                 "TritonGraphBackend: residency warn threshold crossed on device %d "
                 "(used=%zu warn=%lld budget=%lld)",
                 deviceId, devBytes, static_cast<long long>(warnBytes),
                 static_cast<long long>(budget));
      }
    }
  }
}

// ─── Unregister (called from invalidateCache paths) ─────────────────────────
// The caller still owns cuModuleUnload — this function only removes the
// kernel from the residency tracking list so a stale pointer is not consulted
// during a later eviction sweep.
void TritonGraphBackend::unregisterLoadedKernel(CompiledKernel* k) {
  if (k == nullptr) return;
  std::unique_lock<std::mutex> lock(loadedKernelsMtx_);
  for (int dev = 0; dev < kMaxTritonDevices; ++dev) {
    auto& list = loadedKernels_[dev];
    auto it = std::find(list.begin(), list.end(), k);
    if (it != list.end()) {
      list.erase(it);
    }
  }
}

// ─── LRU eviction sweep ─────────────────────────────────────────────────────
// While the per-device byte counter exceeds the budget, find the kernel with
// the lowest lruTick (oldest) on that device, unload its module, and
// decrement the counter.  Skips the supplied dontEvict pointer so a kernel
// that just got loaded does not immediately evict itself.
void TritonGraphBackend::evictIfOverBudget(int deviceId, CompiledKernel* dontEvict) {
  if (deviceId < 0 || deviceId >= kMaxTritonDevices) return;
  const int64_t budget =
      sd::Environment::getInstance().triton().moduleResidencyBudgetBytes();
  if (budget <= 0) return;

  std::unique_lock<std::mutex> lock(loadedKernelsMtx_);

  int evicted = 0;
  size_t bytesFreed = 0;
  while (true) {
    const size_t currentBytes = getTritonModuleMemory(deviceId);
    if (static_cast<int64_t>(currentBytes) <= budget) break;

    auto& list = loadedKernels_[deviceId];
    if (list.empty()) break;

    // Pick the kernel with the lowest lruTick that still has a loaded module.
    CompiledKernel* victim = nullptr;
    uint64_t victimTick = std::numeric_limits<uint64_t>::max();
    size_t victimIdx = 0;
    for (size_t i = 0; i < list.size(); ++i) {
      CompiledKernel* k = list[i];
      if (k == nullptr) continue;
      if (k == dontEvict) continue;
      if (k->gpuModule == nullptr) continue;
      if (k->lruTick < victimTick) {
        victimTick = k->lruTick;
        victim = k;
        victimIdx = i;
      }
    }
    if (victim == nullptr) break;  // nothing evictable left

    const size_t victimBytes = victim->estimatedModuleBytes;
    void* victimModule = victim->gpuModule;

    // Null the module *before* releasing the lock so a concurrent launch that
    // wins reloadModuleIfEvicted's lookup observes the eviction and reloads.
    victim->gpuModule = nullptr;
    victim->kernelFunction = nullptr;
    victim->loadedDeviceId = -1;
    list.erase(list.begin() + victimIdx);
    recordModuleFree(deviceId, victimBytes);

    // Drop the lock briefly so the actual cuModuleUnload (which serializes on
    // the CUDA driver loadModuleMtx inside TritonTargetDispatch) does not
    // hold our LRU mutex.  Re-acquire after.
    lock.unlock();
    TritonTargetDispatch::unloadModule(victimModule);
    lock.lock();

    ++evicted;
    bytesFreed += victimBytes;
  }

  if (evicted > 0) {
    DSP_DIAG(MEMORY,
             "TritonGraphBackend: ModuleResidencyCache evicted %d module(s) (%zu bytes) "
             "from device %d to satisfy budget (%lld bytes); now %zu bytes",
             evicted, bytesFreed, deviceId, static_cast<long long>(budget),
             getTritonModuleMemory(deviceId));
  }
}

// ─── Reload an evicted kernel from the disk cache ───────────────────────────
// Called by the launch path when gpuModule == nullptr but diskCacheHash is
// set.  Loads the cached PTX, calls cuModuleLoadDataEx + cuModuleGetFunction,
// re-records the module bytes, and re-registers with the residency cache.
Status TritonGraphBackend::reloadModuleIfEvicted(CompiledKernel* k) {
  if (k == nullptr) return Status::KERNEL_FAILURE;
  if (k->gpuModule != nullptr) return Status::OK;  // not evicted

  if (k->diskCacheHash.empty() || k->kernelName.empty()) {
    sd_printf("TritonGraphBackend::reloadModuleIfEvicted: cannot reload kernel "
              "[%d-%d] — diskCacheHash or kernelName is empty (was the kernel "
              "loaded with residency tracking enabled?).\n",
              k->startSlot_, k->endSlot_);
    return Status::KERNEL_FAILURE;
  }

  int currentDevice = 0;
#ifdef SD_CUDA
  cudaGetDevice(&currentDevice);
#endif

  TritonCompiledBinary binary = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", 0, 0};
  if (!loadBinaryFromDiskCacheByHash(k->diskCacheHash, k->kernelName, binary)) {
    sd_printf("TritonGraphBackend::reloadModuleIfEvicted: disk cache MISS for "
              "evicted kernel [%d-%d] hash=%s — cannot recover.  Disk cache "
              "may have been wiped or the cache path is misconfigured.\n",
              k->startSlot_, k->endSlot_, k->diskCacheHash.c_str());
    return Status::KERNEL_FAILURE;
  }

  void* gpuModule = TritonTargetDispatch::loadModule(binary);
  if (gpuModule == nullptr) {
    sd_printf("TritonGraphBackend::reloadModuleIfEvicted: cuModuleLoadDataEx "
              "failed for evicted kernel [%d-%d] hash=%s\n",
              k->startSlot_, k->endSlot_, k->diskCacheHash.c_str());
    delete[] static_cast<char*>(binary.data);
    return Status::KERNEL_FAILURE;
  }

  void* kernelFunc = TritonTargetDispatch::getKernelFunction(gpuModule, k->kernelName);
  if (kernelFunc == nullptr) {
    sd_printf("TritonGraphBackend::reloadModuleIfEvicted: cuModuleGetFunction('%s') "
              "failed after reload for [%d-%d] hash=%s\n",
              k->kernelName.c_str(), k->startSlot_, k->endSlot_,
              k->diskCacheHash.c_str());
    TritonTargetDispatch::unloadModule(gpuModule);
    delete[] static_cast<char*>(binary.data);
    return Status::KERNEL_FAILURE;
  }

  // Successful reload — patch the kernel and account for the new bytes.
  k->gpuModule = gpuModule;
  k->kernelFunction = kernelFunc;
  k->estimatedModuleBytes = binary.size;
  recordModuleAlloc(currentDevice, binary.size);
  registerLoadedKernel(k, currentDevice);

  delete[] static_cast<char*>(binary.data);

  DSP_DIAG(MEMORY,
           "TritonGraphBackend: reloaded evicted kernel [%d-%d] hash=%s on device %d "
           "(%zu bytes; device now %zu bytes)",
           k->startSlot_, k->endSlot_, k->diskCacheHash.c_str(),
           currentDevice, k->estimatedModuleBytes, getTritonModuleMemory(currentDevice));
  return Status::OK;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
