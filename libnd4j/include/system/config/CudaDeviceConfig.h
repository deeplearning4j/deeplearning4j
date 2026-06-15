/* ******************************************************************************
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

#ifndef LIBND4J_CUDA_DEVICE_CONFIG_H
#define LIBND4J_CUDA_DEVICE_CONFIG_H

#include <system/common.h>
#include <types/pair.h>

#include <atomic>
#include <vector>

namespace sd {
namespace config {

/**
 * CUDA device configuration: device selection, memory pinning, pool sizes,
 * P2P, allocator, stream/event limits, graph optimization, tensor cores,
 * device scheduling, and CUDA device limits (stack, heap, printf, L2, etc.).
 */
class SD_LIB_EXPORT CudaDeviceConfig {
 private:
  std::atomic<int> _cudaDeviceCount{0};
  std::atomic<int> _cudaCurrentDevice{0};
  std::atomic<bool> _cudaMemoryPinned{false};
  std::atomic<bool> _cudaUseManagedMemory{false};
  std::atomic<int> _cudaMemoryPoolSize{0};
  std::atomic<bool> _cudaForceP2P{false};
  std::atomic<bool> _cudaAllocatorEnabled{true};
  std::atomic<int> _cudaMaxBlocks{0};
  std::atomic<int> _cudaMaxThreadsPerBlock{0};
  std::atomic<bool> _cudaAsyncExecution{true};
  std::atomic<int> _cudaStreamLimit{4};
  std::atomic<bool> _cudaUseDeviceHost{false};
  std::atomic<int> _cudaEventLimit{4};
  std::atomic<int> _cudaCachingAllocatorLimit{0};
  std::atomic<int64_t> _cudaPinnedHostLimit{8LL * 1024};
  std::atomic<int> _cudaSoftLimitPercent{0};  // proactive failover threshold (0=disabled, 1-100)
  std::atomic<bool> _cudaUseUnifiedMemory{false};
  std::atomic<int> _cudaPrefetchSize{0};
  std::atomic<bool> _cudaGraphOptimization{false};
  std::atomic<bool> _cudaTensorCoreEnabled{true};
  std::atomic<int> _cudaBlockingSync{0};
  std::atomic<int> _cudaDeviceSchedule{0};

  // CUDA Device Limits
  std::atomic<size_t> _cudaStackSize{0};
  std::atomic<size_t> _cudaMallocHeapSize{0};
  std::atomic<size_t> _cudaPrintfFifoSize{0};
  std::atomic<size_t> _cudaDevRuntimeSyncDepth{0};
  std::atomic<size_t> _cudaDevRuntimePendingLaunchCount{0};
  std::atomic<size_t> _cudaMaxL2FetchGranularity{0};
  std::atomic<size_t> _cudaPersistingL2CacheSize{0};

  // Device capabilities
  std::vector<Pair> _capabilities;

  // BLAS version tracking (populated during CUDA init)
  int _blasMajorVersion{0};
  int _blasMinorVersion{0};
  int _blasPatchVersion{0};

 public:
  CudaDeviceConfig();

  // --- Device management ---
  int cudaDeviceCount() { return _cudaDeviceCount.load(); }
  void setCudaDeviceCount(int count) { _cudaDeviceCount.store(count); }
  int cudaCurrentDevice() { return _cudaCurrentDevice.load(); }
  void setCudaCurrentDevice(int device);

  // --- Memory ---
  bool cudaMemoryPinned() { return _cudaMemoryPinned.load(); }
  void setCudaMemoryPinned(bool v) { _cudaMemoryPinned.store(v); }
  bool cudaUseManagedMemory() { return _cudaUseManagedMemory.load(); }
  void setCudaUseManagedMemory(bool v) { _cudaUseManagedMemory.store(v); }
  int cudaMemoryPoolSize() { return _cudaMemoryPoolSize.load(); }
  void setCudaMemoryPoolSize(int sizeInMB);
  bool cudaForceP2P() { return _cudaForceP2P.load(); }
  void setCudaForceP2P(bool v) { _cudaForceP2P.store(v); }
  bool cudaAllocatorEnabled() { return _cudaAllocatorEnabled.load(); }
  void setCudaAllocatorEnabled(bool v) { _cudaAllocatorEnabled.store(v); }
  bool cudaUseUnifiedMemory() { return _cudaUseUnifiedMemory.load(); }
  void setCudaUseUnifiedMemory(bool v) { _cudaUseUnifiedMemory.store(v); }
  int cudaPrefetchSize() { return _cudaPrefetchSize.load(); }
  void setCudaPrefetchSize(int sizeInMB);
  int64_t cudaPinnedHostLimit() { return _cudaPinnedHostLimit.load(); }
  void setCudaPinnedHostLimit(int64_t limitInMB);
  int cudaSoftLimitPercent() { return _cudaSoftLimitPercent.load(); }
  void setCudaSoftLimitPercent(int percent);
  int cudaCachingAllocatorLimit() { return _cudaCachingAllocatorLimit.load(); }
  void setCudaCachingAllocatorLimit(int limitInMB);

  // --- Execution ---
  int cudaMaxBlocks() { return _cudaMaxBlocks.load(); }
  void setCudaMaxBlocks(int blocks);
  int cudaMaxThreadsPerBlock() { return _cudaMaxThreadsPerBlock.load(); }
  void setCudaMaxThreadsPerBlock(int threads);
  bool cudaAsyncExecution() { return _cudaAsyncExecution.load(); }
  void setCudaAsyncExecution(bool v) { _cudaAsyncExecution.store(v); }
  int cudaStreamLimit() { return _cudaStreamLimit.load(); }
  void setCudaStreamLimit(int limit);
  bool cudaUseDeviceHost() { return _cudaUseDeviceHost.load(); }
  void setCudaUseDeviceHost(bool v) { _cudaUseDeviceHost.store(v); }
  int cudaEventLimit() { return _cudaEventLimit.load(); }
  void setCudaEventLimit(int limit);
  bool cudaGraphOptimization() { return _cudaGraphOptimization.load(); }
  void setCudaGraphOptimization(bool v) { _cudaGraphOptimization.store(v); }
  bool cudaTensorCoreEnabled() { return _cudaTensorCoreEnabled.load(); }
  void setCudaTensorCoreEnabled(bool enabled);
  int cudaBlockingSync() { return _cudaBlockingSync.load(); }
  void setCudaBlockingSync(int mode);
  int cudaDeviceSchedule() { return _cudaDeviceSchedule.load(); }
  void setCudaDeviceSchedule(int schedule);

  // --- CUDA Device Limits ---
  size_t cudaStackSize() { return _cudaStackSize.load(); }
  void setCudaStackSize(size_t size);
  size_t cudaMallocHeapSize() { return _cudaMallocHeapSize.load(); }
  void setCudaMallocHeapSize(size_t size);
  size_t cudaPrintfFifoSize() { return _cudaPrintfFifoSize.load(); }
  void setCudaPrintfFifoSize(size_t size);
  size_t cudaDevRuntimeSyncDepth() { return _cudaDevRuntimeSyncDepth.load(); }
  void setCudaDevRuntimeSyncDepth(size_t depth);
  size_t cudaDevRuntimePendingLaunchCount() { return _cudaDevRuntimePendingLaunchCount.load(); }
  void setCudaDevRuntimePendingLaunchCount(size_t count);
  size_t cudaMaxL2FetchGranularity() { return _cudaMaxL2FetchGranularity.load(); }
  void setCudaMaxL2FetchGranularity(size_t size);
  size_t cudaPersistingL2CacheSize() { return _cudaPersistingL2CacheSize.load(); }
  void setCudaPersistingL2CacheSize(size_t size);
  bool setCudaDeviceLimit(int limitType, size_t value);

  // --- Capabilities & BLAS version ---
  std::vector<Pair>& capabilities() { return _capabilities; }
  int blasMajorVersion() const { return _blasMajorVersion; }
  int blasMinorVersion() const { return _blasMinorVersion; }
  int blasPatchVersion() const { return _blasPatchVersion; }
  void setBlasMajorVersion(int v) { _blasMajorVersion = v; }
  void setBlasMinorVersion(int v) { _blasMinorVersion = v; }
  void setBlasPatchVersion(int v) { _blasPatchVersion = v; }

  /**
   * Initialize from environment variables + CUDA runtime queries.
   */
  void initFromEnvironment();
  void initCudaDeviceLimits();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_CUDA_DEVICE_CONFIG_H
