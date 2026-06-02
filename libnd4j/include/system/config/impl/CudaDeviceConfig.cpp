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

#include <system/config/CudaDeviceConfig.h>
#include <system/config/EnvHelper.h>
#include <helpers/logger.h>

#ifdef SD_CUDA
#include <system/CudaLimitType.h>
#include <legacy/cuda/Environment_cuda.h>
#include <legacy/cuda/Environment_CudaConfig_cuda.h>
#endif

namespace sd {
namespace config {

CudaDeviceConfig::CudaDeviceConfig() {
  // Defaults set via member initializers.
}

void CudaDeviceConfig::setCudaCurrentDevice(int device) {
#ifdef SD_CUDA
  if (sd::Environment_setCudaCurrentDevice_cuda(device, _cudaDeviceCount.load())) {
    _cudaCurrentDevice.store(device);
  }
#endif
}

void CudaDeviceConfig::setCudaMemoryPoolSize(int sizeInMB) {
  if (sizeInMB >= 0) _cudaMemoryPoolSize.store(sizeInMB);
}

void CudaDeviceConfig::setCudaPrefetchSize(int sizeInMB) {
  if (sizeInMB >= 0) _cudaPrefetchSize.store(sizeInMB);
}

void CudaDeviceConfig::setCudaPinnedHostLimit(int64_t limitInMB) {
  if (limitInMB >= 0) _cudaPinnedHostLimit.store(limitInMB);
}

void CudaDeviceConfig::setCudaSoftLimitPercent(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  _cudaSoftLimitPercent.store(percent);
}

void CudaDeviceConfig::setCudaCachingAllocatorLimit(int limitInMB) {
  if (limitInMB >= 0) _cudaCachingAllocatorLimit.store(limitInMB);
}

void CudaDeviceConfig::setCudaMaxBlocks(int blocks) {
  if (blocks > 0) _cudaMaxBlocks.store(blocks);
}

void CudaDeviceConfig::setCudaMaxThreadsPerBlock(int threads) {
  if (threads > 0) _cudaMaxThreadsPerBlock.store(threads);
}

void CudaDeviceConfig::setCudaStreamLimit(int limit) {
  if (limit > 0) _cudaStreamLimit.store(limit);
}

void CudaDeviceConfig::setCudaEventLimit(int limit) {
  if (limit > 0) _cudaEventLimit.store(limit);
}

bool CudaDeviceConfig::setCudaDeviceLimit(int limitType, size_t value) {
#ifdef SD_CUDA
  return sd::Environment_setCudaDeviceLimit_cuda(limitType, value);
#else
  return false;
#endif
}

void CudaDeviceConfig::setCudaStackSize(size_t size) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_STACK_SIZE, size)) _cudaStackSize.store(size);
#endif
}

void CudaDeviceConfig::setCudaMallocHeapSize(size_t size) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_MALLOC_HEAP_SIZE, size)) _cudaMallocHeapSize.store(size);
#endif
}

void CudaDeviceConfig::setCudaPrintfFifoSize(size_t size) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_PRINTF_FIFO_SIZE, size)) _cudaPrintfFifoSize.store(size);
#endif
}

void CudaDeviceConfig::setCudaDevRuntimeSyncDepth(size_t depth) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH, depth)) _cudaDevRuntimeSyncDepth.store(depth);
#endif
}

void CudaDeviceConfig::setCudaDevRuntimePendingLaunchCount(size_t count) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, count)) _cudaDevRuntimePendingLaunchCount.store(count);
#endif
}

void CudaDeviceConfig::setCudaMaxL2FetchGranularity(size_t size) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY, size)) _cudaMaxL2FetchGranularity.store(size);
#endif
}

void CudaDeviceConfig::setCudaPersistingL2CacheSize(size_t size) {
#ifdef SD_CUDA
  if (setCudaDeviceLimit(CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE, size)) _cudaPersistingL2CacheSize.store(size);
#endif
}

void CudaDeviceConfig::setCudaTensorCoreEnabled(bool enabled) {
#ifdef SD_CUDA
  _cudaTensorCoreEnabled.store(enabled);
  int deviceId = _cudaCurrentDevice.load();
  int deviceCount = _cudaDeviceCount.load();
  int computeMajor = (deviceId >= 0 && deviceId < deviceCount) ? _capabilities[deviceId].first() : 0;
  sd::Environment_setCudaTensorCoreEnabled_cuda(enabled, deviceId, deviceCount, computeMajor);
#endif
}

void CudaDeviceConfig::setCudaBlockingSync(int mode) {
#ifdef SD_CUDA
  if (mode >= 0 && mode <= 1) {
    _cudaBlockingSync.store(mode);
    sd::Environment_setCudaBlockingSync_cuda(mode);
  }
#endif
}

void CudaDeviceConfig::setCudaDeviceSchedule(int schedule) {
#ifdef SD_CUDA
  if (schedule >= 0 && schedule <= 3) {
    _cudaDeviceSchedule.store(schedule);
    sd::Environment_setCudaDeviceSchedule_cuda(schedule);
  }
#endif
}

void CudaDeviceConfig::initCudaDeviceLimits() {
#ifdef SD_CUDA
  {
    size_t stackSize = 0, mallocHeapSize = 0, printfFifoSize = 0;
    size_t devRuntimeSyncDepth = 0, devRuntimePendingLaunchCount = 0;
    size_t maxL2FetchGranularity = 0, persistingL2CacheSize = 0;
    sd::Environment_queryCudaDeviceLimits(stackSize, mallocHeapSize, printfFifoSize,
                                      devRuntimeSyncDepth, devRuntimePendingLaunchCount,
                                      maxL2FetchGranularity, persistingL2CacheSize);
    _cudaStackSize.store(stackSize);
    _cudaMallocHeapSize.store(mallocHeapSize);
    _cudaPrintfFifoSize.store(printfFifoSize);
    _cudaDevRuntimeSyncDepth.store(devRuntimeSyncDepth);
    _cudaDevRuntimePendingLaunchCount.store(devRuntimePendingLaunchCount);
    _cudaMaxL2FetchGranularity.store(maxL2FetchGranularity);
    _cudaPersistingL2CacheSize.store(persistingL2CacheSize);
  }

  // Load custom limits from environment variables
  {
    int64_t v = readInt64Env("SD_CUDA_STACK_SIZE", -1);
    if (v > 0) setCudaStackSize(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_MALLOC_HEAP_SIZE", -1);
    if (v > 0) setCudaMallocHeapSize(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_PRINTF_FIFO_SIZE", -1);
    if (v > 0) setCudaPrintfFifoSize(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_DEV_RUNTIME_SYNC_DEPTH", -1);
    if (v > 0) setCudaDevRuntimeSyncDepth(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_DEV_RUNTIME_PENDING_LAUNCH_COUNT", -1);
    if (v > 0) setCudaDevRuntimePendingLaunchCount(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_MAX_L2_FETCH_GRANULARITY", -1);
    if (v > 0) setCudaMaxL2FetchGranularity(static_cast<size_t>(v));
  }
  {
    int64_t v = readInt64Env("SD_CUDA_PERSISTING_L2_CACHE_SIZE", -1);
    if (v > 0) setCudaPersistingL2CacheSize(static_cast<size_t>(v));
  }
#endif
}

void CudaDeviceConfig::initFromEnvironment() {
#ifdef SD_CUDA
  // CUDA device init (query device count, capabilities, BLAS version)
  {
    int devCnt = 0;
    sd::Environment_initCuda(devCnt, _capabilities,
                         _blasMajorVersion, _blasMinorVersion, _blasPatchVersion);
    _cudaDeviceCount.store(devCnt);
  }

  // Device selection: handled by Nd4j.getEnvironment().setCudaCurrentDevice()
  // from Java after backend init. No env var reading here.

  // Memory settings
  {
    int v = readBoolEnvTriState("SD_CUDA_PINNED_MEMORY");
    if (v >= 0) _cudaMemoryPinned.store(v == 1);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_MANAGED_MEMORY");
    if (v >= 0) _cudaUseManagedMemory.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_MEMORY_POOL_SIZE", -1);
    if (v > 0) _cudaMemoryPoolSize.store(v);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_FORCE_P2P");
    if (v >= 0) _cudaForceP2P.store(v == 1);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_ALLOCATOR_ENABLED");
    if (v >= 0) _cudaAllocatorEnabled.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_MAX_BLOCKS", -1);
    if (v > 0) _cudaMaxBlocks.store(v);
  }
  {
    int v = readIntEnv("SD_CUDA_MAX_THREADS_PER_BLOCK", -1);
    if (v > 0) _cudaMaxThreadsPerBlock.store(v);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_ASYNC_EXECUTION");
    if (v >= 0) _cudaAsyncExecution.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_STREAM_LIMIT", -1);
    if (v > 0) _cudaStreamLimit.store(v);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_USE_DEVICE_HOST");
    if (v >= 0) _cudaUseDeviceHost.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_EVENT_LIMIT", -1);
    if (v > 0) _cudaEventLimit.store(v);
  }
  {
    int v = readIntEnv("SD_CUDA_CACHING_ALLOCATOR_LIMIT", -1);
    if (v > 0) _cudaCachingAllocatorLimit.store(v);
  }
  {
    int64_t v = readInt64Env("SD_CUDA_PINNED_HOST_LIMIT", -1);
    if (v > 0) _cudaPinnedHostLimit.store(v);
  }
  {
    int v = readIntEnv("SD_CUDA_SOFT_LIMIT_PERCENT", -1);
    if (v >= 0 && v <= 100) _cudaSoftLimitPercent.store(v);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_USE_UNIFIED_MEMORY");
    if (v >= 0) _cudaUseUnifiedMemory.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_PREFETCH_SIZE", -1);
    if (v > 0) _cudaPrefetchSize.store(v);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_GRAPH_OPTIMIZATION");
    if (v >= 0) _cudaGraphOptimization.store(v == 1);
  }
  {
    int v = readBoolEnvTriState("SD_CUDA_TENSOR_CORE_ENABLED");
    if (v >= 0) _cudaTensorCoreEnabled.store(v == 1);
  }
  {
    int v = readIntEnv("SD_CUDA_BLOCKING_SYNC", -1);
    if (v >= 0 && v <= 1) _cudaBlockingSync.store(v);
  }
  {
    int v = readIntEnv("SD_CUDA_DEVICE_SCHEDULE", -1);
    if (v >= 0 && v <= 3) _cudaDeviceSchedule.store(v);
  }

  // Initialize CUDA device limits
  initCudaDeviceLimits();

  _cudaCurrentDevice.store(0);
#endif
}

}  // namespace config
}  // namespace sd
