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
// Created by raver119 on 06.10.2017.
//
#include <system/Environment.h>

#include <helpers/BlasHelper.h>
#include <helpers/StringUtils.h>
#include <helpers/logger.h>
#include <memory/MemoryCounter.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

// Lifecycle tracker includes for enabling/disabling via Environment
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <legacy/cuda/Environment_cuda.h>

namespace sd {

Environment::Environment() {
 _tadThreshold.store(1);
 _elementThreshold.store(1024);
 _verbose.store(false);
 _debug.store(false);
 _profile.store(false);
 _precBoost.store(false);
 _leaks.store(false);
 _dataType.store(FLOAT32);
 _maxThreads = std::thread::hardware_concurrency();
 _maxMasterThreads = _maxThreads.load();
 deleteShapeInfo = deleteShapeInfo.load();
 _logNDArrayEvenuts.store(false);
#ifndef ANDROID
 const char *omp_threads = std::getenv("OMP_NUM_THREADS");
 if (omp_threads != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string omp(omp_threads);
     int val = std::stoi(omp);
     _maxThreads.store(val);
     _maxMasterThreads.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string omp(omp_threads);
   int val = std::stoi(omp);
   _maxThreads.store(val);
   _maxMasterThreads.store(val);
#endif
 }
#endif
 /**
  * Defines size of thread pool used for parallelism
  */
 const char *max_threads = std::getenv("SD_MAX_THREADS");
 if (max_threads != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string t(max_threads);
     int val = std::stoi(t);
     _maxThreads.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string t(max_threads);
   int val = std::stoi(t);
   _maxThreads.store(val);
#endif
 }

 /**
  * Defines max number of threads usable at once
  */
 const char *max_master_threads = std::getenv("SD_MASTER_THREADS");
 if (max_master_threads != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string t(max_master_threads);
     int val = std::stoi(t);
     _maxMasterThreads.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string t(max_master_threads);
   int val = std::stoi(t);
   _maxMasterThreads.store(val);
#endif
 }

 if (_maxMasterThreads.load() > _maxThreads.load()) {
   sd_printf("Warning! MAX_MASTER_THREADS > MAX_THREADS, tuning them down to match each other\n", "");
   _maxMasterThreads.store(_maxThreads.load());
 }

 /**
  * If this env var is defined - we'll disallow use of platform-specific helpers (mkldnn, cudnn, etc)
  */
 const char *forbid_helpers = std::getenv("SD_FORBID_HELPERS");
 if (forbid_helpers != nullptr) {
   _allowHelpers = false;
 }

 /**
  * This var defines max amount of host memory library can allocate
  */
 const char *max_primary_memory = std::getenv("SD_MAX_PRIMARY_BYTES");
 if (max_primary_memory != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string t(max_primary_memory);
     auto val = std::stol(t);
     _maxTotalPrimaryMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string t(max_primary_memory);
   auto val = std::stol(t);
   _maxTotalPrimaryMemory.store(val);
#endif
 }

 /**
  * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
  */
 const char *max_special_memory = std::getenv("SD_MAX_SPECIAL_BYTES");
 if (max_special_memory != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string t(max_special_memory);
     auto val = std::stol(t);
     _maxTotalSpecialMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string t(max_special_memory);
   auto val = std::stol(t);
   _maxTotalSpecialMemory.store(val);
#endif
 }

 /**
  * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
  */
 const char *max_device_memory = std::getenv("SD_MAX_DEVICE_BYTES");
 if (max_device_memory != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string t(max_device_memory);
     auto val = std::stol(t);
     _maxDeviceMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
#else
   std::string t(max_device_memory);
   auto val = std::stol(t);
   _maxDeviceMemory.store(val);
#endif
 }

 const char *blas_fallback = std::getenv("SD_BLAS_FALLBACK");
 if (blas_fallback != nullptr) {
   _blasFallback = true;
 }

 // NDArray lifecycle tracking configuration (only effective when SD_GCC_FUNCTRACE is defined)
#if defined(SD_GCC_FUNCTRACE)
 // Default is now FALSE to prevent backward-cpp crashes during early JVM initialization
 // Users can enable it with SD_LIFECYCLE_TRACKING=1 after JVM is fully initialized
 const char *lifecycle_tracking = std::getenv("SD_LIFECYCLE_TRACKING");
 if (lifecycle_tracking != nullptr) {
   std::string val(lifecycle_tracking);
   if (val == "0" || val == "false" || val == "FALSE") {
     _lifecycleTracking.store(false);
   } else if (val == "1" || val == "true" || val == "TRUE") {
     _lifecycleTracking.store(true);
   }
 }

 // Track views by default, but allow override
 const char *track_views = std::getenv("SD_TRACK_VIEWS");
 if (track_views != nullptr) {
   std::string val(track_views);
   if (val == "0" || val == "false" || val == "FALSE") {
     _trackViews.store(false);
   }
 }

 // Track deletions by default, but allow override
 const char *track_deletions = std::getenv("SD_TRACK_DELETIONS");
 if (track_deletions != nullptr) {
   std::string val(track_deletions);
   if (val == "0" || val == "false" || val == "FALSE") {
     _trackDeletions.store(false);
   }
 }

 // Stack depth for traces (default 32)
 const char *stack_depth = std::getenv("SD_STACK_DEPTH");
 if (stack_depth != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string val(stack_depth);
     int depth = std::stoi(val);
     if (depth > 0) {
       _stackDepth.store(depth);
     }
   } catch (std::invalid_argument &e) {
     // keep default
   } catch (std::out_of_range &e) {
     // keep default
   }
#else
   std::string val(stack_depth);
   int depth = std::stoi(val);
   if (depth > 0) {
     _stackDepth.store(depth);
   }
#endif
 }

 // Report interval in seconds (default 300 = 5 minutes)
 const char *report_interval = std::getenv("SD_REPORT_INTERVAL");
 if (report_interval != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string val(report_interval);
     int interval = std::stoi(val);
     if (interval > 0) {
       _reportInterval.store(interval);
     }
   } catch (std::invalid_argument &e) {
     // keep default
   } catch (std::out_of_range &e) {
     // keep default
   }
#else
   std::string val(report_interval);
   int interval = std::stoi(val);
   if (interval > 0) {
     _reportInterval.store(interval);
   }
#endif
 }

 // Max deletion history (default 10000)
 const char *max_deletion_history = std::getenv("SD_MAX_DELETION_HISTORY");
 if (max_deletion_history != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string val(max_deletion_history);
     size_t max_hist = std::stoul(val);
     _maxDeletionHistory.store(max_hist);
   } catch (std::invalid_argument &e) {
     // keep default
   } catch (std::out_of_range &e) {
     // keep default
   }
#else
   std::string val(max_deletion_history);
   size_t max_hist = std::stoul(val);
   _maxDeletionHistory.store(max_hist);
#endif
 }

 // Snapshot files - write periodic file snapshots (default off)
 const char *snapshot_files = std::getenv("SD_LIFECYCLE_SNAPSHOT_FILES");
 if (snapshot_files != nullptr) {
   std::string val(snapshot_files);
   if (val == "1" || val == "true" || val == "TRUE") {
     _snapshotFiles.store(true);
   }
 }

 // Track operations - enable operation name tracking (default off)
 const char *track_operations = std::getenv("SD_LIFECYCLE_TRACK_OPERATIONS");
 if (track_operations != nullptr) {
   std::string val(track_operations);
   if (val == "1" || val == "true" || val == "TRUE") {
     _trackOperations.store(true);
   }
 }
#endif

#ifdef SD_CUDA
  // Delegate all CUDA runtime calls to Environment_cuda.cu
  {
    int devCnt = 0;
    Environment_initCuda(devCnt, _capabilities,
                         _blasMajorVersion, _blasMinorVersion, _blasPatchVersion);
    _cudaDeviceCount.store(devCnt);
  }

  // Initialize CUDA environment settings (env-var parsing + device limits)
  try {
    initCudaEnvironment();
    initCudaDeviceLimits();
  } catch (...) {
    sd_printf("Environment::Environment: WARNING - CUDA environment initialization failed\n");
  }

  _cudaCurrentDevice.store(0);
#else
 // No CUDA environment to initialize
#endif
}

bool Environment::setCudaDeviceLimit(int limitType, size_t value) {
#ifdef SD_CUDA
 return Environment_setCudaDeviceLimit_cuda(limitType, value);
#else
 return false;
#endif
}

// Then update all the individual methods:
void Environment::setCudaStackSize(size_t size) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_STACK_SIZE, size)) {
   _cudaStackSize.store(size);
 }
#endif
}

void Environment::setCudaMallocHeapSize(size_t size) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_MALLOC_HEAP_SIZE, size)) {
   _cudaMallocHeapSize.store(size);
 }
#endif
}

void Environment::setCudaPrintfFifoSize(size_t size) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_PRINTF_FIFO_SIZE, size)) {
   _cudaPrintfFifoSize.store(size);
 }
#endif
}

void Environment::setCudaDevRuntimeSyncDepth(size_t depth) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH, depth)) {
   _cudaDevRuntimeSyncDepth.store(depth);
 }
#endif
}

void Environment::setCudaDevRuntimePendingLaunchCount(size_t count) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, count)) {
   _cudaDevRuntimePendingLaunchCount.store(count);
 }
#endif
}

void Environment::setCudaMaxL2FetchGranularity(size_t size) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY, size)) {
   _cudaMaxL2FetchGranularity.store(size);
 }
#endif
}

void Environment::setCudaPersistingL2CacheSize(size_t size) {
#ifdef SD_CUDA
 if (setCudaDeviceLimit(CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE, size)) {
   _cudaPersistingL2CacheSize.store(size);
 }
#endif
}


void Environment::initCudaDeviceLimits() {
 // Get the current values for all device limits to initialize our variables
#ifdef SD_CUDA
 {
   size_t stackSize = 0, mallocHeapSize = 0, printfFifoSize = 0;
   size_t devRuntimeSyncDepth = 0, devRuntimePendingLaunchCount = 0;
   size_t maxL2FetchGranularity = 0, persistingL2CacheSize = 0;
   Environment_queryCudaDeviceLimits(stackSize, mallocHeapSize, printfFifoSize,
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
 const char* stackSizeVar = std::getenv("SD_CUDA_STACK_SIZE");
 if (stackSizeVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(stackSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaStackSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(stackSizeVar);
   size_t size = std::stol(sizeStr);
   setCudaStackSize(size);
#endif
 }

 const char* heapSizeVar = std::getenv("SD_CUDA_MALLOC_HEAP_SIZE");
 if (heapSizeVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(heapSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaMallocHeapSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(heapSizeVar);
   size_t size = std::stol(sizeStr);
   setCudaMallocHeapSize(size);
#endif
 }

 const char* printfSizeVar = std::getenv("SD_CUDA_PRINTF_FIFO_SIZE");
 if (printfSizeVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(printfSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaPrintfFifoSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(printfSizeVar);
   size_t size = std::stol(sizeStr);
   setCudaPrintfFifoSize(size);
#endif
 }

 const char* syncDepthVar = std::getenv("SD_CUDA_DEV_RUNTIME_SYNC_DEPTH");
 if (syncDepthVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string depthStr(syncDepthVar);
     size_t depth = std::stol(depthStr);
     setCudaDevRuntimeSyncDepth(depth);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string depthStr(syncDepthVar);
   size_t depth = std::stol(depthStr);
   setCudaDevRuntimeSyncDepth(depth);
#endif
 }

 const char* pendingLaunchVar = std::getenv("SD_CUDA_DEV_RUNTIME_PENDING_LAUNCH_COUNT");
 if (pendingLaunchVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string countStr(pendingLaunchVar);
     size_t count = std::stol(countStr);
     setCudaDevRuntimePendingLaunchCount(count);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string countStr(pendingLaunchVar);
   size_t count = std::stol(countStr);
   setCudaDevRuntimePendingLaunchCount(count);
#endif
 }

 const char* l2FetchVar = std::getenv("SD_CUDA_MAX_L2_FETCH_GRANULARITY");
 if (l2FetchVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(l2FetchVar);
     size_t size = std::stol(sizeStr);
     setCudaMaxL2FetchGranularity(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(l2FetchVar);
   size_t size = std::stol(sizeStr);
   setCudaMaxL2FetchGranularity(size);
#endif
 }

 const char* l2CacheVar = std::getenv("SD_CUDA_PERSISTING_L2_CACHE_SIZE");
 if (l2CacheVar != nullptr) {
#if CUDART_VERSION >= 10000
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(l2CacheVar);
     size_t size = std::stol(sizeStr);
     setCudaPersistingL2CacheSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(l2CacheVar);
   size_t size = std::stol(sizeStr);
   setCudaPersistingL2CacheSize(size);
#endif
#else
   sd_printf("Warning: SD_CUDA_PERSISTING_L2_CACHE_SIZE requires CUDA 10.0 or newer\n", "");
#endif
 }

#endif
}

void Environment::initCudaEnvironment() {
#ifdef SD_CUDA
 // Initialize CUDA environment settings from environment variables
 const char* cudaDeviceVar = std::getenv("SD_CUDA_DEVICE");
 if (cudaDeviceVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string devStr(cudaDeviceVar);
     int device = std::stoi(devStr);
     if (device >= 0 && device < _cudaDeviceCount.load()) {
       _cudaCurrentDevice.store(device);
       Environment_cudaSetDeviceForInit(device);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string devStr(cudaDeviceVar);
   int device = std::stoi(devStr);
   if (device >= 0 && device < _cudaDeviceCount.load()) {
     _cudaCurrentDevice.store(device);
     Environment_cudaSetDeviceForInit(device);
   }
#endif
#endif
 }

 const char* cudaPinnedVar = std::getenv("SD_CUDA_PINNED_MEMORY");
#ifdef SD_CUDA
 if (cudaPinnedVar != nullptr) {
   std::string pinnedStr(cudaPinnedVar);
   if (pinnedStr == "true" || pinnedStr == "1" || pinnedStr == "yes") {
     _cudaMemoryPinned.store(true);
   } else {
     _cudaMemoryPinned.store(false);
   }
 }
#endif
 const char* cudaManagedVar = std::getenv("SD_CUDA_MANAGED_MEMORY");
#ifdef SD_CUDA

 if (cudaManagedVar != nullptr) {
   std::string managedStr(cudaManagedVar);
   if (managedStr == "true" || managedStr == "1" || managedStr == "yes") {
     _cudaUseManagedMemory.store(true);
   } else {
     _cudaUseManagedMemory.store(false);
   }
 }
#endif
 const char* cudaPoolSizeVar = std::getenv("SD_CUDA_MEMORY_POOL_SIZE");
#ifdef SD_CUDA
 if (cudaPoolSizeVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(cudaPoolSizeVar);
     int size = std::stoi(sizeStr);
     if (size > 0) {
       _cudaMemoryPoolSize.store(size);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(cudaPoolSizeVar);
   int size = std::stoi(sizeStr);
   if (size > 0) {
     _cudaMemoryPoolSize.store(size);
   }
#endif
 }
#endif

 const char* cudaForceP2PVar = std::getenv("SD_CUDA_FORCE_P2P");
#ifdef SD_CUDA
 if (cudaForceP2PVar != nullptr) {
   std::string p2pStr(cudaForceP2PVar);
   if (p2pStr == "true" || p2pStr == "1" || p2pStr == "yes") {
     _cudaForceP2P.store(true);
   } else {
     _cudaForceP2P.store(false);
   }
 }
#endif
 const char* cudaAllocatorVar = std::getenv("SD_CUDA_ALLOCATOR_ENABLED");
#ifdef SD_CUDA
 if (cudaAllocatorVar != nullptr) {
   std::string allocStr(cudaAllocatorVar);
   if (allocStr == "false" || allocStr == "0" || allocStr == "no") {
     _cudaAllocatorEnabled.store(false);
   } else {
     _cudaAllocatorEnabled.store(true);
   }
 }
#endif
 const char* cudaMaxBlocksVar = std::getenv("SD_CUDA_MAX_BLOCKS");
#ifdef SD_CUDA
 if (cudaMaxBlocksVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string blocksStr(cudaMaxBlocksVar);
     int blocks = std::stoi(blocksStr);
     if (blocks > 0) {
       _cudaMaxBlocks.store(blocks);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string blocksStr(cudaMaxBlocksVar);
   int blocks = std::stoi(blocksStr);
   if (blocks > 0) {
     _cudaMaxBlocks.store(blocks);
   }
#endif
 }
#endif

 const char* cudaMaxThreadsVar = std::getenv("SD_CUDA_MAX_THREADS_PER_BLOCK");
#ifdef SD_CUDA
 if (cudaMaxThreadsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string threadsStr(cudaMaxThreadsVar);
     int threads = std::stoi(threadsStr);
     if (threads > 0) {
       _cudaMaxThreadsPerBlock.store(threads);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string threadsStr(cudaMaxThreadsVar);
   int threads = std::stoi(threadsStr);
   if (threads > 0) {
     _cudaMaxThreadsPerBlock.store(threads);
   }
#endif
 }
#endif
 const char* cudaAsyncVar = std::getenv("SD_CUDA_ASYNC_EXECUTION");
#ifdef SD_CUDA
 if (cudaAsyncVar != nullptr) {
   std::string asyncStr(cudaAsyncVar);
   if (asyncStr == "false" || asyncStr == "0" || asyncStr == "no") {
     _cudaAsyncExecution.store(false);
   } else {
     _cudaAsyncExecution.store(true);
   }
 }
#endif
 const char* cudaStreamLimitVar = std::getenv("SD_CUDA_STREAM_LIMIT");
#ifdef SD_CUDA
 if (cudaStreamLimitVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string limitStr(cudaStreamLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaStreamLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string limitStr(cudaStreamLimitVar);
   int limit = std::stoi(limitStr);
   if (limit > 0) {
     _cudaStreamLimit.store(limit);
   }
#endif
 }
#endif
 const char* cudaDeviceHostVar = std::getenv("SD_CUDA_USE_DEVICE_HOST");
#ifdef SD_CUDA
 if (cudaDeviceHostVar != nullptr) {
   std::string deviceStr(cudaDeviceHostVar);
   if (deviceStr == "true" || deviceStr == "1" || deviceStr == "yes") {
     _cudaUseDeviceHost.store(true);
   } else {
     _cudaUseDeviceHost.store(false);
   }
 }
#endif
 const char* cudaEventLimitVar = std::getenv("SD_CUDA_EVENT_LIMIT");
#ifdef SD_CUDA
 if (cudaEventLimitVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string limitStr(cudaEventLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaEventLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string limitStr(cudaEventLimitVar);
   int limit = std::stoi(limitStr);
   if (limit > 0) {
     _cudaEventLimit.store(limit);
   }
#endif
 }
#endif
 const char* cudaCachingLimitVar = std::getenv("SD_CUDA_CACHING_ALLOCATOR_LIMIT");
#ifdef SD_CUDA
 if (cudaCachingLimitVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string limitStr(cudaCachingLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaCachingAllocatorLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string limitStr(cudaCachingLimitVar);
   int limit = std::stoi(limitStr);
   if (limit > 0) {
     _cudaCachingAllocatorLimit.store(limit);
   }
#endif
 }
#endif
 const char* cudaPinnedHostLimitVar = std::getenv("SD_CUDA_PINNED_HOST_LIMIT");
#ifdef SD_CUDA
 if (cudaPinnedHostLimitVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string limitStr(cudaPinnedHostLimitVar);
     int64_t limit = std::stoll(limitStr);
     if (limit > 0) {
       _cudaPinnedHostLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string limitStr(cudaPinnedHostLimitVar);
   int64_t limit = std::stoll(limitStr);
   if (limit > 0) {
     _cudaPinnedHostLimit.store(limit);
   }
#endif
 }
#endif
 const char* cudaUnifiedMemVar = std::getenv("SD_CUDA_USE_UNIFIED_MEMORY");
#ifdef SD_CUDA
 if (cudaUnifiedMemVar != nullptr) {
   std::string unifiedStr(cudaUnifiedMemVar);
   if (unifiedStr == "true" || unifiedStr == "1" || unifiedStr == "yes") {
     _cudaUseUnifiedMemory.store(true);
   } else {
     _cudaUseUnifiedMemory.store(false);
   }
 }
#endif
 const char* cudaPrefetchVar = std::getenv("SD_CUDA_PREFETCH_SIZE");
#ifdef SD_CUDA
 if (cudaPrefetchVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sizeStr(cudaPrefetchVar);
     int size = std::stoi(sizeStr);
     if (size > 0) {
       _cudaPrefetchSize.store(size);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sizeStr(cudaPrefetchVar);
   int size = std::stoi(sizeStr);
   if (size > 0) {
     _cudaPrefetchSize.store(size);
   }
#endif
 }
#endif
 const char* cudaGraphVar = std::getenv("SD_CUDA_GRAPH_OPTIMIZATION");
#ifdef SD_CUDA
 if (cudaGraphVar != nullptr) {
   std::string graphStr(cudaGraphVar);
   if (graphStr == "true" || graphStr == "1" || graphStr == "yes") {
     _cudaGraphOptimization.store(true);
   } else {
     _cudaGraphOptimization.store(false);
   }
 }
#endif
 const char* cudaTensorCoreVar = std::getenv("SD_CUDA_TENSOR_CORE_ENABLED");
#ifdef SD_CUDA
 if (cudaTensorCoreVar != nullptr) {
   std::string tensorStr(cudaTensorCoreVar);
   if (tensorStr == "false" || tensorStr == "0" || tensorStr == "no") {
     _cudaTensorCoreEnabled.store(false);
   } else {
     _cudaTensorCoreEnabled.store(true);
   }
 }
#endif
 const char* cudaBlockingSyncVar = std::getenv("SD_CUDA_BLOCKING_SYNC");
#ifdef SD_CUDA
 if (cudaBlockingSyncVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string syncStr(cudaBlockingSyncVar);
     int sync = std::stoi(syncStr);
     if (sync >= 0 && sync <= 1) {
       _cudaBlockingSync.store(sync);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string syncStr(cudaBlockingSyncVar);
   int sync = std::stoi(syncStr);
   if (sync >= 0 && sync <= 1) {
     _cudaBlockingSync.store(sync);
   }
#endif
 }
#endif
const char* cudaDeviceScheduleVar = std::getenv("SD_CUDA_DEVICE_SCHEDULE");
#ifdef SD_CUDA
 if (cudaDeviceScheduleVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string scheduleStr(cudaDeviceScheduleVar);
     int schedule = std::stoi(scheduleStr);
     if (schedule >= 0 && schedule <= 3) {
       _cudaDeviceSchedule.store(schedule);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string scheduleStr(cudaDeviceScheduleVar);
   int schedule = std::stoi(scheduleStr);
   if (schedule >= 0 && schedule <= 3) {
     _cudaDeviceSchedule.store(schedule);
   }
#endif
 }

 const char* tritonBuildThreadsVar = std::getenv("ND4J_TRITON_BUILD_THREADS");
 if (tritonBuildThreadsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string threadsStr(tritonBuildThreadsVar);
     int threads = std::stoi(threadsStr);
     setTritonBuildThreads(threads);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string threadsStr(tritonBuildThreadsVar);
   int threads = std::stoi(threadsStr);
   setTritonBuildThreads(threads);
#endif
 }

 const char* tritonCacheEnabledVar = std::getenv("ND4J_TRITON_CACHE_ENABLE");
 if (tritonCacheEnabledVar != nullptr) {
   std::string cacheStr(tritonCacheEnabledVar);
   if (cacheStr == "false" || cacheStr == "0" || cacheStr == "no") {
     setTritonCacheEnabled(false);
   } else {
     setTritonCacheEnabled(true);
   }
 }

 const char* tritonCooperativeLaunchVar = std::getenv("ND4J_TRITON_COOPERATIVE_LAUNCH");
 if (tritonCooperativeLaunchVar != nullptr) {
   std::string val(tritonCooperativeLaunchVar);
   setTritonCooperativeLaunch(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonCoopTargetBlocksVar = std::getenv("ND4J_TRITON_COOP_TARGET_BLOCKS");
 if (tritonCoopTargetBlocksVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string blocksStr(tritonCoopTargetBlocksVar);
     int blocks = std::stoi(blocksStr);
     setTritonCoopTargetBlocks(blocks);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string blocksStr(tritonCoopTargetBlocksVar);
   int blocks = std::stoi(blocksStr);
   setTritonCoopTargetBlocks(blocks);
#endif
 }

 const char* tritonMaxSubsegmentOpsVar = std::getenv("ND4J_TRITON_MAX_SUBSEGMENT_OPS");
 if (tritonMaxSubsegmentOpsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string opsStr(tritonMaxSubsegmentOpsVar);
     int ops = std::stoi(opsStr);
     setTritonMaxSubsegmentOps(ops);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string opsStr(tritonMaxSubsegmentOpsVar);
   int ops = std::stoi(opsStr);
   setTritonMaxSubsegmentOps(ops);
#endif
 }

 const char* tritonMaxSubsegmentSectionsVar = std::getenv("ND4J_TRITON_MAX_SUBSEGMENT_SECTIONS");
 if (tritonMaxSubsegmentSectionsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string sectionsStr(tritonMaxSubsegmentSectionsVar);
     int sections = std::stoi(sectionsStr);
     setTritonMaxSubsegmentSections(sections);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string sectionsStr(tritonMaxSubsegmentSectionsVar);
   int sections = std::stoi(sectionsStr);
   setTritonMaxSubsegmentSections(sections);
#endif
 }

 const char* tritonVerboseVar = std::getenv("ND4J_TRITON_VERBOSE");
 if (tritonVerboseVar != nullptr) {
   std::string verboseStr(tritonVerboseVar);
   if (verboseStr == "false" || verboseStr == "0" || verboseStr == "no") {
     setTritonVerbose(false);
   } else {
     setTritonVerbose(true);
   }
 }

 const char* tritonDumpSectionsVar = std::getenv("ND4J_TRITON_DUMP_SECTIONS");
 if (tritonDumpSectionsVar != nullptr) {
   std::string dumpSectionsStr(tritonDumpSectionsVar);
   if (dumpSectionsStr == "false" || dumpSectionsStr == "0" || dumpSectionsStr == "no") {
     setTritonDumpSections(false);
   } else {
     setTritonDumpSections(true);
   }
 }

 const char* tritonDumpArgsVar = std::getenv("ND4J_TRITON_DUMP_ARGS");
 if (tritonDumpArgsVar != nullptr) {
   std::string dumpArgsStr(tritonDumpArgsVar);
   if (dumpArgsStr == "false" || dumpArgsStr == "0" || dumpArgsStr == "no") {
     setTritonDumpArgs(false);
   } else {
     setTritonDumpArgs(true);
   }
 }

 const char* tritonLogAllPatternsVar = std::getenv("ND4J_TRITON_LOG_ALL_PATTERNS");
 if (tritonLogAllPatternsVar != nullptr) {
   std::string patternsStr(tritonLogAllPatternsVar);
   if (patternsStr == "false" || patternsStr == "0" || patternsStr == "no") {
     setTritonLogAllPatterns(false);
   } else {
     setTritonLogAllPatterns(true);
   }
 }

 const char* tritonCacheDirVar = std::getenv("ND4J_TRITON_CACHE_DIR");
 if (tritonCacheDirVar != nullptr) {
   setTritonCacheDir(std::string(tritonCacheDirVar));
 }

 const char* tritonDumpDirVar = std::getenv("ND4J_TRITON_DUMP_DIR");
 if (tritonDumpDirVar != nullptr) {
   setTritonDumpDir(std::string(tritonDumpDirVar));
 }

 const char* tritonOverrideDirVar = std::getenv("ND4J_TRITON_OVERRIDE_DIR");
 if (tritonOverrideDirVar != nullptr) {
   setTritonOverrideDir(std::string(tritonOverrideDirVar));
 }

 const char* tritonOverrideArchVar = std::getenv("ND4J_TRITON_OVERRIDE_ARCH");
 if (tritonOverrideArchVar != nullptr) {
   setTritonOverrideArch(std::string(tritonOverrideArchVar));
 }

 const char* tritonAlwaysCompileVar = std::getenv("ND4J_TRITON_ALWAYS_COMPILE");
 if (tritonAlwaysCompileVar != nullptr) {
   std::string alwaysCompileStr(tritonAlwaysCompileVar);
   if (alwaysCompileStr == "false" || alwaysCompileStr == "0" || alwaysCompileStr == "no") {
     setTritonAlwaysCompile(false);
   } else {
     setTritonAlwaysCompile(true);
   }
 }

 const char* tritonKernelDumpVar = std::getenv("ND4J_TRITON_KERNEL_DUMP");
 if (tritonKernelDumpVar != nullptr) {
   std::string kernelDumpStr(tritonKernelDumpVar);
   if (kernelDumpStr == "false" || kernelDumpStr == "0" || kernelDumpStr == "no") {
     setTritonKernelDump(false);
   } else {
     setTritonKernelDump(true);
   }
 }

 const char* tritonKernelOverrideVar = std::getenv("ND4J_TRITON_KERNEL_OVERRIDE");
 if (tritonKernelOverrideVar != nullptr) {
   std::string kernelOverrideStr(tritonKernelOverrideVar);
   if (kernelOverrideStr == "false" || kernelOverrideStr == "0" || kernelOverrideStr == "no") {
     setTritonKernelOverride(false);
   } else {
     setTritonKernelOverride(true);
   }
 }

 const char* tritonEnableFpFusionVar = std::getenv("ND4J_TRITON_ENABLE_FP_FUSION");
 if (tritonEnableFpFusionVar != nullptr) {
   std::string fpFusionStr(tritonEnableFpFusionVar);
   if (fpFusionStr == "false" || fpFusionStr == "0" || fpFusionStr == "no") {
     setTritonEnableFpFusion(false);
   } else {
     setTritonEnableFpFusion(true);
   }
 }

 const char* tritonDisableLineInfoVar = std::getenv("ND4J_TRITON_DISABLE_LINE_INFO");
 if (tritonDisableLineInfoVar != nullptr) {
   std::string lineInfoStr(tritonDisableLineInfoVar);
   if (lineInfoStr == "false" || lineInfoStr == "0" || lineInfoStr == "no") {
     setTritonDisableLineInfo(false);
   } else {
     setTritonDisableLineInfo(true);
   }
 }

 const char* tritonNumWarpsVar = std::getenv("ND4J_TRITON_NUM_WARPS");
 if (tritonNumWarpsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string warpsStr(tritonNumWarpsVar);
     int warps = std::stoi(warpsStr);
     setTritonNumWarps(warps);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string warpsStr(tritonNumWarpsVar);
   int warps = std::stoi(warpsStr);
   setTritonNumWarps(warps);
#endif
 }

 const char* tritonNumStagesVar = std::getenv("ND4J_TRITON_NUM_STAGES");
 if (tritonNumStagesVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string stagesStr(tritonNumStagesVar);
     int stages = std::stoi(stagesStr);
     setTritonNumStages(stages);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string stagesStr(tritonNumStagesVar);
   int stages = std::stoi(stagesStr);
   setTritonNumStages(stages);
#endif
 }

 const char* tritonNumCTAsVar = std::getenv("ND4J_TRITON_NUM_CTAS");
 if (tritonNumCTAsVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string ctasStr(tritonNumCTAsVar);
     int ctas = std::stoi(ctasStr);
     setTritonNumCTAs(ctas);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string ctasStr(tritonNumCTAsVar);
   int ctas = std::stoi(ctasStr);
   setTritonNumCTAs(ctas);
#endif
 }

 const char* tritonMaxNregVar = std::getenv("ND4J_TRITON_MAXNREG");
 if (tritonMaxNregVar != nullptr) {
#ifdef __cpp_exceptions
   try {
     std::string maxNregStr(tritonMaxNregVar);
     int maxNreg = std::stoi(maxNregStr);
     setTritonMaxNreg(maxNreg);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   std::string maxNregStr(tritonMaxNregVar);
   int maxNreg = std::stoi(maxNregStr);
   setTritonMaxNreg(maxNreg);
#endif
 }

 const char* tritonAllowFallbackCaptureVar = std::getenv("ND4J_TRITON_ALLOW_FALLBACK_CAPTURE");
 if (tritonAllowFallbackCaptureVar != nullptr) {
   std::string val(tritonAllowFallbackCaptureVar);
   setTritonAllowFallbackCapture(val != "false" && val != "0" && val != "no");
 }

 const char* tritonGraphCaptureVar = std::getenv("ND4J_TRITON_GRAPH_CAPTURE");
 if (tritonGraphCaptureVar != nullptr) {
   std::string val(tritonGraphCaptureVar);
   setTritonGraphCapture(val != "false" && val != "0" && val != "no");
 }

 const char* tritonDumpGraphDotVar = std::getenv("ND4J_TRITON_DUMP_GRAPH_DOT");
 if (tritonDumpGraphDotVar != nullptr) {
   std::string val(tritonDumpGraphDotVar);
   setTritonDumpGraphDot(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 // Triton CUDA graph replay configuration knobs — all default OFF
 auto readBoolEnv = [](const char* name) -> int {
   const char* v = std::getenv(name);
   if (!v) return -1;
   std::string s(v);
   return (s == "1" || s == "true" || s == "TRUE" || s == "ON") ? 1 : 0;
 };
 { int v = readBoolEnv("ND4J_TRITON_GRAPH_CTX_PUSH");     if (v >= 0) setTritonGraphCtxPush(v); }
 { int v = readBoolEnv("ND4J_TRITON_GRAPH_REINSTANTIATE"); if (v >= 0) setTritonGraphReinstantiate(v); }
 { int v = readBoolEnv("ND4J_TRITON_GRAPH_AUTOFREE");      if (v >= 0) setTritonGraphAutoFree(v); }
 { int v = readBoolEnv("ND4J_TRITON_GRAPH_DOT_VERBOSE");   if (v >= 0) setTritonGraphDotVerbose(v); }
 { int v = readBoolEnv("ND4J_TRITON_COMPILE_ALL");         if (v >= 0) setTritonCompileAll(v); }
 { int v = readBoolEnv("ND4J_DSP_BATCH_ZERO");             if (v >= 0) setDspBatchZero(v); }
 { int v = readBoolEnv("ND4J_DSP_BATCH_ZERO_VERBOSE");     if (v >= 0) setDspBatchZeroVerbose(v); }
 { int v = readBoolEnv("ND4J_DSP_BATCH_ZERO_GAP_ONLY");    if (v >= 0) setDspBatchZeroGapOnly(v); }
 { int v = readBoolEnv("ND4J_DSP_BATCH_ZERO_KERNEL");      if (v >= 0) setDspBatchZeroKernel(v); }

 const char* tritonExcludeOpsVar = std::getenv("ND4J_TRITON_EXCLUDE_OPS");
 if (tritonExcludeOpsVar != nullptr) {
   setTritonExcludeOps(std::string(tritonExcludeOpsVar));
 }

 const char* tritonIncludeTypesVar = std::getenv("ND4J_TRITON_INCLUDE_TYPES");
 if (tritonIncludeTypesVar != nullptr) {
   setTritonIncludeTypes(std::string(tritonIncludeTypesVar));
 }

 const char* tritonSkipKernelsVar = std::getenv("ND4J_TRITON_SKIP_KERNELS");
 if (tritonSkipKernelsVar != nullptr) {
   std::string val(tritonSkipKernelsVar);
   setTritonSkipKernels(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonVerifyKernelsVar = std::getenv("ND4J_TRITON_VERIFY_KERNELS");
 if (tritonVerifyKernelsVar != nullptr) {
   std::string val(tritonVerifyKernelsVar);
   setTritonVerifyKernels(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonVerifyKeepNativeVar = std::getenv("ND4J_TRITON_VERIFY_KEEP_NATIVE");
 if (tritonVerifyKeepNativeVar != nullptr) {
   std::string val(tritonVerifyKeepNativeVar);
   setTritonVerifyKeepNative(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonMaxSubKernelIndexVar = std::getenv("ND4J_TRITON_MAX_SUB_KERNEL_INDEX");
 if (tritonMaxSubKernelIndexVar != nullptr) {
   setTritonMaxSubKernelIndex(std::atoi(tritonMaxSubKernelIndexVar));
 }

 const char* tritonVerifyFullSnapshotVar = std::getenv("ND4J_TRITON_VERIFY_FULL_SNAPSHOT");
 if (tritonVerifyFullSnapshotVar != nullptr) {
   std::string val(tritonVerifyFullSnapshotVar);
   setTritonVerifyFullSnapshot(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonForceRecaptureVar = std::getenv("ND4J_TRITON_FORCE_RECAPTURE");
 if (tritonForceRecaptureVar != nullptr) {
   std::string val(tritonForceRecaptureVar);
   setTritonForceRecapture(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* tritonCaptureMinExecVar = std::getenv("ND4J_TRITON_CAPTURE_MIN_EXEC");
 if (tritonCaptureMinExecVar != nullptr) {
   setTritonCaptureMinExec(std::atoi(tritonCaptureMinExecVar));
 }

 const char* dspCastEliminationVar = std::getenv("ND4J_DSP_CAST_ELIMINATION");
 if (dspCastEliminationVar != nullptr) {
   std::string val(dspCastEliminationVar);
   setDspCastElimination(val != "false" && val != "0" && val != "no");
 }

 const char* dspMatmulSegmentationVar = std::getenv("ND4J_DSP_MATMUL_SEGMENTATION");
 if (dspMatmulSegmentationVar != nullptr) {
   std::string val(dspMatmulSegmentationVar);
   setDspMatmulSegmentation(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 const char* dspFp16ComputeVar = std::getenv("ND4J_DSP_FP16_COMPUTE");
 if (dspFp16ComputeVar != nullptr) {
   std::string val(dspFp16ComputeVar);
   setDspFp16Compute(val == "1" || val == "true" || val == "TRUE" || val == "ON");
 }

 { int v = readBoolEnv("ND4J_CUBLAS_TF32");                    if (v >= 0) setCublasTf32Enabled(v); }
 { int v = readBoolEnv("ND4J_DSP_CAST_SINK_MATMUL");           if (v >= 0) setDspCastSinkMatmul(v); }
 { int v = readBoolEnv("ND4J_TRITON_CONSOLIDATED_ARG_TABLE");   if (v >= 0) setTritonConsolidatedArgTable(v); }
 { int v = readBoolEnv("ND4J_TRITON_ARG_DIRTY_TRACKING");       if (v >= 0) setTritonArgDirtyTracking(v); }
 { int v = readBoolEnv("ND4J_TRITON_SECTION_FUSION");           if (v >= 0) setTritonSectionFusion(v); }
}
#endif


// CUDA configuration setters moved to Environment_CudaConfig.cpp

 bool Environment::isCheckOutputChange() { return _checkOutputChange.load(); }

 void Environment::setCheckOutputChange(bool reallyCheck) { _checkOutputChange.store(reallyCheck); }

 void Environment::setLogNativeNDArrayCreation(bool reallyLog) { _logNativeNDArrayCreation.store(reallyLog); }
 bool Environment::isLogNativeNDArrayCreation() { return _logNativeNDArrayCreation.load(); }

 /**
* When log ndarray events is set,
* more logging will happen around ndarrays such as what constructors are being called.
* @return
  */
 bool Environment::isLogNDArrayEvents() { return _logNDArrayEvenuts.load(); }
 void Environment::setLogNDArrayEvents(bool logNDArrayEvents) { _logNDArrayEvenuts.store(logNDArrayEvents); }

 bool Environment::isCheckInputChange() { return _checkInputChange.load(); }
 void Environment::setCheckInputChange(bool reallyCheck) { _checkInputChange.store(reallyCheck); }

 bool Environment::isDeleteShapeInfo() { return deleteShapeInfo; }
 void Environment::setDeleteShapeInfo(bool reallyDelete) { deleteShapeInfo = reallyDelete; }

 bool Environment::blasFallback() { return _blasFallback; }

bool Environment::isSerializeBlasCalls() {
  // Delegate to BlasHelper which manages the actual serialization
  return BlasHelper::getInstance().isSerializeBlasCalls();
}

void Environment::setSerializeBlasCalls(bool serialize) {
  _serializeBlasCallsSet.store(true);
  BlasHelper::getInstance().setSerializeBlasCalls(serialize);
}

int Environment::getOpenBlasThreads() {
  return BlasHelper::getInstance().getOpenblasThreads();
}

void Environment::setOpenBlasThreads(int threads) {
  _openBlasThreads.store(threads);
  BlasHelper::getInstance().setOpenblasThreads(threads);
}

 Environment::~Environment() {
   //
 }

 void Environment::setMaxPrimaryMemory(uint64_t maxBytes) { _maxTotalPrimaryMemory = maxBytes; }

 void Environment::setMaxSpecialyMemory(uint64_t maxBytes) { _maxTotalSpecialMemory = maxBytes; }

 void Environment::setMaxDeviceMemory(uint64_t maxBytes) { _maxDeviceMemory = maxBytes; }

 Environment &Environment::getInstance() {
   static Environment instance;
   return instance;
 }

 std::string Environment::homeDirectory() const {
#ifdef _WIN32
   const char* homeDrive = std::getenv("HOMEDRIVE");
   const char* homePath = std::getenv("HOMEPATH");
   if (homeDrive != nullptr && homePath != nullptr && homeDrive[0] != '\0' &&
       homePath[0] != '\0') {
     return std::string(homeDrive) + std::string(homePath);
   }
#endif
   const char* home = std::getenv("HOME");
   if (home != nullptr && home[0] != '\0') {
     return std::string(home);
   }
   return "";
 }

 std::string Environment::cudaToolkitPath() const {
   const char* cudaPath = std::getenv("CUDA_PATH");
   if (cudaPath != nullptr && cudaPath[0] != '\0') {
     return std::string(cudaPath);
   }
   return "";
 }

 bool Environment::isVerbose() { return _verbose.load(); }

 bool Environment::isExperimentalBuild() { return _experimental; }

 DataType Environment::defaultFloatDataType() { return _dataType.load(); }

 std::vector<Pair> &Environment::capabilities() { return _capabilities; }

 void Environment::setDefaultFloatDataType(DataType dtype) {
   if (dtype != FLOAT32 && dtype != DOUBLE && dtype != FLOAT8 && dtype != HALF)
     THROW_EXCEPTION("Default Float data type must be one of [FLOAT8, FLOAT16, FLOAT32, DOUBLE]");

   _dataType.store(dtype);
 }

 void Environment::setDeletePrimary(bool reallyDelete) { deletePrimary = reallyDelete; }

 bool Environment::isDeletePrimary() { return deletePrimary; }

 void Environment::setDeleteSpecial(bool reallyDelete) { deleteSpecial = reallyDelete; }

 bool Environment::isDeleteSpecial() { return deleteSpecial; }

 void Environment::setVerbose(bool reallyVerbose) { _verbose = reallyVerbose; }

 bool Environment::isDebug() { return _debug.load(); }

 bool Environment::isProfiling() { return _profile.load(); }

 bool Environment::isDetectingLeaks() { return _leaks.load(); }

 void Environment::setLeaksDetector(bool reallyDetect) { _leaks.store(reallyDetect); }

 void Environment::setProfiling(bool reallyProfile) { _profile.store(reallyProfile); }

 bool Environment::isDebugAndVerbose() { return this->isDebug() && this->isVerbose(); }

 void Environment::setDebug(bool reallyDebug) { _debug = reallyDebug; }

 int Environment::tadThreshold() { return _tadThreshold.load(); }

 void Environment::setTadThreshold(int threshold) { _tadThreshold = threshold; }

 int Environment::elementwiseThreshold() { return _elementThreshold.load(); }

 void Environment::setElementwiseThreshold(int threshold) { _elementThreshold = threshold; }

 int Environment::maxThreads() { return _maxThreads.load(); }

 int Environment::maxMasterThreads() { return _maxMasterThreads.load(); }

 void Environment::setMaxThreads(int max) {
   // allocate more threads if we want or limit number of threads
   _maxThreads.store(max);
 }

 void Environment::setMaxMasterThreads(int max) {
   if (max > maxThreads()) {
     max = maxThreads();
   }

   if (max < 1) return;

   _maxMasterThreads = max;
 }

 bool Environment::precisionBoostAllowed() { return _precBoost.load(); }

 void Environment::allowPrecisionBoost(bool reallyAllow) { _precBoost.store(reallyAllow); }

 bool Environment::isCPU() {
#ifdef SD_CUDA
   return false;
#else
   return true;
#endif
 }

 int Environment::blasMajorVersion() { return _blasMajorVersion; }

 int Environment::blasMinorVersion() { return _blasMinorVersion; }

 int Environment::blasPatchVersion() { return _blasPatchVersion; }

 bool Environment::helpersAllowed() { return _allowHelpers.load(); }

 void Environment::allowHelpers(bool reallyAllow) { _allowHelpers.store(reallyAllow); }

 void Environment::setGroupLimit(int group, LongType numBytes) {
   memory::MemoryCounter::getInstance().setGroupLimit((memory::MemoryType)group, numBytes);
 }

 void Environment::setDeviceLimit(int deviceId, LongType numBytes) {
   memory::MemoryCounter::getInstance().setDeviceLimit(deviceId, numBytes);
 }

 LongType Environment::getGroupLimit(int group) {
   return memory::MemoryCounter::getInstance().groupLimit((memory::MemoryType)group);
 }

 LongType Environment::getDeviceLimit(int deviceId) {
   return memory::MemoryCounter::getInstance().deviceLimit(deviceId);
 }

 LongType Environment::getGroupCounter(int group) {
   return memory::MemoryCounter::getInstance().allocatedGroup((memory::MemoryType)group);
 }

 LongType Environment::getDeviceCounter(int deviceId) {
   return memory::MemoryCounter::getInstance().allocatedDevice(deviceId);
 }

 uint64_t Environment::maxPrimaryMemory() { return _maxTotalPrimaryMemory.load(); }

 uint64_t Environment::maxSpecialMemory() { return _maxTotalSpecialMemory.load(); }

 bool Environment::isFuncTracePrintAllocate() { return this->funcTracePrintAllocate; }

 bool Environment::isFuncTracePrintDeallocate() { return this->funcTracePrintDeallocate; }

 void Environment::setFuncTracePrintAllocate(bool reallyPrint) { this->funcTracePrintAllocate = reallyPrint; }

 void Environment::setFuncTracePrintDeallocate(bool reallyPrint) { this->funcTracePrintDeallocate = reallyPrint; }

 // NDArray lifecycle tracking getters/setters
 bool Environment::isLifecycleTracking() { return _lifecycleTracking.load(); }

 void Environment::setLifecycleTracking(bool enabled) { _lifecycleTracking.store(enabled); }

 bool Environment::isTrackViews() { return _trackViews.load(); }

 void Environment::setTrackViews(bool track) { _trackViews.store(track); }

 bool Environment::isTrackDeletions() { return _trackDeletions.load(); }

 void Environment::setTrackDeletions(bool track) { _trackDeletions.store(track); }

 int Environment::getStackDepth() { return _stackDepth.load(); }

 void Environment::setStackDepth(int depth) {
   if (depth > 0) {
     _stackDepth.store(depth);
   }
 }

 int Environment::getReportInterval() { return _reportInterval.load(); }

 void Environment::setReportInterval(int seconds) {
   if (seconds > 0) {
     _reportInterval.store(seconds);
   }
 }

 size_t Environment::getMaxDeletionHistory() { return _maxDeletionHistory.load(); }

 void Environment::setMaxDeletionHistory(size_t max) { _maxDeletionHistory.store(max); }

 bool Environment::isSnapshotFiles() { return _snapshotFiles.load(); }

 void Environment::setSnapshotFiles(bool enabled) { _snapshotFiles.store(enabled); }

 bool Environment::isTrackOperations() { return _trackOperations.load(); }

 void Environment::setTrackOperations(bool enabled) { _trackOperations.store(enabled); }

 // Individual tracker enable/disable methods
 bool Environment::isNDArrayTracking() { return _ndArrayTracking.load(); }

 void Environment::setNDArrayTracking(bool enabled) {
   _ndArrayTracking.store(enabled);
   array::NDArrayLifecycleTracker::getInstance().setEnabled(enabled);
 }

 bool Environment::isDataBufferTracking() { return _dataBufferTracking.load(); }

 void Environment::setDataBufferTracking(bool enabled) {
   _dataBufferTracking.store(enabled);
   array::DataBufferLifecycleTracker::getInstance().setEnabled(enabled);
 }

 bool Environment::isTADCacheTracking() { return _tadCacheTracking.load(); }

 void Environment::setTADCacheTracking(bool enabled) {
   _tadCacheTracking.store(enabled);
   array::TADCacheLifecycleTracker::getInstance().setEnabled(enabled);
 }

 bool Environment::isShapeCacheTracking() { return _shapeCacheTracking.load(); }

 void Environment::setShapeCacheTracking(bool enabled) {
   _shapeCacheTracking.store(enabled);
   array::ShapeCacheLifecycleTracker::getInstance().setEnabled(enabled);
 }

 bool Environment::isOpContextTracking() { return _opContextTracking.load(); }

 void Environment::setOpContextTracking(bool enabled) {
   _opContextTracking.store(enabled);
   graph::OpContextLifecycleTracker::getInstance().setEnabled(enabled);
 }

 // NDArray print options (NumPy-style printoptions)
 void Environment::setPrintEdgeItems(int edgeItems) {
   if (edgeItems > 0) {
     _printEdgeItems.store(edgeItems);
   }
 }

 void Environment::setPrintThreshold(int threshold) {
   if (threshold > 0) {
     _printThreshold.store(threshold);
   }
 }

 void Environment::setPrintLineWidth(int lineWidth) {
   if (lineWidth > 0) {
     _printLineWidth.store(lineWidth);
   }
 }

  void Environment::setPrintPrecision(int precision) {
    if (precision >= 0 && precision <= 20) {
      _printPrecision.store(precision);
    }
  }

  // Triton GPU compilation settings
  void Environment::setTritonBuildThreads(int threads) {
    if (threads > 0 && threads <= 16) {
      _tritonBuildThreads.store(threads);
    }
  }

  void Environment::setTritonCacheEnabled(bool enabled) {
    _tritonCacheEnabled.store(enabled);
  }

  void Environment::setTritonCooperativeLaunch(bool enabled) {
    _tritonCooperativeLaunch.store(enabled);
  }

  void Environment::setTritonCoopTargetBlocks(int blocks) {
    if (blocks < 0) {
      blocks = 0;
    }
    _tritonCoopTargetBlocks.store(blocks);
  }

  void Environment::setTritonMaxSubsegmentOps(int ops) {
    if (ops < 0) {
      ops = 0;
    }
    _tritonMaxSubsegmentOps.store(ops);
  }

  void Environment::setTritonMaxSubsegmentSections(int sections) {
    if (sections < 0) {
      sections = 0;
    }
    _tritonMaxSubsegmentSections.store(sections);
  }

  void Environment::setTritonVerbose(bool verbose) {
    _tritonVerbose.store(verbose);
  }

  void Environment::setTritonDumpSections(bool dumpSections) {
    _tritonDumpSections.store(dumpSections);
  }

  void Environment::setTritonDumpArgs(bool dumpArgs) {
    _tritonDumpArgs.store(dumpArgs);
  }

  void Environment::setTritonLogAllPatterns(bool logAllPatterns) {
    _tritonLogAllPatterns.store(logAllPatterns);
  }

  void Environment::setTritonAlwaysCompile(bool alwaysCompile) {
    _tritonAlwaysCompile.store(alwaysCompile);
  }

  void Environment::setTritonKernelDump(bool kernelDump) {
    _tritonKernelDump.store(kernelDump);
  }

  void Environment::setTritonKernelOverride(bool kernelOverride) {
    _tritonKernelOverride.store(kernelOverride);
  }

  void Environment::setTritonNumWarps(int warps) {
    if (warps < 0) {
      warps = 0;
    }
    _tritonNumWarps.store(warps);
  }

  void Environment::setTritonNumStages(int stages) {
    if (stages < 0) {
      stages = 0;
    }
    _tritonNumStages.store(stages);
  }

  void Environment::setTritonNumCTAs(int ctas) {
    if (ctas < 1) {
      ctas = 1;
    }
    _tritonNumCTAs.store(ctas);
  }

  void Environment::setTritonMaxNreg(int maxNreg) {
    if (maxNreg < 0) {
      maxNreg = 0;
    }
    _tritonMaxNreg.store(maxNreg);
  }

  void Environment::setTritonEnableFpFusion(bool enableFpFusion) {
    _tritonEnableFpFusion.store(enableFpFusion);
  }

  void Environment::setTritonDisableLineInfo(bool disableLineInfo) {
    _tritonDisableLineInfo.store(disableLineInfo);
  }

  void Environment::setTritonCacheDir(const std::string& cacheDir) {
    _tritonCacheDir = cacheDir;
  }

  void Environment::setTritonDumpDir(const std::string& dumpDir) {
    _tritonDumpDir = dumpDir;
  }

  void Environment::setTritonOverrideDir(const std::string& overrideDir) {
    _tritonOverrideDir = overrideDir;
  }

  void Environment::setTritonAllowFallbackCapture(bool allow) {
    _tritonAllowFallbackCapture.store(allow);
  }

  void Environment::setTritonSkipKernels(bool skip) {
    _tritonSkipKernels.store(skip);
  }

  void Environment::setTritonVerifyKernels(bool verify) {
    _tritonVerifyKernels.store(verify);
  }

  void Environment::setTritonVerifyKeepNative(bool v) {
    _tritonVerifyKeepNative.store(v);
  }

  void Environment::setTritonMaxSubKernelIndex(int idx) {
    _tritonMaxSubKernelIndex.store(idx);
  }

  void Environment::setTritonVerifyFullSnapshot(bool v) {
    _tritonVerifyFullSnapshot.store(v);
  }

  void Environment::setTritonForceRecapture(bool v) {
    _tritonForceRecapture.store(v);
  }

  void Environment::setTritonCaptureMinExec(int v) {
    _tritonCaptureMinExec.store(v);
  }

  bool Environment::isTritonExcludedOp(const std::string& opName) const {
    if (_tritonExcludeOps.empty()) return false;
    // Parse comma-separated exclusion list and check for match.
    // Supports both exact match and case-insensitive match.
    std::string lower;
    lower.reserve(opName.size());
    for (char c : opName) lower += static_cast<char>(std::tolower(c));

    size_t start = 0;
    while (start < _tritonExcludeOps.size()) {
      size_t end = _tritonExcludeOps.find(',', start);
      if (end == std::string::npos) end = _tritonExcludeOps.size();
      // Trim whitespace
      size_t s = start, e = end;
      while (s < e && std::isspace(_tritonExcludeOps[s])) s++;
      while (e > s && std::isspace(_tritonExcludeOps[e - 1])) e--;
      if (e > s) {
        std::string token;
        token.reserve(e - s);
        for (size_t i = s; i < e; i++) token += static_cast<char>(std::tolower(_tritonExcludeOps[i]));
        if (token == lower) return true;
      }
      start = end + 1;
    }
    return false;
  }

  void Environment::setDspCastElimination(bool enabled) {
    _dspCastElimination.store(enabled);
  }

  void Environment::setDspMatmulSegmentation(bool enabled) {
    _dspMatmulSegmentation.store(enabled);
  }

  void Environment::setDspFp16Compute(bool enabled) {
    _dspFp16Compute.store(enabled);
  }

  void Environment::setCublasTf32Enabled(bool enabled) {
    _cublasTf32Enabled.store(enabled);
  }

  void Environment::setDspCastSinkMatmul(bool enabled) {
    _dspCastSinkMatmul.store(enabled);
  }

  void Environment::setTritonConsolidatedArgTable(bool enabled) {
    _tritonConsolidatedArgTable.store(enabled);
  }

  void Environment::setTritonArgDirtyTracking(bool enabled) {
    _tritonArgDirtyTracking.store(enabled);
  }

  void Environment::setTritonSectionFusion(bool enabled) {
    _tritonSectionFusion.store(enabled);
  }

  void Environment::setTritonOverrideArch(const std::string& overrideArch) {
    _tritonOverrideArch = overrideArch;
  }
}
