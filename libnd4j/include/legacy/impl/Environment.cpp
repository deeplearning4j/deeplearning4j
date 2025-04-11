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

#include <helpers/StringUtils.h>
#include <helpers/logger.h>
#include <memory/MemoryCounter.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime.h>
#include <system/BlasVersionHelper.h>

#endif

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
 }
#endif
 /**
  * Defines size of thread pool used for parallelism
  */
 const char *max_threads = std::getenv("SD_MAX_THREADS");
 if (max_threads != nullptr) {
   try {
     std::string t(max_threads);
     int val = std::stoi(t);
     _maxThreads.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
 }

 /**
  * Defines max number of threads usable at once
  */
 const char *max_master_threads = std::getenv("SD_MASTER_THREADS");
 if (max_master_threads != nullptr) {
   try {
     std::string t(max_master_threads);
     int val = std::stoi(t);
     _maxMasterThreads.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
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
   try {
     std::string t(max_primary_memory);
     auto val = std::stol(t);
     _maxTotalPrimaryMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
 }

 /**
  * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
  */
 const char *max_special_memory = std::getenv("SD_MAX_SPECIAL_BYTES");
 if (max_special_memory != nullptr) {
   try {
     std::string t(max_special_memory);
     auto val = std::stol(t);
     _maxTotalSpecialMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
 }

 /**
  * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
  */
 const char *max_device_memory = std::getenv("SD_MAX_DEVICE_BYTES");
 if (max_device_memory != nullptr) {
   try {
     std::string t(max_device_memory);
     auto val = std::stol(t);
     _maxDeviceMemory.store(val);
   } catch (std::invalid_argument &e) {
     // just do nothing
   } catch (std::out_of_range &e) {
     // still do nothing
   }
 }

 const char *blas_fallback = std::getenv("SD_BLAS_FALLBACK");
 if (blas_fallback != nullptr) {
   _blasFallback = true;
 }

#ifdef __CUDABLAS__
 int devCnt = 0;
 cudaGetDeviceCount(&devCnt);
 _cudaDeviceCount.store(devCnt);
 printf("During environment initialization we found [%i] CUDA devices\n", devCnt);
 auto devProperties = new cudaDeviceProp[devCnt];
 for (int i = 0; i < devCnt; i++) {
   cudaSetDevice(i);
   cudaGetDeviceProperties(&devProperties[i], i);

   Pair p(devProperties[i].major, devProperties[i].minor);
   _capabilities.emplace_back(p);
 }

 BlasVersionHelper ver;
 _blasMajorVersion = ver._blasMajorVersion;
 _blasMinorVersion = ver._blasMinorVersion;
 _blasPatchVersion = ver._blasPatchVersion;

 // Initialize CUDA environment settings
 initCudaEnvironment();

 // Initialize CUDA device limits
 initCudaDeviceLimits();

 // Set initial device to 0
 cudaSetDevice(0);
 _cudaCurrentDevice.store(0);

 delete[] devProperties;
#else
 // No CUDA environment to initialize
#endif
}

bool Environment::setCudaDeviceLimit(int limitType, size_t value) {
 CudaLimitType limitType2 = static_cast<CudaLimitType>(limitType);
#ifdef __CUDABLAS__
 cudaLimit cudaLimitValue;

 // Map our enum to CUDA's enum
 switch (limitType2) {
   case CUDA_LIMIT_STACK_SIZE:
     cudaLimitValue = cudaLimitStackSize;
     break;
   case CUDA_LIMIT_MALLOC_HEAP_SIZE:
     cudaLimitValue = cudaLimitMallocHeapSize;
     break;
   case CUDA_LIMIT_PRINTF_FIFO_SIZE:
     cudaLimitValue = cudaLimitPrintfFifoSize;
     break;
   case CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
     cudaLimitValue = cudaLimitDevRuntimeSyncDepth;
     break;
   case CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
     cudaLimitValue = cudaLimitDevRuntimePendingLaunchCount;
     break;
   case CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY:
     cudaLimitValue = cudaLimitMaxL2FetchGranularity;
     break;
   case CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE:
#if CUDART_VERSION >= 10000
     cudaLimitValue = cudaLimitPersistingL2CacheSize;
#else
     sd_printf("Warning: CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE requires CUDA 10.0 or newer\n", "");
     return false;
#endif
     break;
   default:
     sd_printf("Warning: Unknown CUDA limit type: %d\n", limitType);
     return false;
 }

 cudaError_t err = cudaDeviceSetLimit(cudaLimitValue, value);
 if (err != cudaSuccess) {
   sd_printf("Warning: Failed to set CUDA device limit, error: %s\n", cudaGetErrorString(err));
   return false;
 }
 return true;
#else
 return false;
#endif
}

// Then update all the individual methods:
void Environment::setCudaStackSize(size_t size) {
 if (setCudaDeviceLimit(CUDA_LIMIT_STACK_SIZE, size)) {
   _cudaStackSize.store(size);
 }
}

void Environment::setCudaMallocHeapSize(size_t size) {
 if (setCudaDeviceLimit(CUDA_LIMIT_MALLOC_HEAP_SIZE, size)) {
   _cudaMallocHeapSize.store(size);
 }
}

void Environment::setCudaPrintfFifoSize(size_t size) {
 if (setCudaDeviceLimit(CUDA_LIMIT_PRINTF_FIFO_SIZE, size)) {
   _cudaPrintfFifoSize.store(size);
 }
}

void Environment::setCudaDevRuntimeSyncDepth(size_t depth) {
 if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH, depth)) {
   _cudaDevRuntimeSyncDepth.store(depth);
 }
}

void Environment::setCudaDevRuntimePendingLaunchCount(size_t count) {
 if (setCudaDeviceLimit(CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, count)) {
   _cudaDevRuntimePendingLaunchCount.store(count);
 }
}

void Environment::setCudaMaxL2FetchGranularity(size_t size) {
 if (setCudaDeviceLimit(CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY, size)) {
   _cudaMaxL2FetchGranularity.store(size);
 }
}

void Environment::setCudaPersistingL2CacheSize(size_t size) {
 if (setCudaDeviceLimit(CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE, size)) {
   _cudaPersistingL2CacheSize.store(size);
 }
}


void Environment::initCudaDeviceLimits() {
 // Get the current values for all device limits to initialize our variables
 size_t value;
 if (cudaDeviceGetLimit(&value, cudaLimitStackSize) == cudaSuccess) {
   _cudaStackSize.store(value);
 }

 if (cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize) == cudaSuccess) {
   _cudaMallocHeapSize.store(value);
 }

 if (cudaDeviceGetLimit(&value, cudaLimitPrintfFifoSize) == cudaSuccess) {
   _cudaPrintfFifoSize.store(value);
 }

 if (cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth) == cudaSuccess) {
   _cudaDevRuntimeSyncDepth.store(value);
 }

 if (cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount) == cudaSuccess) {
   _cudaDevRuntimePendingLaunchCount.store(value);
 }

 if (cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity) == cudaSuccess) {
   _cudaMaxL2FetchGranularity.store(value);
 }

#if CUDART_VERSION >= 10000
 if (cudaDeviceGetLimit(&value, cudaLimitPersistingL2CacheSize) == cudaSuccess) {
   _cudaPersistingL2CacheSize.store(value);
 }
#endif

 // Load custom limits from environment variables
 const char* stackSizeVar = std::getenv("SD_CUDA_STACK_SIZE");
 if (stackSizeVar != nullptr) {
   try {
     std::string sizeStr(stackSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaStackSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* heapSizeVar = std::getenv("SD_CUDA_MALLOC_HEAP_SIZE");
 if (heapSizeVar != nullptr) {
   try {
     std::string sizeStr(heapSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaMallocHeapSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* printfSizeVar = std::getenv("SD_CUDA_PRINTF_FIFO_SIZE");
 if (printfSizeVar != nullptr) {
   try {
     std::string sizeStr(printfSizeVar);
     size_t size = std::stol(sizeStr);
     setCudaPrintfFifoSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* syncDepthVar = std::getenv("SD_CUDA_DEV_RUNTIME_SYNC_DEPTH");
 if (syncDepthVar != nullptr) {
   try {
     std::string depthStr(syncDepthVar);
     size_t depth = std::stol(depthStr);
     setCudaDevRuntimeSyncDepth(depth);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* pendingLaunchVar = std::getenv("SD_CUDA_DEV_RUNTIME_PENDING_LAUNCH_COUNT");
 if (pendingLaunchVar != nullptr) {
   try {
     std::string countStr(pendingLaunchVar);
     size_t count = std::stol(countStr);
     setCudaDevRuntimePendingLaunchCount(count);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* l2FetchVar = std::getenv("SD_CUDA_MAX_L2_FETCH_GRANULARITY");
 if (l2FetchVar != nullptr) {
   try {
     std::string sizeStr(l2FetchVar);
     size_t size = std::stol(sizeStr);
     setCudaMaxL2FetchGranularity(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* l2CacheVar = std::getenv("SD_CUDA_PERSISTING_L2_CACHE_SIZE");
 if (l2CacheVar != nullptr) {
#if CUDART_VERSION >= 10000
   try {
     std::string sizeStr(l2CacheVar);
     size_t size = std::stol(sizeStr);
     setCudaPersistingL2CacheSize(size);
   } catch (std::exception &e) {
     // Do nothing on error
   }
#else
   sd_printf("Warning: SD_CUDA_PERSISTING_L2_CACHE_SIZE requires CUDA 10.0 or newer\n", "");
#endif
 }
}

void Environment::initCudaEnvironment() {
#ifdef __CUDABLAS__
 // Initialize CUDA environment settings from environment variables
 const char* cudaDeviceVar = std::getenv("SD_CUDA_DEVICE");
 if (cudaDeviceVar != nullptr) {
   try {
     std::string devStr(cudaDeviceVar);
     int device = std::stoi(devStr);
     if (device >= 0 && device < _cudaDeviceCount.load()) {
       _cudaCurrentDevice.store(device);
       cudaSetDevice(device);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
#endif
 }

 const char* cudaPinnedVar = std::getenv("SD_CUDA_PINNED_MEMORY");
 if (cudaPinnedVar != nullptr) {
   std::string pinnedStr(cudaPinnedVar);
   if (pinnedStr == "true" || pinnedStr == "1" || pinnedStr == "yes") {
     _cudaMemoryPinned.store(true);
   } else {
     _cudaMemoryPinned.store(false);
   }
 }

 const char* cudaManagedVar = std::getenv("SD_CUDA_MANAGED_MEMORY");
 if (cudaManagedVar != nullptr) {
   std::string managedStr(cudaManagedVar);
   if (managedStr == "true" || managedStr == "1" || managedStr == "yes") {
     _cudaUseManagedMemory.store(true);
   } else {
     _cudaUseManagedMemory.store(false);
   }
 }

 const char* cudaPoolSizeVar = std::getenv("SD_CUDA_MEMORY_POOL_SIZE");
 if (cudaPoolSizeVar != nullptr) {
   try {
     std::string sizeStr(cudaPoolSizeVar);
     int size = std::stoi(sizeStr);
     if (size > 0) {
       _cudaMemoryPoolSize.store(size);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaForceP2PVar = std::getenv("SD_CUDA_FORCE_P2P");
 if (cudaForceP2PVar != nullptr) {
   std::string p2pStr(cudaForceP2PVar);
   if (p2pStr == "true" || p2pStr == "1" || p2pStr == "yes") {
     _cudaForceP2P.store(true);
   } else {
     _cudaForceP2P.store(false);
   }
 }

 const char* cudaAllocatorVar = std::getenv("SD_CUDA_ALLOCATOR_ENABLED");
 if (cudaAllocatorVar != nullptr) {
   std::string allocStr(cudaAllocatorVar);
   if (allocStr == "false" || allocStr == "0" || allocStr == "no") {
     _cudaAllocatorEnabled.store(false);
   } else {
     _cudaAllocatorEnabled.store(true);
   }
 }

 const char* cudaMaxBlocksVar = std::getenv("SD_CUDA_MAX_BLOCKS");
 if (cudaMaxBlocksVar != nullptr) {
   try {
     std::string blocksStr(cudaMaxBlocksVar);
     int blocks = std::stoi(blocksStr);
     if (blocks > 0) {
       _cudaMaxBlocks.store(blocks);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaMaxThreadsVar = std::getenv("SD_CUDA_MAX_THREADS_PER_BLOCK");
 if (cudaMaxThreadsVar != nullptr) {
   try {
     std::string threadsStr(cudaMaxThreadsVar);
     int threads = std::stoi(threadsStr);
     if (threads > 0) {
       _cudaMaxThreadsPerBlock.store(threads);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaAsyncVar = std::getenv("SD_CUDA_ASYNC_EXECUTION");
 if (cudaAsyncVar != nullptr) {
   std::string asyncStr(cudaAsyncVar);
   if (asyncStr == "false" || asyncStr == "0" || asyncStr == "no") {
     _cudaAsyncExecution.store(false);
   } else {
     _cudaAsyncExecution.store(true);
   }
 }

 const char* cudaStreamLimitVar = std::getenv("SD_CUDA_STREAM_LIMIT");
 if (cudaStreamLimitVar != nullptr) {
   try {
     std::string limitStr(cudaStreamLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaStreamLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaDeviceHostVar = std::getenv("SD_CUDA_USE_DEVICE_HOST");
 if (cudaDeviceHostVar != nullptr) {
   std::string deviceStr(cudaDeviceHostVar);
   if (deviceStr == "true" || deviceStr == "1" || deviceStr == "yes") {
     _cudaUseDeviceHost.store(true);
   } else {
     _cudaUseDeviceHost.store(false);
   }
 }

 const char* cudaEventLimitVar = std::getenv("SD_CUDA_EVENT_LIMIT");
 if (cudaEventLimitVar != nullptr) {
   try {
     std::string limitStr(cudaEventLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaEventLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaCachingLimitVar = std::getenv("SD_CUDA_CACHING_ALLOCATOR_LIMIT");
 if (cudaCachingLimitVar != nullptr) {
   try {
     std::string limitStr(cudaCachingLimitVar);
     int limit = std::stoi(limitStr);
     if (limit > 0) {
       _cudaCachingAllocatorLimit.store(limit);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaUnifiedMemVar = std::getenv("SD_CUDA_USE_UNIFIED_MEMORY");
 if (cudaUnifiedMemVar != nullptr) {
   std::string unifiedStr(cudaUnifiedMemVar);
   if (unifiedStr == "true" || unifiedStr == "1" || unifiedStr == "yes") {
     _cudaUseUnifiedMemory.store(true);
   } else {
     _cudaUseUnifiedMemory.store(false);
   }
 }

 const char* cudaPrefetchVar = std::getenv("SD_CUDA_PREFETCH_SIZE");
 if (cudaPrefetchVar != nullptr) {
   try {
     std::string sizeStr(cudaPrefetchVar);
     int size = std::stoi(sizeStr);
     if (size > 0) {
       _cudaPrefetchSize.store(size);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaGraphVar = std::getenv("SD_CUDA_GRAPH_OPTIMIZATION");
 if (cudaGraphVar != nullptr) {
   std::string graphStr(cudaGraphVar);
   if (graphStr == "true" || graphStr == "1" || graphStr == "yes") {
     _cudaGraphOptimization.store(true);
   } else {
     _cudaGraphOptimization.store(false);
   }
 }

 const char* cudaTensorCoreVar = std::getenv("SD_CUDA_TENSOR_CORE_ENABLED");
 if (cudaTensorCoreVar != nullptr) {
   std::string tensorStr(cudaTensorCoreVar);
   if (tensorStr == "false" || tensorStr == "0" || tensorStr == "no") {
     _cudaTensorCoreEnabled.store(false);
   } else {
     _cudaTensorCoreEnabled.store(true);
   }
 }

 const char* cudaBlockingSyncVar = std::getenv("SD_CUDA_BLOCKING_SYNC");
 if (cudaBlockingSyncVar != nullptr) {
   try {
     std::string syncStr(cudaBlockingSyncVar);
     int sync = std::stoi(syncStr);
     if (sync >= 0 && sync <= 1) {
       _cudaBlockingSync.store(sync);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }

 const char* cudaDeviceScheduleVar = std::getenv("SD_CUDA_DEVICE_SCHEDULE");
 if (cudaDeviceScheduleVar != nullptr) {
   try {
     std::string scheduleStr(cudaDeviceScheduleVar);
     int schedule = std::stoi(scheduleStr);
     if (schedule >= 0 && schedule <= 3) {
       _cudaDeviceSchedule.store(schedule);
     }
   } catch (std::exception &e) {
     // Do nothing on error
   }
 }
}

void Environment::setCudaCurrentDevice(int device) {
 if (device >= 0 && device < _cudaDeviceCount.load()) {
   cudaError_t err = cudaSetDevice(device);
   if (err == cudaSuccess) {
     _cudaCurrentDevice.store(device);
   } else {
     sd_printf("Warning: Failed to set CUDA device to %d, error: %s\n", device, cudaGetErrorString(err));
   }
 } else {
   sd_printf("Warning: Attempted to set invalid CUDA device %d (valid range: 0-%d)\n", device, _cudaDeviceCount.load() - 1);
 }
}

void Environment::setCudaMemoryPinned(bool pinned) {
 _cudaMemoryPinned.store(pinned);
}

void Environment::setCudaUseManagedMemory(bool managed) {
 _cudaUseManagedMemory.store(managed);
}

void Environment::setCudaMemoryPoolSize(int sizeInMB) {
 if (sizeInMB >= 0) {
   _cudaMemoryPoolSize.store(sizeInMB);
 }
}

void Environment::setCudaForceP2P(bool forceP2P) {
 _cudaForceP2P.store(forceP2P);
}

void Environment::setCudaAllocatorEnabled(bool enabled) {
 _cudaAllocatorEnabled.store(enabled);
}

void Environment::setCudaMaxBlocks(int blocks) {
 if (blocks > 0) {
   _cudaMaxBlocks.store(blocks);
 }
}

void Environment::setCudaMaxThreadsPerBlock(int threads) {
 if (threads > 0) {
   _cudaMaxThreadsPerBlock.store(threads);
 }
}

void Environment::setCudaAsyncExecution(bool async) {
 _cudaAsyncExecution.store(async);
}

void Environment::setCudaStreamLimit(int limit) {
 if (limit > 0) {
   _cudaStreamLimit.store(limit);
 }
}

void Environment::setCudaUseDeviceHost(bool useDeviceHost) {
 _cudaUseDeviceHost.store(useDeviceHost);
}

void Environment::setCudaEventLimit(int limit) {
 if (limit > 0) {
   _cudaEventLimit.store(limit);
 }
}

void Environment::setCudaCachingAllocatorLimit(int limitInMB) {
 if (limitInMB >= 0) {
   _cudaCachingAllocatorLimit.store(limitInMB);
 }
}

void Environment::setCudaUseUnifiedMemory(bool unified) {
 _cudaUseUnifiedMemory.store(unified);
}

void Environment::setCudaPrefetchSize(int sizeInMB) {
 if (sizeInMB >= 0) {
   _cudaPrefetchSize.store(sizeInMB);
 }
}

void Environment::setCudaGraphOptimization(bool enabled) {
 _cudaGraphOptimization.store(enabled);
}

void Environment::setCudaTensorCoreEnabled(bool enabled) {
 _cudaTensorCoreEnabled.store(enabled);

 // Apply TensorCore settings if the device supports it
 if (_cudaCurrentDevice.load() >= 0 && _cudaCurrentDevice.load() < _cudaDeviceCount.load()) {
   int deviceId = _cudaCurrentDevice.load();
   if (_capabilities[deviceId].first() >= 7) {  // Volta and newer architectures support TensorCores
     // Instead of using attribute directly, use the math mode flags
     // which are more widely supported across CUDA versions
     cudaError_t err;
     if (enabled) {
       // Use the most permissive math mode that allows tensor cores
       err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
       if (err != cudaSuccess) {
         sd_printf("Warning: Failed to set shared memory config for tensor cores, error: %s\n",
                   cudaGetErrorString(err));
       }
     }
   }
 }
}

void Environment::setCudaBlockingSync(int mode) {
 if (mode >= 0 && mode <= 1) {
   _cudaBlockingSync.store(mode);
   cudaSetDeviceFlags(mode == 1 ? cudaDeviceBlockingSync : cudaDeviceScheduleSpin);
 }
}

void Environment::setCudaDeviceSchedule(int schedule) {
 if (schedule >= 0 && schedule <= 3) {
   _cudaDeviceSchedule.store(schedule);

   unsigned int flag;
   switch (schedule) {
     case 1:
       flag = cudaDeviceScheduleSpin;
       break;
     case 2:
       flag = cudaDeviceScheduleYield;
       break;
     case 3:
       flag = cudaDeviceScheduleBlockingSync;
       break;
     case 0:
     default:
       flag = cudaDeviceScheduleAuto;
       break;
   }

   cudaSetDeviceFlags(flag);
 }
}

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
#ifdef __CUDABLAS__
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
}
