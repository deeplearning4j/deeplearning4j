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
// CUDA API implementations extracted from Environment.cpp.
// These functions wrap all CUDA runtime calls so that Environment.cpp
// can be compiled purely by the host compiler (gcc/clang).
//

#ifdef SD_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/logger.h>
#include <system/CudaLimitType.h>
#include <system/BlasVersionHelper.h>
#include <legacy/cuda/Environment_cuda.h>

namespace sd {

// ---------------------------------------------------------------------------
// Constructor-phase CUDA initialisation
// ---------------------------------------------------------------------------
void Environment_initCuda(int& deviceCount,
                          std::vector<Pair>& capabilities,
                          int& blasMajor, int& blasMinor, int& blasPatch) {
  // --- device enumeration ---
  int devCnt = 0;
  cudaError_t err = cudaGetDeviceCount(&devCnt);

  if (err != cudaSuccess) {
    sd_printf("Environment_initCuda: ERROR - cudaGetDeviceCount failed: %s\n",
              cudaGetErrorString(err));
    devCnt = 0;
  }

  if (devCnt < 0 || devCnt > 128) {
    sd_printf("Environment_initCuda: ERROR - Invalid device count %d, limiting to 0\n", devCnt);
    devCnt = 0;
  }

  deviceCount = devCnt;
  sd_printf("During environment initialization we found [%i] CUDA devices\n", devCnt);

  // --- device properties ---
  if (devCnt > 0) {
    cudaDeviceProp* devProperties = nullptr;
    try {
      devProperties = new cudaDeviceProp[devCnt];
    } catch (...) {
      sd_printf("Environment_initCuda: ERROR - Failed to allocate device properties array\n");
      devCnt = 0;
    }

    if (devProperties != nullptr) {
      for (int i = 0; i < devCnt; i++) {
        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
          sd_printf("Environment_initCuda: WARNING - cudaSetDevice(%d) failed: %s\n",
                    i, cudaGetErrorString(err));
          continue;
        }

        err = cudaGetDeviceProperties(&devProperties[i], i);
        if (err != cudaSuccess) {
          sd_printf("Environment_initCuda: WARNING - cudaGetDeviceProperties(%d) failed: %s\n",
                    i, cudaGetErrorString(err));
          continue;
        }

        if (devProperties[i].major >= 0 && devProperties[i].minor >= 0) {
          try {
            Pair p(devProperties[i].major, devProperties[i].minor);
            capabilities.emplace_back(p);
          } catch (...) {
            sd_printf("Environment_initCuda: WARNING - Failed to add capability for device %d\n", i);
          }
        }
      }
      delete[] devProperties;
    }
  }

  // --- BLAS version ---
  try {
    BlasVersionHelper ver;
    blasMajor = ver._blasMajorVersion;
    blasMinor = ver._blasMinorVersion;
    blasPatch = ver._blasPatchVersion;
  } catch (...) {
    sd_printf("Environment_initCuda: WARNING - BlasVersionHelper initialization failed\n");
    blasMajor = 0;
    blasMinor = 0;
    blasPatch = 0;
  }

  // --- set device 0 ---
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    sd_printf("Environment_initCuda: WARNING - cudaSetDevice(0) failed: %s\n",
              cudaGetErrorString(err));
  }
}

// ---------------------------------------------------------------------------
// setCudaDeviceLimit — maps CudaLimitType enum to cudaLimit and calls
// cudaDeviceSetLimit.
// ---------------------------------------------------------------------------
bool Environment_setCudaDeviceLimit_cuda(int limitType, size_t value) {
  CudaLimitType lt = static_cast<CudaLimitType>(limitType);
  cudaLimit cudaLimitValue;

  switch (lt) {
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
    sd_printf("Warning: Failed to set CUDA device limit, error: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// queryCudaDeviceLimits — reads all CUDA device limits via cudaDeviceGetLimit.
// ---------------------------------------------------------------------------
void Environment_queryCudaDeviceLimits(size_t& stackSize,
                                       size_t& mallocHeapSize,
                                       size_t& printfFifoSize,
                                       size_t& devRuntimeSyncDepth,
                                       size_t& devRuntimePendingLaunchCount,
                                       size_t& maxL2FetchGranularity,
                                       size_t& persistingL2CacheSize) {
  size_t value;
  if (cudaDeviceGetLimit(&value, cudaLimitStackSize) == cudaSuccess)
    stackSize = value;

  if (cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize) == cudaSuccess)
    mallocHeapSize = value;

  if (cudaDeviceGetLimit(&value, cudaLimitPrintfFifoSize) == cudaSuccess)
    printfFifoSize = value;

  if (cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth) == cudaSuccess)
    devRuntimeSyncDepth = value;

  if (cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount) == cudaSuccess)
    devRuntimePendingLaunchCount = value;

  if (cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity) == cudaSuccess)
    maxL2FetchGranularity = value;

#if CUDART_VERSION >= 10000
  if (cudaDeviceGetLimit(&value, cudaLimitPersistingL2CacheSize) == cudaSuccess)
    persistingL2CacheSize = value;
#endif
}

// ---------------------------------------------------------------------------
// cudaSetDevice wrapper for initCudaEnvironment
// ---------------------------------------------------------------------------
bool Environment_cudaSetDeviceForInit(int device) {
  cudaError_t err = cudaSetDevice(device);
  return (err == cudaSuccess);
}

}  // namespace sd

#endif  // SD_CUDA
