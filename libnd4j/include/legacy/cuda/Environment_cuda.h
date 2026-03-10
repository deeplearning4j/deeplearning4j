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

#pragma once

//
// Dispatch header for CUDA API calls extracted from Environment.cpp.
// The .cu implementations live in Environment_cuda.cu.
// CPU builds get inline stubs so callers need no #ifdef guards.
//

#include <system/common.h>
#include <types/pair.h>
#include <vector>

namespace sd {

#ifdef SD_CUDA

/**
 * Perform all CUDA initialization done in the Environment constructor:
 * device enumeration, property queries, BlasVersionHelper, initCudaEnvironment,
 * initCudaDeviceLimits, and setting device 0.
 *
 * @param[out] deviceCount   number of CUDA devices found
 * @param[out] capabilities  vector to receive (major,minor) compute capabilities
 * @param[out] blasMajor     cuBLAS major version
 * @param[out] blasMinor     cuBLAS minor version
 * @param[out] blasPatch     cuBLAS patch version
 */
void Environment_initCuda(int& deviceCount,
                          std::vector<Pair>& capabilities,
                          int& blasMajor, int& blasMinor, int& blasPatch);

/**
 * Implementation of cudaDeviceSetLimit with enum mapping from CudaLimitType.
 * Returns true on success.
 */
bool Environment_setCudaDeviceLimit_cuda(int limitType, size_t value);

/**
 * Query all current CUDA device limits and return them via out-parameters.
 * @param[out] stackSize          cudaLimitStackSize
 * @param[out] mallocHeapSize     cudaLimitMallocHeapSize
 * @param[out] printfFifoSize     cudaLimitPrintfFifoSize
 * @param[out] devRuntimeSyncDepth    cudaLimitDevRuntimeSyncDepth
 * @param[out] devRuntimePendingLaunchCount  cudaLimitDevRuntimePendingLaunchCount
 * @param[out] maxL2FetchGranularity cudaLimitMaxL2FetchGranularity
 * @param[out] persistingL2CacheSize cudaLimitPersistingL2CacheSize (CUDA 10+)
 */
void Environment_queryCudaDeviceLimits(size_t& stackSize,
                                       size_t& mallocHeapSize,
                                       size_t& printfFifoSize,
                                       size_t& devRuntimeSyncDepth,
                                       size_t& devRuntimePendingLaunchCount,
                                       size_t& maxL2FetchGranularity,
                                       size_t& persistingL2CacheSize);

/**
 * Call cudaSetDevice for initCudaEnvironment's SD_CUDA_DEVICE handling.
 * Returns true on success.
 */
bool Environment_cudaSetDeviceForInit(int device);

#else

// CPU stubs
static inline void Environment_initCuda(int& deviceCount,
                                        std::vector<Pair>&,
                                        int&, int&, int&) {
  deviceCount = 0;
}

static inline bool Environment_setCudaDeviceLimit_cuda(int, size_t) {
  return false;
}

static inline void Environment_queryCudaDeviceLimits(size_t&, size_t&, size_t&,
                                                     size_t&, size_t&, size_t&,
                                                     size_t&) {}

static inline bool Environment_cudaSetDeviceForInit(int) {
  return false;
}

#endif

}  // namespace sd
