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
// Split from Environment.cpp to reduce object file size
// Contains: CUDA configuration setter methods
//

#include <system/Environment.h>
#include <helpers/logger.h>
#include <legacy/cuda/Environment_CudaConfig_cuda.h>

namespace sd {

void Environment::setCudaCurrentDevice(int device) {
  if (Environment_setCudaCurrentDevice_cuda(device, _cudaDeviceCount.load())) {
    _cudaCurrentDevice.store(device);
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

void Environment::setCudaPinnedHostLimit(int64_t limitInMB) {
  if (limitInMB >= 0) {
    _cudaPinnedHostLimit.store(limitInMB);
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
#ifdef SD_CUDA
  _cudaTensorCoreEnabled.store(enabled);

  int deviceId = _cudaCurrentDevice.load();
  int deviceCount = _cudaDeviceCount.load();
  int computeMajor = (deviceId >= 0 && deviceId < deviceCount) ? _capabilities[deviceId].first() : 0;
  Environment_setCudaTensorCoreEnabled_cuda(enabled, deviceId, deviceCount, computeMajor);
#endif
}

void Environment::setCudaBlockingSync(int mode) {
#ifdef SD_CUDA
  if (mode >= 0 && mode <= 1) {
    _cudaBlockingSync.store(mode);
    Environment_setCudaBlockingSync_cuda(mode);
  }
#endif
}

void Environment::setCudaDeviceSchedule(int schedule) {
#ifdef SD_CUDA
  if (schedule >= 0 && schedule <= 3) {
    _cudaDeviceSchedule.store(schedule);
    Environment_setCudaDeviceSchedule_cuda(schedule);
  }
#endif
}

}  // namespace sd
