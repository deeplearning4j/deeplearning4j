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

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace sd {

void Environment::setCudaCurrentDevice(int device) {
#ifdef SD_CUDA
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
#endif
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

  if (_cudaCurrentDevice.load() >= 0 && _cudaCurrentDevice.load() < _cudaDeviceCount.load()) {
    int deviceId = _cudaCurrentDevice.load();
    if (_capabilities[deviceId].first() >= 7) {
      cudaError_t err;
      if (enabled) {
        err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (err != cudaSuccess) {
          sd_printf("Warning: Failed to set shared memory config for tensor cores, error: %s\n",
                    cudaGetErrorString(err));
        }
      }
    }
  }
#endif
}

void Environment::setCudaBlockingSync(int mode) {
#ifdef SD_CUDA
  if (mode >= 0 && mode <= 1) {
    _cudaBlockingSync.store(mode);
    cudaSetDeviceFlags(mode == 1 ? cudaDeviceBlockingSync : cudaDeviceScheduleSpin);
  }
#endif
}

void Environment::setCudaDeviceSchedule(int schedule) {
#ifdef SD_CUDA
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
#endif
}

}  // namespace sd
