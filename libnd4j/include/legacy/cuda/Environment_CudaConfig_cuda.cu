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
// CUDA implementations of Environment CUDA-config setter helpers.
// Split from Environment_CudaConfig.cpp so that CUDA API calls live
// exclusively in .cu translation units compiled by nvcc.
//

#ifdef SD_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/logger.h>
#include <legacy/cuda/Environment_CudaConfig_cuda.h>

namespace sd {

bool Environment_setCudaCurrentDevice_cuda(int deviceId, int deviceCount) {
  if (deviceId >= 0 && deviceId < deviceCount) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err == cudaSuccess) {
      return true;
    } else {
      sd_printf("Warning: Failed to set CUDA device to %d, error: %s\n", deviceId, cudaGetErrorString(err));
      return false;
    }
  } else {
    sd_printf("Warning: Attempted to set invalid CUDA device %d (valid range: 0-%d)\n", deviceId, deviceCount - 1);
    return false;
  }
}

bool Environment_setCudaTensorCoreEnabled_cuda(bool enabled, int deviceId, int deviceCount, int computeMajor) {
  if (deviceId >= 0 && deviceId < deviceCount) {
    if (computeMajor >= 7) {
      if (enabled) {
        cudaError_t err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (err != cudaSuccess) {
          sd_printf("Warning: Failed to set shared memory config for tensor cores, error: %s\n",
                    cudaGetErrorString(err));
          return false;
        }
      }
    }
  }
  return true;
}

void Environment_setCudaBlockingSync_cuda(int mode) {
  if (mode >= 0 && mode <= 1) {
    cudaSetDeviceFlags(mode == 1 ? cudaDeviceBlockingSync : cudaDeviceScheduleSpin);
  }
}

void Environment_setCudaDeviceSchedule_cuda(int schedule) {
  if (schedule >= 0 && schedule <= 3) {
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

}  // namespace sd

#endif  // SD_CUDA
