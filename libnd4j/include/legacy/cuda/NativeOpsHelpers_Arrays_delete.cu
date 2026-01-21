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

#include <array/NDArray.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <cuda_runtime.h>

#include <legacy/NativeOps.h>

/**
 * CUDA implementation of deleteNDArray - deletes an NDArray with device synchronization.
 *
 * This function is called from OpaqueNDArrayDeallocator.deallocate() via JNI.
 */
void deleteNDArray(OpaqueNDArray array) {
  if (array == nullptr) {
    return;
  }

  cudaError_t pendingErr = cudaGetLastError();
  if (pendingErr != cudaSuccess) {
    sd_debug("deleteNDArray [CUDA]: Cleared pending CUDA error: %s\n", 0, cudaGetErrorString(pendingErr));
  }

  // Try to get the device ID from the array's data buffer
  // We need to sync the correct device before freeing
  int bufferDeviceId = 0;
  bool gotDeviceId = false;

  // Safely try to get device ID - the array may be corrupted
  // Use try-catch because accessing dataBuffer() can throw if corrupted
#ifdef __cpp_exceptions
  try {
#endif
    if (array->dataBuffer() != nullptr) {
      bufferDeviceId = array->dataBuffer()->deviceId();
      gotDeviceId = true;
    }
#ifdef __cpp_exceptions
  } catch (...) {
    // Array is corrupted, use default device 0
    sd_debug("deleteNDArray [CUDA]: Could not get device ID from array, using device 0%s\n", 0, "");
    gotDeviceId = false;
  }
#endif

  // Get current device to restore later
  int currentDevice = 0;
  cudaGetDevice(&currentDevice);

  if (gotDeviceId && bufferDeviceId != currentDevice) {
    cudaError_t setDevErr = cudaSetDevice(bufferDeviceId);
    if (setDevErr != cudaSuccess) {
      // Can't set device - fall back to syncing ALL devices
      cudaGetLastError();  // Clear the error
      sd_debug("deleteNDArray [CUDA]: Failed to set device %d: %s. Syncing all devices.\n",
               bufferDeviceId, cudaGetErrorString(setDevErr));

      // Sync all devices as fallback - expensive but safe
      int numDevices = 0;
      cudaGetDeviceCount(&numDevices);
      for (int d = 0; d < numDevices; d++) {
        cudaSetDevice(d);
        cudaDeviceSynchronize();
      }
      cudaSetDevice(currentDevice);  // Restore
    }
  }

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    // Sync failed - this is a problem. Log it but we still need to try to clean up.
    // It's safer to proceed than to leak, but the caller should investigate why sync failed.
    sd_debug("deleteNDArray [CUDA]: cudaDeviceSynchronize failed: %s. Proceeding with deletion.\n",
             0, cudaGetErrorString(syncErr));
    cudaGetLastError();  // Clear the error state
  }

  // Now safe to delete the array - all CUDA operations are complete
  delete array;

  // Restore original device context if we switched devices
  if (gotDeviceId && bufferDeviceId != currentDevice) {
    cudaSetDevice(currentDevice);
  }
}
