/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author raver119@gmail.com
//
#include <array/CudaPointerDeallocator.h>
#include <memory/cuda/CudaMemoryPool.h>

namespace sd {

void CudaPointerDeallocator::release(void *ptr) {
  if (ptr == nullptr) return;

  // Check if this is a valid device pointer before freeing
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);

  if (result == cudaSuccess) {
    // Only free if it's a device pointer (regular or managed)
    if (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) {
      // CRITICAL FIX: Use the device ID from attributes, not the current device!
      // The pointer was allocated on attributes.device, not necessarily the current device.
      // Using the wrong device ID can cause heap corruption when freeing cross-device pointers.
      int deviceId = attributes.device;
      memory::CudaMemoryPool::getInstance().free(ptr, deviceId, nullptr);
    }
    // Don't free other types (host memory, constant memory, etc.)
  } else {
    // Clear the error and don't try to free this pointer
    cudaGetLastError();
  }
}
}  // namespace sd
