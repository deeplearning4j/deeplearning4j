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

namespace sd {

void CudaPointerDeallocator::release(void *ptr) {
  if (ptr == nullptr) return;

  // Check if this is a valid device pointer before freeing
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);

  if (result == cudaSuccess) {
    // Only free if it's a regular device pointer
    // cudaMemoryTypeDevice is for regular allocations we can free
    if (attributes.type == cudaMemoryTypeDevice) {
      cudaFree(ptr);
    }
    // Don't free other types (like constant memory)
  } else {
    // Clear the error and don't try to free this pointer
    cudaGetLastError(); // Clear the error state
  }
}
}  // namespace sd
