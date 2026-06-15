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
#include <system/env_functions.h>
#include <atomic>
#include <legacy/NativeOps.h>


// OpaqueNDArray allocation tracking - defined in NativeOpsHelpers_Arrays.cpp
extern std::atomic<size_t> g_opaqueArrayCount;
extern std::atomic<size_t> g_opaqueArrayBytes;


/**
 * CPU implementation of deleteNDArray - deletes an NDArray.
 * On CPU, no device synchronization is needed since all operations
 * are synchronous.
 *
 * This function is called from OpaqueNDArrayDeallocator.deallocate() via JNI.
 */
void deleteNDArray(OpaqueNDArray array) {
  if (array == nullptr) {
    return;
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    sd_printf("deleteNDArray: deleting NDArray at %p, shapeInfo=%p\n",
              array, array->shapeInfo());
    fflush(stdout);
  }

  g_opaqueArrayCount.fetch_sub(1, std::memory_order_relaxed);

  // On CPU, just delete the array directly - all operations are synchronous
  delete array;

  if(sd::env_isLogNativeNDArrayCreation()) {
    sd_printf("deleteNDArray: successfully deleted NDArray\n");
    fflush(stdout);
  }
}
