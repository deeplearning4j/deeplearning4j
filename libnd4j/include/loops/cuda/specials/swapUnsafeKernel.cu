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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author GS <sgazeos@gmail.com>, created on 25.01.2019
//
#include <loops/special_kernels.h>
#include <execution/cuda/LaunchDims.h>

    namespace sd {

  template <typename T>
  SD_KERNEL void swapUnsafeKernel(
      void* theFirstBuffer,
      LongType const* theFirstShape,
      void* theSecondBuffer,
      LongType const* theSecondShape) {

    // thread & grid info
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    // cache relevant info in shared memory
    __shared__ bool sameOffsets, sameOrders;
    __shared__ sd::LongType resultLength;

    __shared__ sd::LongType firstRank;
    __shared__ const sd::LongType* firstShapePtr;
    __shared__ const sd::LongType* firstStridePtr;

    __shared__ sd::LongType secondRank;
    __shared__ const sd::LongType* secondShapePtr;
    __shared__ const sd::LongType* secondStridePtr;

    __shared__ T* outPtr;
    __shared__ T* inPtr;

    if (threadIdx.x == 0) {
      resultLength   = shape::length(theFirstShape);

      // first shape/stride info
      firstRank      = shape::rank(theFirstShape);
      firstShapePtr  = shape::shapeOf(theFirstShape);
      firstStridePtr = shape::stride(theFirstShape);

      // second shape/stride info
      secondRank      = shape::rank(theSecondShape);
      secondShapePtr  = shape::shapeOf(theSecondShape);
      secondStridePtr = shape::stride(theSecondShape);

      outPtr          = reinterpret_cast<T*>(theFirstBuffer);
      inPtr           = reinterpret_cast<T*>(theSecondBuffer);

      sameOffsets     = shape::haveSameShapeAndStrides(theFirstShape, theSecondShape);
      sameOrders      = (shape::order(theFirstShape) == shape::order(theSecondShape));
    }
    __syncthreads();

    for (sd::LongType i = tid; i < resultLength; i += totalThreads) {
      sd::LongType firstCoords[SD_MAX_RANK];
      sd::LongType secondCoords[SD_MAX_RANK];

      // offsets in each array
      sd::LongType firstOffset;
      sd::LongType secondOffset;

      // compute coordinates in the first array
      INDEX2COORDS(i, firstRank, firstShapePtr, firstCoords);
      COORDS2INDEX(firstRank, firstStridePtr, firstCoords, firstOffset);

      // compute coordinates in the second array
      INDEX2COORDS(i, secondRank, secondShapePtr, secondCoords);
      COORDS2INDEX(secondRank, secondStridePtr, secondCoords, secondOffset);

      if (sameOrders && firstOffset >= 0 && secondOffset >= 0) {
        // direct swap with the known offsets
        math::sd_swap(outPtr[firstOffset], inPtr[secondOffset]);
      }
      else if (sameOffsets) {
        // same shape/strides => same offset for both
        math::sd_swap(outPtr[firstOffset], inPtr[firstOffset]);
      }
      else {
        math::sd_swap(outPtr[firstOffset], inPtr[secondOffset]);
      }
    }
  }

  BUILD_SINGLE_TEMPLATE(
      template SD_KERNEL void swapUnsafeKernel,
      (void* theFirstBuffer,
       sd::LongType const* theFirstShape,
       void* theSecondBuffer,
       sd::LongType const* theSecondShape),
      SD_COMMON_TYPES);

  template <typename T>
  void templatedSwapUnsafe(
      void* theFirstBuffer,
      LongType const* theFirstShape,
      void* theSecondBuffer,
      LongType const* theSecondShape,
      cudaStream_t* theStream) {

    dim3 launchDims = getLaunchDims("swap_unsafe");

    swapUnsafeKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *theStream>>>(
        theFirstBuffer,
        theFirstShape,
        theSecondBuffer,
        theSecondShape);

    DebugHelper::checkGlobalErrorCode("templatedSwapUnsafe(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void templatedSwapUnsafe,
      (void* theFirstBuffer,
       sd::LongType const* theFirstShape,
       void* theSecondBuffer,
       sd::LongType const* theSecondShape,
       cudaStream_t* theStream),
      SD_COMMON_TYPES);

}  // namespace sd
