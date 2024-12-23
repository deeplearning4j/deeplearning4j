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
// @author GS <sgazeos@gmail.com>, created on 16.01.2019
//
#include <loops/special_kernels.h>
#include <execution/cuda/LaunchDims.h>

    namespace sd {

  template <typename T>
  SD_KERNEL void tileKernel(void const* inputBuffer,
                            LongType const* inputShape,
                            void* outputBuffer,
                            LongType const* outputShape,
                            LongType resultLength) {

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    // Cache shape info to avoid repeated calls
    __shared__ sd::LongType inRank;
    __shared__ const sd::LongType* inShapePtr;
    __shared__ const sd::LongType* inStridePtr;

    __shared__ sd::LongType outRank;
    __shared__ const sd::LongType* outShapePtr;
    __shared__ const sd::LongType* outStridePtr;
    __shared__ char outOrder;

    if (threadIdx.x == 0) {
      inRank      = shape::rank(inputShape);
      inShapePtr  = shape::shapeOf(inputShape);
      inStridePtr = shape::stride(inputShape);

      outRank     = shape::rank(outputShape);
      outShapePtr = shape::shapeOf(outputShape);
      outStridePtr= shape::stride(outputShape);

      outOrder    = shape::order(outputShape);
    }
    __syncthreads();

    const auto inData  = reinterpret_cast<const T*>(inputBuffer);
    auto outData       = reinterpret_cast<T*>(outputBuffer);

    if (outOrder == 'c') {
      // If the output is in 'c' order, we do direct linear indexing in output
      for (LongType i = tid; i < resultLength; i += totalThreads) {
        // We compute the input offset by using the output coordinate
        // to index into the input shape/stride
        sd::LongType coords[SD_MAX_RANK];
        sd::LongType inOffset;

        INDEX2COORDS(i, outRank, outShapePtr, coords);
        COORDS2INDEX(outRank, inStridePtr, coords, inOffset);

        // outData[i] = inData[inOffset]
        // The linear output index is i, so the input is
        // determined by the coords from the output shape
        outData[i] = inData[inOffset];
      }
    }
    else {
      // If the output has some other order, we do a more general coordinate transform
      for (LongType i = tid; i < resultLength; i += totalThreads) {
        // We map the linear index i into coordinates for the output shape
        sd::LongType outCoords[SD_MAX_RANK];
        sd::LongType outOffset;

        INDEX2COORDS(i, outRank, outShapePtr, outCoords);
        COORDS2INDEX(outRank, outStridePtr, outCoords, outOffset);

        // Then we interpret i as an index for the input as well, or use outCoords
        // Actually, the kernel code as written uses the same index i for input coords,
        // but let's remain consistent with the original logic:
        sd::LongType inCoords[SD_MAX_RANK];
        sd::LongType inOffset;

        INDEX2COORDS(i, inRank, inShapePtr, inCoords);
        COORDS2INDEX(inRank, inStridePtr, inCoords, inOffset);

        outData[outOffset] = inData[inOffset];
      }
    }
  }

  // We build specialized versions of tileKernel for all SD_COMMON_TYPES
  BUILD_SINGLE_TEMPLATE(
      template SD_KERNEL void tileKernel,
      (void const* inputBuffer,
       sd::LongType const* inputShape,
       void* outputBuffer,
       sd::LongType const* outputShape,
       sd::LongType resultLength),
      SD_COMMON_TYPES);

  template <typename T>
  void tileKernelH(void const* inputBuffer,
                   LongType const* inputShape,
                   void* outputBuffer,
                   LongType const* outputShape,
                   LongType resultLength,
                   cudaStream_t* stream) {

    dim3 launchDims = getLaunchDims("tile");
    tileKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        inputBuffer, inputShape, outputBuffer, outputShape, resultLength);

    sd::DebugHelper::checkErrorCode(stream, "tileKernel failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void tileKernelH,
      (void const* inputBuffer,
       sd::LongType const* inputShape,
       void* outputBuffer,
       sd::LongType const* outputShape,
       sd::LongType resultLength,
       cudaStream_t* stream),
      SD_COMMON_TYPES);

  // Enhancement for different input (Y) and output (X) data types
  template <typename X, typename Y>
  SD_KERNEL void tileKernelDouble(
      void const* inputBuffer,
      LongType const* inputShape,
      void* outputBuffer,
      LongType const* outputShape,
      LongType resultLength) {

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    __shared__ sd::LongType inRank;
    __shared__ const sd::LongType* inShapePtr;
    __shared__ const sd::LongType* inStridePtr;

    __shared__ sd::LongType outRank;
    __shared__ const sd::LongType* outShapePtr;
    __shared__ const sd::LongType* outStridePtr;
    __shared__ char outOrder;

    if (threadIdx.x == 0) {
      inRank      = shape::rank(inputShape);
      inShapePtr  = shape::shapeOf(inputShape);
      inStridePtr = shape::stride(inputShape);

      outRank     = shape::rank(outputShape);
      outShapePtr = shape::shapeOf(outputShape);
      outStridePtr= shape::stride(outputShape);

      outOrder    = shape::order(outputShape);
    }
    __syncthreads();

    const auto inData  = reinterpret_cast<const Y*>(inputBuffer);
    auto outData       = reinterpret_cast<X*>(outputBuffer);

    if (outOrder == 'c') {
      for (LongType i = tid; i < resultLength; i += totalThreads) {
        // We do direct linear offset for output as i
        // The offset in the input is determined by the out-coords
        // mapped to the input stride
        sd::LongType coords[SD_MAX_RANK];
        sd::LongType inOffset;

        INDEX2COORDS(i, outRank, outShapePtr, coords);
        COORDS2INDEX(outRank, inStridePtr, coords, inOffset);

        outData[i] = static_cast<X>(inData[inOffset]);
      }
    }
    else {
      for (LongType i = tid; i < resultLength; i += totalThreads) {
        sd::LongType outCoords[SD_MAX_RANK];
        sd::LongType outOffset;
        sd::LongType inCoords[SD_MAX_RANK];
        sd::LongType inOffset;

        INDEX2COORDS(i, outRank, outShapePtr, outCoords);
        COORDS2INDEX(outRank, outStridePtr, outCoords, outOffset);

        // The original logic does a symmetrical approach for input.
        // We'll maintain that for consistency:
        INDEX2COORDS(i, inRank, inShapePtr, inCoords);
        COORDS2INDEX(inRank, inStridePtr, inCoords, inOffset);

        outData[outOffset] = static_cast<X>(inData[inOffset]);
      }
    }
  }

  BUILD_SINGLE_TEMPLATE_TWICE(
      template SD_KERNEL void tileKernelDouble,
      (void const* inputBuffer,
       sd::LongType const* inputShape,
       void* outputBuffer,
       sd::LongType const* outputShape,
       sd::LongType resultLength),
      SD_COMMON_TYPES);

  // The host wrapper for tileKernelDouble
  template <typename X, typename Y>
  void tileKernelHH(void const* inputBuffer,
                    LongType const* inputShape,
                    void* outputBuffer,
                    LongType const* outputShape,
                    LongType resultLength,
                    cudaStream_t* stream) {

    dim3 launchDims = getLaunchDims("tile");
    tileKernelDouble<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        inputBuffer, inputShape, outputBuffer, outputShape, resultLength);

    DebugHelper::checkErrorCode(stream, "tileKernelDouble(...) failed");
  }

  BUILD_SINGLE_TEMPLATE_TWICE(
      template void tileKernelHH,
      (void const* inputBuffer,
       sd::LongType const* inputShape,
       void* outputBuffer,
       sd::LongType const* outputShape,
       sd::LongType resultLength,
       cudaStream_t* stream),
      SD_COMMON_TYPES);

}  // namespace sd
