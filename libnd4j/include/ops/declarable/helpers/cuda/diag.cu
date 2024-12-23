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
// Created by GS <sgazeos@gmail.com> on 4/6/2018.
//
#include <array/ResultSet.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/diag.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diag functor cuda kernel
// outputBuffer - output tensor buffer
// outputShape - output tensor shape
// inputBuffer - input tensor buffer - this tensor should be placed on diagonal position of output
// inputShape - input tensor shape
// inputLength - length for input tensor
//
template <typename T>
static SD_KERNEL void diagFunctorKernel(void* outputBuffer, const LongType* outputShape, void const* inputBuffer,
                                        const LongType* inputShape, LongType inputLength) {
  __shared__ T* z;
  __shared__ T const* x;
  __shared__ LongType outputRank, inputRank, outputLength;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;

  if (threadIdx.x == 0) {
    z = reinterpret_cast<T*>(outputBuffer);
    x = reinterpret_cast<T const*>(inputBuffer);

    outputRank = shape::rank(outputShape);
    inputRank = shape::rank(inputShape);
    outputLength = shape::length(outputShape);

    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  LongType zCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];
  LongType zOffset;
  LongType xOffset;

  for (LongType t = tid; t < inputLength; t += step) {
    // Compute coordinates and offsets for output
    INDEX2COORDS(t * (inputLength + 1), outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

    // Compute coordinates and offsets for input
    INDEX2COORDS(t, inputRank, inputShapePtr, xCoords);
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

    // Assign the value to the diagonal position
    z[zOffset] = x[xOffset];
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diag part functor cuda kernel
// outputBuffer - output tensor buffer - linear sequence of diagonal values
// outputShape - output tensor shape
// inputBuffer - input tensor buffer - this tensor should be placed on diagonal position of output
// inputShape - input tensor shape
// outputLength - given length of output
// inputLength - given length for input tensor
//
template <typename T>
static SD_KERNEL void diagPartFunctorKernel(void* outputBuffer, const LongType* outputShape,
                                            void const* inputBuffer, const LongType* inputShape, LongType outputLength, LongType inputLength) {
  __shared__ T* z;
  __shared__ T const* x;
  __shared__ LongType outputRank, inputRank;
  __shared__ const LongType *outputShapePtr, *outputStridePtr;
  __shared__ const LongType *inputShapePtr, *inputStridePtr;

  if (threadIdx.x == 0) {
    z = reinterpret_cast<T*>(outputBuffer);
    x = reinterpret_cast<T const*>(inputBuffer);

    outputRank = shape::rank(outputShape);
    inputRank = shape::rank(inputShape);

    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  LongType zCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];
  LongType zOffset;
  LongType xOffset;
  LongType i = tid * (outputLength + 1);  // position of the diagonal value in the input

  for (LongType t = tid; t < outputLength && i < inputLength; t += step) {
    // Compute coordinates and offsets for output
    INDEX2COORDS(t, outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

    // Compute coordinates and offsets for input
    INDEX2COORDS(i, inputRank, inputShapePtr, xCoords);
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

    // Assign diagonal value
    z[zOffset] = x[xOffset];

    // Move to the next diagonal value
    i += outputLength + 1;
  }
}

//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page:
// https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
static void _diagFunctor(LaunchContext* context, NDArray* input, NDArray* output) {
  auto stream = context->getCudaStream();
  auto inputLength = input->isScalar() ? 1 : input->lengthOf();
  dim3 launchDims = getLaunchDims("diagPart");
  if (!input->isActualOnDeviceSide()) input->syncToDevice();
  diagFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      output->specialBuffer(), output->specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(),
      inputLength);
  DebugHelper::checkErrorCode(stream,"diagFunctorKernel failed");

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagFunctor - caller for diag functor processor
void diagFunctor(LaunchContext* context, NDArray* input, NDArray* output) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _diagFunctor, (context, input, output), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void _diagFunctor, (sd::LaunchContext * context, NDArray* input, NDArray* output);
                      , SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagPartFunctor - caller for diag part functor kernel
template <typename T>
void _diagPartFunctor(LaunchContext* context, NDArray * input, NDArray* output) {
  const int outLen = output->lengthOf();
  const int inLen = input->isScalar() ? 1 : input->lengthOf();
  auto stream = context->getCudaStream();

  dim3 launchDims = getLaunchDims("diagPart");
  if (!input->isActualOnDeviceSide()) input->syncToDevice();

  diagPartFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      output->specialBuffer(), output->specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), outLen,
      inLen);
  DebugHelper::checkErrorCode(stream,"diagFunctorKernel failed");

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagPartFunctor - caller for diag part functor processor
void diagPartFunctor(LaunchContext* context, NDArray * input, NDArray* output) {
  auto zType = output->dataType();
  BUILD_SINGLE_SELECTOR(zType, _diagPartFunctor, (context, input, output), SD_NUMERIC_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
