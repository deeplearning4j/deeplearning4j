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
//  @author GS <sgazeos@gmail.com>
//
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/sequence_mask.h>


#include "helpers/DebugHelper.h"
namespace sd {
namespace ops {
namespace helpers {

template <typename I, typename B>
static SD_KERNEL void sequenceMaskKernel(const void* inputBuf, const LongType* inputShape, void* outputBuf,
                                         const LongType* outputShape, int maxIndex) {
  __shared__ const I* input;
  __shared__ B* output;
  __shared__ LongType inputLen, outputLen;

  // Cache shape information
  __shared__ sd::LongType inputRank, outputRank;
  __shared__ const sd::LongType* inputShapePtr;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* inputStridePtr;
  __shared__ const sd::LongType* outputStridePtr;

  if (threadIdx.x == 0) {
    input = reinterpret_cast<const I*>(inputBuf);
    output = reinterpret_cast<B*>(outputBuf);
    inputLen = shape::length(inputShape);
    outputLen = shape::length(outputShape);

    // Cache shape information
    inputRank = shape::rank(inputShape);
    outputRank = shape::rank(outputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType inputOffset;
  LongType outputOffset;

  for (auto i = blockIdx.x; i < maxIndex; i += gridDim.x)
    for (auto k = threadIdx.x; k < inputLen; k += blockDim.x) {
      INDEX2COORDS(k, inputRank, inputShapePtr, inputCoords);
      COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);
      if (i < input[inputOffset]) {
        INDEX2COORDS(k * maxIndex + i, outputRank, outputShapePtr, outputCoords);
        COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);
        output[outputOffset] = B(true);
      }
    }
}
template <typename I, typename B>
static void sequenceMask_(LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
 dim3 launchDims = getSequenceMaskLaunchDims(maxIndex,*input);
 NDArray::prepareSpecialUse({output}, {input});
 auto stream = context->getCudaStream();
 sequenceMaskKernel<I, B><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
     input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), maxIndex);
 sd::DebugHelper::checkErrorCode(stream, "sequenceMaskKernel failed");

 NDArray::registerSpecialUse({output}, {input});
}

void sequenceMask(LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
 BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sequenceMask_, (context, input, output, maxIndex),
                       SD_INTEGER_TYPES, SD_COMMON_TYPES_EXTENDED);
}

BUILD_DOUBLE_TEMPLATE(template void sequenceMask_,
                     (sd::LaunchContext * context, NDArray* input, NDArray* output, int maxIndex), SD_INTEGER_TYPES,
                     SD_COMMON_TYPES_EXTENDED);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
