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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/matrix_band.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// matrix band kernel
//
// inputBuffer - buffer of input tensor
// inputShape - shape of input tensor
// outputBuffer - buffer of output tensor
// outputShape - shape of output tensor
// lowerBand - lower band of matrix
// upperBand - upper band of matrix
// tadOnlyInputShapeInfo - TAD shape for input
// tadInputOffsets - TAD offsets for input
// tadOnlyOutputShapeInfo - TAD output shape
// tadOutputOffsets - TAD output offsets
// numTads - number of subarrays
// inputLength - input subarray length
//
template <typename T>
static SD_KERNEL void matrixBandKernel(const void* inputBuffer, const LongType* inputShape, void* outputBuffer,
                                       const LongType* outputShape, LongType lowerBand, LongType upperBand,
                                       const LongType* tadOnlyInputShapeInfo, const LongType* tadInputOffsets,
                                       const LongType* tadOnlyOutputShapeInfo, const LongType* tadOutputOffsets,
                                       LongType numTads, LongType inputLength) {
  int totalThreads = blockDim.x;
  LongType rows = shape::sizeAt(inputShape, -2);
  LongType cols = shape::sizeAt(inputShape, -1);
  auto resetBuffer = reinterpret_cast<T *>(outputBuffer);
  auto input = reinterpret_cast<T const *>(inputBuffer);

  for (LongType e = blockIdx.x; e < numTads; e += gridDim.x) {
    auto yOffset = tadInputOffsets[e];
    auto xOffset = tadOutputOffsets[e];
    if (outputBuffer != inputBuffer)  // if not inplace
      for(int i = 0; i < inputLength; i++) {
        resetBuffer[i] = input[i];
      }
    for (LongType i = blockIdx.y; i < rows; i += gridDim.y) {
      for (LongType j = threadIdx.x; j < cols; j += totalThreads) {
        LongType coords[2] = {i, j};
        LongType tadOffsetOut = shape::getOffset(tadOnlyOutputShapeInfo, coords);
        LongType tadOffsetIn = shape::getOffset(tadOnlyInputShapeInfo, coords);

        // If not inplace, copy the input to the output
        *(resetBuffer + xOffset + tadOffsetOut) = *(input + yOffset + tadOffsetIn);

        // Check the lower diagonals
        if (lowerBand >= 0 && (i - j) > lowerBand)
          *(resetBuffer + xOffset + tadOffsetOut) = T(0);

        // Check the upper diagonals
        if (upperBand >= 0 && (j - i) > upperBand)
          *(resetBuffer + xOffset + tadOffsetOut) = T(0);
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// matrixBandPart_ - main algorithm caller
//
template <typename T>
void matrixBandPart_(LaunchContext* context, NDArray* input, NDArray* output, LongType lowerBand, LongType upperBand) {
  dim3 launchDims = getLaunchDims("matrixBand");
  auto stream = context->getCudaStream();

  std::vector<LongType> lastDims({input->rankOf() - 2, input->rankOf() - 1});
  std::vector<LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), lastDims.size(),lastDims.data());

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &lastDims);
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &lastDims);

  const LongType numTads = packX->numberOfTads();

  NDArray::prepareSpecialUse({output}, {input});
  matrixBandKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), lowerBand,
      upperBand, packX->specialShapeInfo(), packX->specialOffsets(), packZ->specialShapeInfo(), packZ->specialOffsets(),
      numTads, input->lengthOf());
  sd::DebugHelper::checkErrorCode(stream, "matrixBandKernel failed");

  NDArray::registerSpecialUse({output}, {input});

  delete dimsToExclude;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void matrixBandPart(LaunchContext* context, NDArray* input, NDArray* output, LongType lowerBand, LongType upperBand) {
  BUILD_SINGLE_SELECTOR(input->dataType(), matrixBandPart_, (context, input, output, lowerBand, upperBand),
                        SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
