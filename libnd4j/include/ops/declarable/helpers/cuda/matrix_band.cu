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
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/matrix_band.h>

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
static SD_KERNEL void matrixBandKernel(const void* inputBuffer, const sd::LongType* inputShape, void* outputBuffer,
                                       const sd::LongType* outputShape, sd::LongType lowerBand, sd::LongType upperBand,
                                       const sd::LongType* tadOnlyInputShapeInfo, const sd::LongType* tadInputOffsets,
                                       const sd::LongType* tadOnlyOutputShapeInfo, const sd::LongType* tadOutputOffsets,
                                       sd::LongType numTads, sd::LongType inputLength) {
  int totalThreads = blockDim.x;
  sd::LongType rows = shape::sizeAt(inputShape, -2);
  sd::LongType cols = shape::sizeAt(inputShape, -1);
  for (sd::LongType e = blockIdx.x; e < numTads; e += gridDim.x) {
    auto yOffset = tadInputOffsets[e];
    auto xOffset = tadOutputOffsets[e];
    for (sd::LongType i = blockIdx.y; i < rows; i += gridDim.y) {
      for (sd::LongType j = threadIdx.x; j < cols; j += totalThreads) {
        sd::LongType coords[2] = {i, j};
        sd::LongType tadOffsetOut = shape::getOffset(tadOnlyOutputShapeInfo, coords);
        sd::LongType tadOffsetIn = shape::getOffset(tadOnlyInputShapeInfo, coords);

        if (i >= j) {  // check lower diagonals
          if (lowerBand > 0) {
            if ((i - j) > lowerBand)
              *(reinterpret_cast<T*>(outputBuffer) + xOffset + tadOffsetOut) = T(0);
            else
              *(reinterpret_cast<T*>(outputBuffer) + xOffset + tadOffsetOut) =
                  *(reinterpret_cast<T const*>(inputBuffer) + yOffset + tadOffsetIn);
          }
        } else if (j > i) {
          if (upperBand > 0)
            if ((j - i) > upperBand)
              *(reinterpret_cast<T*>(outputBuffer) + xOffset + tadOffsetOut) = T(0);
            else
              *(reinterpret_cast<T*>(outputBuffer) + xOffset + tadOffsetOut) =
                  *(reinterpret_cast<T const*>(inputBuffer) + yOffset + tadOffsetIn);
        }
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// matrixBandPart_ - main algorithm caller
//
template <typename T>
void matrixBandPart_(sd::LaunchContext* context, NDArray* input, NDArray* output, sd::LongType lowerBand,
                     sd::LongType upperBand) {
  dim3 launchDims(256, 512, 8192);
  auto stream = context->getCudaStream();

  std::vector<int> lastDims({input->rankOf() - 2, input->rankOf() - 1});
  std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), lastDims);

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), lastDims);
  auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), lastDims);

  const sd::LongType numTads = packX.numberOfTads();

  NDArray::prepareSpecialUse({output}, {input});
  matrixBandKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), lowerBand,
      upperBand, packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(),
      numTads, input->lengthOf());
  NDArray::registerSpecialUse({output}, {input});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void matrixBandPart(sd::LaunchContext* context, NDArray* input, NDArray* output, sd::LongType lowerBand,
                    sd::LongType upperBand) {
  BUILD_SINGLE_SELECTOR(input->dataType(), matrixBandPart_, (context, input, output, lowerBand, upperBand),
                        SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
