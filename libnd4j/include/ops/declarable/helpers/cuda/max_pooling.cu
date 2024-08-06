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
//  @author raver119@gmail.com
//
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/max_pooling.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename Z>
static SD_KERNEL void indicesFiller(void* vz, LongType const* zShapeInfo, LongType part, LongType bSize) {
  auto z = reinterpret_cast<Z*>(vz);

  for (LongType b = blockIdx.x; b < bSize; b += gridDim.x) {
    for (LongType e = threadIdx.x; e < part; e += blockDim.x) {
      z[shape::getIndexOffset(e + b * part, zShapeInfo)] = static_cast<Z>(e);
    }
  }
}

template <typename T, typename Y>
static void maxPoolingFunctor_(graph::Context& block, NDArray* input, NDArray* values,
                               std::vector<LongType> const& params, NDArray* indices) {
  LongType kY = params[0];
  LongType kX = params[1];

  LongType sY = params[2];
  LongType sX = params[3];

  LongType pY = params[4];
  LongType pX = params[5];

  LongType dY = params[6];
  LongType dX = params[7];

  LongType oY = 0;
  LongType oX = 0;

  const LongType bSize = input->sizeAt(0);
  const LongType inD = input->sizeAt(1);
  const LongType inY = input->sizeAt(2);
  const LongType inX = input->sizeAt(3);

  const bool isSameMode = params[8] != 0;

  ConvolutionUtils::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

  if (isSameMode)
    ConvolutionUtils::calcPadding2D(pY, pX, oY, oX, inY, inX, params[0], params[1], params[2], params[3], params[6],
                                    params[7]);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 -
  // poolingMode; 9 - divisor;
  ConvolutionUtils::pooling2d(block, *input, *values, kY, kX, sY, sX, pY, pX, dY, dX, MAX_POOL, 1);

  if (nullptr != indices) {
    // for max_pool_with_argmax
    auto total = input->lengthOf();
    auto part = total / bSize;

    indicesFiller<Y><<<256, 256, 1024, *block.launchContext()->getCudaStream()>>>(
        indices->specialBuffer(), indices->specialShapeInfo(), part, bSize);

  sd::DebugHelper::checkErrorCode(block.launchContext()->getCudaStream(), "indicesFiller failed");
  }
}

void maxPoolingFunctor(LaunchContext* context, graph::Context& block, NDArray* input, NDArray* values,
                       std::vector<LongType> const& params, NDArray* indices) {
  NDArray::prepareSpecialUse({values, indices}, {input});
  auto yType = indices == nullptr ? INT64 : indices->dataType();
  BUILD_DOUBLE_SELECTOR(input->dataType(), yType, maxPoolingFunctor_, (block, input, values, params, indices),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({values, indices}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
