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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
// @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <ops/declarable/helpers/percentile.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename X>
static SD_KERNEL void percentileKernel(void* vx, const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                       const LongType numTads, const LongType tadLength, void* vz,
                                       const LongType* zShapeInfo, const LongType zLength,
                                       const LongType position) {
  for (int t = blockIdx.x; t < numTads; t += gridDim.x) {
    auto x = reinterpret_cast<X*>(vx) + xTadOffsets[t];
    auto z = reinterpret_cast<X*>(vz);

    LongType xCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];
    LongType t0, t1, zOffset, positionOffset;

    // sort tad
    if (tadLength > 1) {
      for (int m = 0; m < tadLength; m++) {
        if (m % 2 == 0) {
          for (int tid = threadIdx.x; tid < tadLength; tid += blockDim.x) {
            auto top = 2 * tid + 1;
            if (top < tadLength) {
              INDEX2COORDS(top - 1, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords);
              COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), xCoords, t0);
              INDEX2COORDS(top, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords);
              COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), xCoords, t1);

              if (x[t0] > x[t1]) {
                // swap values
                X dz0 = x[t0];
                x[t0] = x[t1];
                x[t1] = dz0;
              }
            }
          }
        } else {
          for (int tid = threadIdx.x; tid < tadLength; tid += blockDim.x) {
            auto top = 2 * tid + 2;
            if (top < tadLength) {
              INDEX2COORDS(top - 1, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords);
              COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), xCoords, t0);
              INDEX2COORDS(top, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords);
              COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), xCoords, t1);

              if (x[t0] > x[t1]) {
                // swap values
                X dz0 = x[t0];
                x[t0] = x[t1];
                x[t1] = dz0;
              }
            }
          }
        }
        __syncthreads();
      }
    }

    // saving final value
    if (threadIdx.x == 0) {
      INDEX2COORDS(t, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords);
      COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), zCoords, zOffset);
      INDEX2COORDS(position, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords);
      COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), xCoords, positionOffset);
      z[zOffset] = x[positionOffset];
    }
    __syncthreads();
  }
}

template <typename T>
static void _percentile(LaunchContext* context, NDArray& input, NDArray& output, std::vector<LongType>& axis,
                        const float q, const int interpolation) {
  const int inputRank = input.rankOf();

  if (axis.empty())
    for (int i = 0; i < inputRank; ++i) axis.push_back(i);
  else
    shape::checkDimensions(inputRank, &axis);

  auto tempArray = input.dup();
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(tempArray.shapeInfo(), &axis);

  auto tadLength = shape::length(packX->primaryShapeInfo());

  const float fraction = 1.f - q / 100.;
  LongType position = 0;

  switch (interpolation) {
    case 0:  // lower
      position = static_cast<LongType>(math::sd_ceil<float, T>((tadLength - 1) * fraction));
      break;
    case 1:  // higher
      position = static_cast<LongType>(math::sd_floor<float, T>((tadLength - 1) * fraction));
      break;
    case 2:  // nearest
      position = static_cast<LongType>(math::sd_round<float, T>((tadLength - 1) * fraction));
      break;
  }
  position = tadLength - position - 1;

  dim3 launchDims = getLaunchDims("percentile");
  percentileKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
      tempArray.specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), packX->numberOfTads(), tadLength,
      output.specialBuffer(), output.specialShapeInfo(), output.lengthOf(), position);

  DebugHelper::checkErrorCode(context->getCudaStream(), "percentile");
}

void percentile(LaunchContext* context, NDArray& input, NDArray& output, std::vector<LongType>& axises,
                const float q, const int interpolation) {
  NDArray::prepareSpecialUse({&output}, {&input});

  BUILD_SINGLE_SELECTOR(input.dataType(), _percentile, (context, input, output, axises, q, interpolation),
                        SD_COMMON_TYPES);

  NDArray::registerSpecialUse({&output}, {&input});
}

BUILD_SINGLE_TEMPLATE(template void _percentile,
                      (sd::LaunchContext * context, NDArray& input, NDArray& output, std::vector<sd::LongType>& axises,
                       const float q, const int interpolation),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
