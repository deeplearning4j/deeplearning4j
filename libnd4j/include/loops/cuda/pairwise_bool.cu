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

//  @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018

#ifndef PAIRWISE_BOOL_CU
#define PAIRWISE_BOOL_CU

#include "../pairwise_bool.h"


using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
SD_KERNEL static void pairwiseSimpleShaped(void const* vx, sd::LongType const* xShapeInfo, void const* vy,
                                           sd::LongType const* yShapeInfo, void* vz, sd::LongType const* zShapeInfo,
                                           void* vextraParams) {
  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ sd::LongType len;

  if (threadIdx.x == 0) {
    len = shape::length(xShapeInfo);
  }
  __syncthreads();

  for (sd::LongType i = tid; i < len; i += gridDim.x * blockDim.x) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType yCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset;
    sd::LongType yOffset;
    sd::LongType zOffset;

    INDEX2COORDS(i, shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), xCoords);
    COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), xCoords, xOffset);
    INDEX2COORDS(i, shape::rank(yShapeInfo), shape::shapeOf(yShapeInfo), yCoords);
    COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), yCoords, yOffset);
    INDEX2COORDS(i, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), zCoords, zOffset);

    z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
  }
}

namespace functions {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void SD_HOST PairWiseBoolTransform<X, Z>::intermediateShaped(dim3& launchDims, cudaStream_t* stream, void const* vx,
                                                             sd::LongType const* xShapeInfo, void const* vy,
                                                             sd::LongType const* yShapeInfo, void* vz,
                                                             sd::LongType const* zShapeInfo, void* vextraParams) {
  pairwiseSimpleShaped<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams);
  sd::DebugHelper::checkErrorCode(stream, "PairWiseBoolTransform intermediateShaped(...) failed");

}

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void PairWiseBoolTransform<X, Y>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx,
                                                    sd::LongType const* xShapeInfo, void const* vy,
                                                    sd::LongType const* yShapeInfo, void* vz,
                                                    sd::LongType const* zShapeInfo, void* vextraParams) {
  auto xType = sd::DataTypeUtils::fromT<X>();
  auto yType = sd::DataTypeUtils::fromT<Y>();

  DISPATCH_BY_OPNUM_TT(intermediateShaped,
                       PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams),
                       PAIRWISE_BOOL_OPS);
}

BUILD_DOUBLE_TEMPLATE(template class PairWiseBoolTransform, , SD_COMMON_TYPES, SD_BOOL_TYPES);
}  // namespace pairwise_transforms
}  // namespace functions

#endif  // PAIRWISE_BOOL_CU
