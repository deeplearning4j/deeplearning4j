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

#ifndef PAIRWISE_INT_CU
#define PAIRWISE_INT_CU

#include "../pairwise_int.h"

using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
SD_KERNEL static void pairwiseSimpleShaped(void const* vx, sd::LongType const* xShapeInfo, void const* vy,
                                           sd::LongType const* yShapeInfo, void* vz, sd::LongType const* zShapeInfo,
                                           void* vextraParams) {
  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<X*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int xEws;
  __shared__ int yEws;
  __shared__ int zEws;
  __shared__ char xOrder;
  __shared__ char yOrder;
  __shared__ char zOrder;
  __shared__ sd::LongType len;

  if (threadIdx.x == 0) {
    xEws = shape::elementWiseStride(xShapeInfo);
    yEws = shape::elementWiseStride(yShapeInfo);
    zEws = shape::elementWiseStride(zShapeInfo);
    xOrder = shape::order(xShapeInfo);
    yOrder = shape::order(yShapeInfo);
    zOrder = shape::order(zShapeInfo);
    len = shape::length(xShapeInfo);
  }
  __syncthreads();

  if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == yOrder && xOrder == zOrder) {
    for (sd::LongType i = tid; i < len; i += gridDim.x * blockDim.x) {
      z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], extraParams);
    }
  } else if (vx == vz) {
    for (sd::LongType i = tid; i < len; i += gridDim.x * blockDim.x) {
      auto xOffset = shape::getIndexOffset(i, xShapeInfo);
      auto yOffset = shape::getIndexOffset(i, yShapeInfo);

      z[xOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
    }
  } else {
    for (sd::LongType i = tid; i < len; i += gridDim.x * blockDim.x) {
      auto xOffset = shape::getIndexOffset(i, xShapeInfo);
      auto yOffset = shape::getIndexOffset(i, yShapeInfo);
      auto zOffset = shape::getIndexOffset(i, zShapeInfo);

      z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
    }
  }
}

namespace functions {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
void SD_HOST PairWiseIntTransform<X>::intermediateShaped(dim3& launchDims, cudaStream_t* stream, void const* vx,
                                                         sd::LongType const* xShapeInfo, void const* vy,
                                                         sd::LongType const* yShapeInfo, void* vz,
                                                         sd::LongType const* zShapeInfo, void* vextraParams) {
  pairwiseSimpleShaped<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo,
                                                                                         vz, zShapeInfo, vextraParams);
  sd::DebugHelper::checkErrorCode(stream, "PairWiseIntTransform intermediateShaped(...) failed");

}

////////////////////////////////////////////////////////////////////////////////
template <typename X>
void PairWiseIntTransform<X>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx,
                                                sd::LongType const* xShapeInfo, void const* vy,
                                                sd::LongType const* yShapeInfo, void* vz,
                                                sd::LongType const* zShapeInfo, void* vextraParams) {
  auto xType = sd::DataTypeUtils::fromT<X>();

  DISPATCH_BY_OPNUM_T(intermediateShaped,
                      PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams),
                      PAIRWISE_INT_OPS);
  sd::DebugHelper::checkErrorCode(stream, "PairWiseIntTransform intermediateShaped(...) failed");

}

BUILD_SINGLE_TEMPLATE(template class PairWiseIntTransform, , SD_INTEGER_TYPES);
}  // namespace pairwise_transforms
}  // namespace functions

#endif  // PAIRWISE_INT_CU
