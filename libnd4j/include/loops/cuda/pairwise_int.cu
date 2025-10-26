/********************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
//
#ifndef PAIRWISE_INT_CU
#define PAIRWISE_INT_CU

#include "../pairwise_int.h"

    using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
SD_KERNEL static void pairwiseSimpleShaped(
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams) {

  auto x           = reinterpret_cast<const X*>(vx);
  auto y           = reinterpret_cast<const X*>(vy);
  auto z           = reinterpret_cast<X*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ sd::LongType length;
  __shared__ int xRank;
  __shared__ int yRank;
  __shared__ int zRank;

  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    length      = shape::length(xShapeInfo);

    xRank       = shape::rank(xShapeInfo);
    xShapePtr   = shape::shapeOf(xShapeInfo);
    xStridePtr  = shape::stride(xShapeInfo);

    yRank       = shape::rank(yShapeInfo);
    yShapePtr   = shape::shapeOf(yShapeInfo);
    yStridePtr  = shape::stride(yShapeInfo);

    zRank       = shape::rank(zShapeInfo);
    zShapePtr   = shape::shapeOf(zShapeInfo);
    zStridePtr  = shape::stride(zShapeInfo);
  }
  __syncthreads();

  const auto totalThreads = gridDim.x * blockDim.x;
  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType coordsX[SD_MAX_RANK];
    sd::LongType coordsY[SD_MAX_RANK];
    sd::LongType coordsZ[SD_MAX_RANK];

    sd::LongType xOffset;
    sd::LongType yOffset;
    sd::LongType zOffset;

    INDEX2COORDS(i, xRank, xShapePtr, coordsX);
    COORDS2INDEX(xRank, xStridePtr, coordsX, xOffset);

    INDEX2COORDS(i, yRank, yShapePtr, coordsY);
    COORDS2INDEX(yRank, yStridePtr, coordsY, yOffset);

    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, zOffset);

    z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
  }
}

namespace functions {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
void SD_HOST PairWiseIntTransform<X>::intermediateShaped(
    dim3& launchDims,
    cudaStream_t* stream,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams) {

  pairwiseSimpleShaped<X, OpType>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          vz,
          zShapeInfo,
          vextraParams);

  sd::DebugHelper::checkErrorCode(
      stream,
      "PairWiseIntTransform intermediateShaped(...) failed");
}

////////////////////////////////////////////////////////////////////////////////
template <typename X>
void PairWiseIntTransform<X>::executeCudaShaped(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams) {

  DISPATCH_BY_OPNUM_T(
      intermediateShaped,
      PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams),
      PAIRWISE_INT_OPS);

  sd::DebugHelper::checkErrorCode(
      stream,
      "PairWiseIntTransform executeCudaShaped(...) failed");
}

BUILD_SINGLE_TEMPLATE( class PairWiseIntTransform, , SD_INTEGER_TYPES);

}  // namespace pairwise_transforms
}  // namespace functions

#endif  // PAIRWISE_INT_CU
