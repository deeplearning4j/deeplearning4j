/********************************************************************************
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
 ********************************************************************************/

//
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
//
#ifndef PAIRWISE_BOOL_CU
#define PAIRWISE_BOOL_CU

#include "../pairwise_bool.h"

using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
SD_KERNEL static void pairwiseSimpleShaped(
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams) {

  auto x            = reinterpret_cast<const X*>(vx);
  auto y            = reinterpret_cast<const X*>(vy);
  auto z            = reinterpret_cast<Z*>(vz);
  auto extraParams  = reinterpret_cast<X*>(vextraParams);

  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ sd::LongType length;

  // Cache shape info
  __shared__ int xRank;
  __shared__  sd::LongType* xShapePtr;
  __shared__  sd::LongType* xStridePtr;

  __shared__ int yRank;
  __shared__  sd::LongType* yShapePtr;
  __shared__  sd::LongType* yStridePtr;

  __shared__ int zRank;
  __shared__  sd::LongType* zShapePtr;
  __shared__  sd::LongType* zStridePtr;

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

    sd::LongType offsetX;
    sd::LongType offsetY;
    sd::LongType offsetZ;

    INDEX2COORDS(i, xRank, xShapePtr, coordsX);
    COORDS2INDEX(xRank, xStridePtr, coordsX, offsetX);

    INDEX2COORDS(i, yRank, yShapePtr, coordsY);
    COORDS2INDEX(yRank, yStridePtr, coordsY, offsetY);

    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, offsetZ);

    z[offsetZ] = OpType::op(x[offsetX], y[offsetY], extraParams);
  }
}

namespace functions {
namespace pairwise_transforms {

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void SD_HOST PairWiseBoolTransform<X,Z>::intermediateShaped(
    dim3& launchDims,
    cudaStream_t* stream,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams) {

  pairwiseSimpleShaped<X, Z, OpType>
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
      "PairWiseBoolTransform intermediateShaped(...) failed");
}

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void PairWiseBoolTransform<X,Y>::executeCudaShaped(
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

  DISPATCH_BY_OPNUM_TT(
      intermediateShaped,
      PARAMS(launchDims, stream, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vextraParams),
      PAIRWISE_BOOL_OPS);
}

BUILD_DOUBLE_TEMPLATE( class PairWiseBoolTransform, , SD_COMMON_TYPES, SD_BOOL_TYPES);

}  // namespace pairwise_transforms
}  // namespace functions

#endif  // PAIRWISE_BOOL_CU
