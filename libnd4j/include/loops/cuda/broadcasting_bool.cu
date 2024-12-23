/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/StringUtils.h>
#include <loops/broadcasting_bool.h>
#include <loops/legacy_ops.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

    using namespace simdOps;

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpClass>
static SD_KERNEL void broadcastBoolSimple(
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(
      x,
      xShapeInfo,
      y,
      yShapeInfo,
      z,
      zShapeInfo,
      extraParams,
      dimension,
      dimensionLength,
      tadOnlyShapeInfo,
      tadOffsets,
      tadOnlyShapeInfoZ,
      tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpClass>
static SD_KERNEL void broadcastBoolSimple(
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo,
    void* extraParams) {

  functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(
      x,
      xShapeInfo,
      y,
      yShapeInfo,
      z,
      zShapeInfo,
      extraParams);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpClass>
static SD_KERNEL void broadcastBoolInverseSimple(
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  functions::broadcast::BroadcastBool<X, Z>::template transformInverseCuda<OpClass>(
      x,
      xShapeInfo,
      y,
      yShapeInfo,
      z,
      zShapeInfo,
      extraParams,
      dimension,
      dimensionLength,
      tadOnlyShapeInfo,
      tadOffsets,
      tadOnlyShapeInfoZ,
      tadOffsetsZ);
}

namespace functions {
namespace broadcast {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  broadcastBoolSimple<X, Z, OpClass>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          x,
          xShapeInfo,
          y,
          yShapeInfo,
          z,
          zShapeInfo,
          extraParams,
          dimension,
          dimensionLength,
          tadOnlyShapeInfo,
          tadOffsets,
          tadOnlyShapeInfoZ,
          tadOffsetsZ);

  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo,
    void* extraParams) {

  broadcastBoolSimple<X, Z, OpClass>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          x,
          xShapeInfo,
          y,
          yShapeInfo,
          z,
          zShapeInfo,
          extraParams);

  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  DISPATCH_BY_OPNUM_TT(
      intermediateBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams,
             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
             tadOnlyShapeInfoZ, tadOffsetsZ),
      OPS_A(BROADCAST_BOOL_OPS));

  sd::DebugHelper::checkErrorCode(stream, "execBroadcast(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    const int opNum,
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo,
    void* extraParams) {

  DISPATCH_BY_OPNUM_TT(
      intermediateBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams),
      OPS_A(BROADCAST_BOOL_OPS));

  DEBUG_KERNEL(stream, opNum);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateInverseBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  broadcastBoolInverseSimple<X, Z, OpClass>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          x,
          xShapeInfo,
          y,
          yShapeInfo,
          z,
          zShapeInfo,
          extraParams,
          dimension,
          dimensionLength,
          tadOnlyShapeInfo,
          tadOffsets,
          tadOnlyShapeInfoZ,
          tadOffsetsZ);

  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execInverseBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* x,
    sd::LongType const* xShapeInfo,
    void const* y,
    sd::LongType const* yShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void* extraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  DISPATCH_BY_OPNUM_TT(
      intermediateInverseBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams,
             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
             tadOnlyShapeInfoZ, tadOffsetsZ),
      OPS_A(BROADCAST_BOOL_OPS));

  sd::DebugHelper::checkErrorCode(stream, "execInverseBroadcast(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformInverseCuda(
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ       = tadOffsets;
  }

  auto x          = reinterpret_cast<const X*>(vx);
  auto y          = reinterpret_cast<const X*>(vy);
  auto z          = reinterpret_cast<Z*>(vz);
  auto extraParams= reinterpret_cast<X*>(vextraParams);

  __shared__ sd::LongType tadLength;
  __shared__ int numTads;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType tadRank;
  __shared__ const sd::LongType* tadShapePtr;
  __shared__ const sd::LongType* tadStridePtr;

  __shared__ sd::LongType tadRankZ;
  __shared__ const sd::LongType* tadShapePtrZ;
  __shared__ const sd::LongType* tadStridePtrZ;

  if (threadIdx.x == 0) {
    tadLength = shape::length(tadOnlyShapeInfo);
    numTads   = shape::length(yShapeInfo) / tadLength;

    xRank     = shape::rank(xShapeInfo);
    xShapePtr = shape::shapeOf(xShapeInfo);
    xStridePtr= shape::stride(xShapeInfo);

    tadRank     = shape::rank(tadOnlyShapeInfo);
    tadShapePtr = shape::shapeOf(tadOnlyShapeInfo);
    tadStridePtr= shape::stride(tadOnlyShapeInfo);

    tadRankZ     = shape::rank(tadOnlyShapeInfoZ);
    tadShapePtrZ = shape::shapeOf(tadOnlyShapeInfoZ);
    tadStridePtrZ= shape::stride(tadOnlyShapeInfoZ);
  }
  __syncthreads();

  for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
    auto rZ = z + tadOffsetsZ[r];
    auto rY = y + tadOffsets[r];

    for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
      sd::LongType coordsX[SD_MAX_RANK];
      sd::LongType coordsY[SD_MAX_RANK];
      sd::LongType coordsZ[SD_MAX_RANK];
      sd::LongType xOffset;
      sd::LongType yOffset;
      sd::LongType zOffset;

      // for x
      INDEX2COORDS(i, xRank, xShapePtr, coordsX);
      COORDS2INDEX(xRank, xStridePtr, coordsX, xOffset);

      // for y (tad)
      INDEX2COORDS(i, tadRank, tadShapePtr, coordsY);
      COORDS2INDEX(tadRank, tadStridePtr, coordsY, yOffset);

      // for z (tadZ)
      INDEX2COORDS(i, tadRankZ, tadShapePtrZ, coordsZ);
      COORDS2INDEX(tadRankZ, tadStridePtrZ, coordsZ, zOffset);

      rZ[zOffset] = OpType::op(x[xOffset], rY[yOffset], extraParams);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformCuda(
    void const* vx,
    sd::LongType const* xShapeInfo,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void* vextraParams,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {

  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ       = tadOffsets;
  }

  auto x          = reinterpret_cast<const X*>(vx);
  auto y          = reinterpret_cast<const X*>(vy);
  auto z          = reinterpret_cast<Z*>(vz);
  auto extraParams= reinterpret_cast<X*>(vextraParams);

  __shared__ sd::LongType tadLength;
  __shared__ sd::LongType numTads;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType yRank;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  __shared__ sd::LongType zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    tadLength = shape::length(tadOnlyShapeInfo);
    numTads   = shape::length(xShapeInfo) / tadLength;

    xRank     = shape::rank(tadOnlyShapeInfo);
    xShapePtr = shape::shapeOf(tadOnlyShapeInfo);
    xStridePtr= shape::stride(tadOnlyShapeInfo);

    yRank     = shape::rank(yShapeInfo);
    yShapePtr = shape::shapeOf(yShapeInfo);
    yStridePtr= shape::stride(yShapeInfo);

    zRank     = shape::rank(tadOnlyShapeInfoZ);
    zShapePtr = shape::shapeOf(tadOnlyShapeInfoZ);
    zStridePtr= shape::stride(tadOnlyShapeInfoZ);
  }
  __syncthreads();

  for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    auto rX = x + tadOffsets[r];
    auto rZ = z + tadOffsetsZ[r];

    for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
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

      rZ[zOffset] = OpType::op(rX[xOffset], y[yOffset], extraParams);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformCuda(
    const void* vx,
    const sd::LongType* xShapeInfo,
    const void* vy,
    const sd::LongType* yShapeInfo,
    void* vz,
    const sd::LongType* zShapeInfo,
    void* vextraParams) {

  const auto x = reinterpret_cast<const X*>(vx);
  const auto y = reinterpret_cast<const X*>(vy);
  auto z       = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  __shared__ sd::LongType zLen;
  __shared__ int xRank, yRank, zRank;
  __shared__ bool xzSameOffsets, yzSameOffsets;

  // We'll store stride data for each shape in shared mem
  __shared__ const sd::LongType* xStridePtr;
  __shared__ const sd::LongType* yStridePtr;
  __shared__ const sd::LongType* zStridePtr;

  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* zShapePtr;

  if (threadIdx.x == 0) {
    zLen           = shape::length(zShapeInfo);

    xRank          = shape::rank(xShapeInfo);
    yRank          = shape::rank(yShapeInfo);
    zRank          = shape::rank(zShapeInfo);

    xShapePtr      = shape::shapeOf(xShapeInfo);
    yShapePtr      = shape::shapeOf(yShapeInfo);
    zShapePtr      = shape::shapeOf(zShapeInfo);

    xStridePtr     = shape::stride(xShapeInfo);
    yStridePtr     = shape::stride(yShapeInfo);
    zStridePtr     = shape::stride(zShapeInfo);

    xzSameOffsets  = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    yzSameOffsets  = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);
  }
  __syncthreads();

  const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const auto totalThreads = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    sd::LongType coords[SD_MAX_RANK];
    COORDS2INDEX_1D_TO_ND(i, zRank, zShapePtr, coords);

    sd::LongType zOffset;
    COORDS2INDEX(zRank, zStridePtr, coords, zOffset);

    sd::LongType xOffset;
    if (xzSameOffsets) {
      xOffset = zOffset;
    } else {
      COORDS2INDEX(xRank, xStridePtr, coords, xOffset);
    }

    sd::LongType yOffset;
    if (yzSameOffsets) {
      yOffset = zOffset;
    } else {
      COORDS2INDEX(yRank, yStridePtr, coords, yOffset);
    }

    z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
  }
}

// build the class
BUILD_DOUBLE_TEMPLATE(template class BroadcastBool, , SD_COMMON_TYPES, SD_BOOL_TYPES);

}  // namespace broadcast
}  // namespace functions
