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
static SD_KERNEL void broadcastBoolSimple(void const* x, sd::LongType const* xShapeInfo, void const* y,
                                          sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo,
                                          void* extraParams, sd::LongType* dimension, sd::LongType dimensionLength,
                                          sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
                                          sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength, tadOnlyShapeInfo,
      tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpClass>
static SD_KERNEL void broadcastBoolSimple(const void const* x, const sd::LongType const* xShapeInfo,
                                          const void const* y, const sd::LongType const* yShapeInfo, void* z,
                                          const sd::LongType const* zShapeInfo, void* extraParams) {
  functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(x, xShapeInfo, y, yShapeInfo, z,
                                                                             zShapeInfo, extraParams);
}
//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpClass>
static SD_KERNEL void broadcastBoolInverseSimple(void const* x, sd::LongType const* xShapeInfo, void const* y,
                                                 sd::LongType const* yShapeInfo, void* z,
                                                 sd::LongType const* zShapeInfo, void* extraParams,
                                                 sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                                 sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                                 sd::LongType const* tadOffsetsZ) {
  functions::broadcast::BroadcastBool<X, Z>::template transformInverseCuda<OpClass>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength, tadOnlyShapeInfo,
      tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
}

namespace functions {
namespace broadcast {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateBroadcast(
    dim3 launchDims, cudaStream_t* stream, void const* x, sd::LongType const* xShapeInfo, void const* y,
    sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, void* extraParams,
    sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  broadcastBoolSimple<X, Z, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength, tadOnlyShapeInfo,
      tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateBroadcast(dim3 launchDims, cudaStream_t* stream, const void* x,
                                                        const sd::LongType* xShapeInfo, const void* y,
                                                        const sd::LongType* yShapeInfo, void* z,
                                                        const sd::LongType* zShapeInfo, void* extraParams) {
  broadcastBoolSimple<X, Z, OpClass>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x,
                                                sd::LongType const* xShapeInfo, void const* y,
                                                sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo,
                                                void* extraParams, sd::LongType* dimension,
                                                sd::LongType dimensionLength,
                                                sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
                                                sd::LongType const* tadOnlyShapeInfoZ,
                                                sd::LongType const* tadOffsetsZ) {
  DISPATCH_BY_OPNUM_TT(intermediateBroadcast,
                       PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension,
                              dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                       OPS_A(BROADCAST_BOOL_OPS))
  DEBUG_KERNEL(stream, opNum);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execBroadcast(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x,
                                                const sd::LongType* xShapeInfo, const void* y,
                                                const sd::LongType* yShapeInfo, void* z, const sd::LongType* zShapeInfo,
                                                void* extraParams) {
  DISPATCH_BY_OPNUM_TT(intermediateBroadcast,
                       PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams),
                       OPS_A(BROADCAST_BOOL_OPS))
  DEBUG_KERNEL(stream, opNum);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpClass>
SD_HOST void BroadcastBool<X, Z>::intermediateInverseBroadcast(
    dim3 launchDims, cudaStream_t* stream, void const* x, sd::LongType const* xShapeInfo, void const* y,
    sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, void* extraParams,
    sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  broadcastBoolInverseSimple<X, Z, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength, tadOnlyShapeInfo,
      tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
  sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void BroadcastBool<X, Y>::execInverseBroadcast(
    dim3 launchDims, cudaStream_t* stream, int opNum, void const* x, sd::LongType const* xShapeInfo, void const* y,
    sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, void* extraParams,
    sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  DISPATCH_BY_OPNUM_TT(intermediateInverseBroadcast,
                       PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension,
                              dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                       OPS_A(BROADCAST_BOOL_OPS))

  DEBUG_KERNEL(stream, opNum);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformInverseCuda(
    void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* vz,
    sd::LongType const* zShapeInfo, void* vextraParams, sd::LongType* dimension, sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {
  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ = tadOffsets;
  }

  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  // decompose in to several sub tads after
  // moving all dimensions (in sorted order)
  // to the back.
  // permuted version of the x shape info for setting up the tad problem
  __shared__ sd::LongType tadLength;
  __shared__ sd::LongType tadEWS;
  __shared__ int numTads;
  __shared__ sd::LongType xEWS;
  __shared__ sd::LongType zEWS;

  if (threadIdx.x == 0) {
    tadLength = shape::length(tadOnlyShapeInfo);
    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
    numTads = shape::length(yShapeInfo) / tadLength;
    xEWS = shape::elementWiseStride(xShapeInfo);
    zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);
  }
  __syncthreads();

  for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
    auto rZ = z + tadOffsetsZ[r];
    auto rY = y + tadOffsets[r];

    if (tadEWS > 0 && zEWS > 0 && xEWS > 0 && dimensionLength == 1) {
      for (int i = threadIdx.x; i < tadLength; i += blockDim.x)
        rZ[i * zEWS] = OpType::op(x[i * xEWS], rY[i * tadEWS], extraParams);
    } else {
      // it is expected that x and z tads and y array all have the same length
      for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
        auto xOffset = shape::getIndexOffset(i, xShapeInfo);
        auto yOffset = shape::getIndexOffset(i, tadOnlyShapeInfo);
        auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ);

        rZ[zOffset] = OpType::op(x[xOffset], rY[yOffset], extraParams);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformCuda(void const* vx, sd::LongType const* xShapeInfo, void const* vy,
                                                  sd::LongType const* yShapeInfo, void* vz,
                                                  sd::LongType const* zShapeInfo, void* vextraParams,
                                                  sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                                  sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                                  sd::LongType const* tadOffsetsZ) {
  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ = tadOffsets;
  }

  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  // decompose in to several sub tads after
  // moving all dimensions (in sorted order)
  // to the back.
  // permuted version of the x shape info for setting up the tad problem
  __shared__ sd::LongType tadLength;
  __shared__ sd::LongType tadEWS;
  __shared__ int numTads;
  __shared__ sd::LongType yEWS;
  __shared__ sd::LongType zEWS;

  if (threadIdx.x == 0) {
    tadLength = shape::length(tadOnlyShapeInfo);
    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
    numTads = shape::length(xShapeInfo) / tadLength;
    yEWS = shape::elementWiseStride(yShapeInfo);
    zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);
  }
  __syncthreads();

  __shared__ Z* rZ;
  __shared__ X const* rX;

  for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
    if (threadIdx.x == 0) {
      rZ = z + tadOffsetsZ[r];
      rX = x + tadOffsets[r];
    }
    __syncthreads();

    if (tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {
      for (int i = threadIdx.x; i < tadLength; i += blockDim.x)
        rZ[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS], extraParams);
    } else {
      // it is expected that x and z tads and y array all have the same length
      for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
        auto xOffset = shape::getIndexOffset(i, tadOnlyShapeInfo);
        auto yOffset = shape::getIndexOffset(i, yShapeInfo);
        auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ);

        rZ[zOffset] = OpType::op(rX[xOffset], y[yOffset], extraParams);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void BroadcastBool<X, Z>::transformCuda(const void* vx, const sd::LongType* xShapeInfo, const void* vy,
                                                  const sd::LongType* yShapeInfo, void* vz,
                                                  const sd::LongType* zShapeInfo, void* vextraParams) {
  const X* x = reinterpret_cast<const X*>(vx);
  const X* y = reinterpret_cast<const X*>(vy);
  Z* z = reinterpret_cast<Z*>(vz);

  auto extraParams = reinterpret_cast<X*>(vextraParams);

  __shared__ sd::LongType zLen;
  __shared__ int rank;
  __shared__ bool xzSameOffsets, yzSameOffsets;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    rank = shape::rank(zShapeInfo);

    xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    yzSameOffsets = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  sd::LongType coords[SD_MAX_RANK];

  for (sd::LongType i = tid; i < zLen; i += blockDim.x * gridDim.x) {
    shape::index2coords(i, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);
    const auto xOffset = xzSameOffsets ? zOffset : shape::getOffset(xShapeInfo, coords);
    const auto yOffset = yzSameOffsets ? zOffset : shape::getOffset(yShapeInfo, coords);

    z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
  }
}

BUILD_DOUBLE_TEMPLATE(template class BroadcastBool, , SD_COMMON_TYPES, SD_BOOL_TYPES);
}  // namespace broadcast
}  // namespace functions
