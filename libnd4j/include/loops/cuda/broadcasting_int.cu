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
#include <loops/broadcasting_int.h>
#include <loops/legacy_ops.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include <stdexcept>
#include <string>

using namespace simdOps;

//////////////////////////////////////////////////////////////////////////
template <typename X, typename OpClass>
static SD_KERNEL void broadcastIntSimple(void const* x, sd::LongType const* xShapeInfo, void const* y,
                                         sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo,
                                         sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                         sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                         sd::LongType const* tadOffsetsZ) {
  functions::broadcast::BroadcastInt<X>::template transformCuda<OpClass>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,
                                                                         dimension, dimensionLength, tadOnlyShapeInfo,
                                                                         tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename OpClass>
static SD_KERNEL void broadcastIntSimple(const void* x, const sd::LongType const* xShapeInfo, const void* y,
                                         const sd::LongType const* yShapeInfo, void* z,
                                         const sd::LongType const* zShapeInfo) {
  functions::broadcast::BroadcastInt<X>::template transformCuda<OpClass>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename OpClass>
static SD_KERNEL void broadcastBoolInverseSimple(void const* x, sd::LongType const* xShapeInfo, void const* y,
                                                 sd::LongType const* yShapeInfo, void* z,
                                                 sd::LongType const* zShapeInfo, sd::LongType* dimension,
                                                 sd::LongType dimensionLength,
                                                 sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
                                                 sd::LongType const* tadOnlyShapeInfoZ,
                                                 sd::LongType const* tadOffsetsZ) {
  functions::broadcast::BroadcastInt<X>::template transformInverseCuda<OpClass>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
      tadOnlyShapeInfoZ, tadOffsetsZ);
}

namespace functions {
namespace broadcast {
//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateBroadcast(
    dim3 launchDims, cudaStream_t* stream, void const* x, sd::LongType const* xShapeInfo, void const* y,
    sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {
  broadcastIntSimple<X, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
      tadOnlyShapeInfoZ, tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateBroadcast(dim3 launchDims, cudaStream_t* stream, const void* x,
                                                    const sd::LongType* xShapeInfo, const void* y,
                                                    const sd::LongType* yShapeInfo, void* z,
                                                    const sd::LongType* zShapeInfo) {
  broadcastIntSimple<X, OpClass>
  <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
SD_HOST void BroadcastInt<X>::execBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x,
                                            sd::LongType const* xShapeInfo, void const* y,
                                            sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo,
                                            sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                            sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                            sd::LongType const* tadOffsetsZ) {
  DISPATCH_BY_OPNUM_T(intermediateBroadcast,
                      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                      OPS_A(BROADCAST_INT_OPS))
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
SD_HOST void BroadcastInt<X>::execBroadcast(dim3 launchDims, cudaStream_t* stream, const int opNum, const void* x,
                                            const sd::LongType const* xShapeInfo, const void* y,
                                            const sd::LongType const* yShapeInfo, void* z,
                                            const sd::LongType const* zShapeInfo) {
  DISPATCH_BY_OPNUM_T(intermediateBroadcast, PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo),
                      OPS_A(BROADCAST_INT_OPS))
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateInverseBroadcast(
    dim3 launchDims, cudaStream_t* stream, void const* x, sd::LongType const* xShapeInfo, void const* y,
    sd::LongType const* yShapeInfo, void* z, sd::LongType const* zShapeInfo, sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {
  broadcastBoolInverseSimple<X, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
      tadOnlyShapeInfoZ, tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
SD_HOST void BroadcastInt<X>::execInverseBroadcast(dim3 launchDims, cudaStream_t* stream, int opNum, void const* x,
                                                   sd::LongType const* xShapeInfo, void const* y,
                                                   sd::LongType const* yShapeInfo, void* z,
                                                   sd::LongType const* zShapeInfo, sd::LongType* dimension,
                                                   sd::LongType dimensionLength,
                                                   sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
                                                   sd::LongType const* tadOnlyShapeInfoZ,
                                                   sd::LongType const* tadOffsetsZ) {
  DISPATCH_BY_OPNUM_T(intermediateInverseBroadcast,
                      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                      OPS_A(BROADCAST_INT_OPS))
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void BroadcastInt<X>::transformInverseCuda(
    void const* vx, sd::LongType const* xShapeInfo, void const* vy, sd::LongType const* yShapeInfo, void* vz,
    sd::LongType const* zShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ = tadOffsets;
  }

  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<X*>(vz);

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
    tadLength = shape::length(tadOnlyShapeInfo);  // shape::tadLength(xShapeInfo, dimension, dimensionLength);
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
      for (int i = threadIdx.x; i < tadLength; i += blockDim.x) rZ[i * zEWS] = OpType::op(x[i * xEWS], rY[i * tadEWS]);
    } else {
      // it is expected that x and z tads and y array all have the same length
      for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
        auto xOffset = shape::getIndexOffset(i, xShapeInfo);
        auto yOffset = shape::getIndexOffset(i, tadOnlyShapeInfo);
        auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ);

        rZ[zOffset] = OpType::op(x[xOffset], rY[yOffset]);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void BroadcastInt<X>::transformCuda(void const* vx, sd::LongType const* xShapeInfo, void const* vy,
                                              sd::LongType const* yShapeInfo, void* vz, sd::LongType const* zShapeInfo,
                                              sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                              sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                              sd::LongType const* tadOffsetsZ) {
  if (tadOnlyShapeInfoZ == nullptr) {
    tadOnlyShapeInfoZ = tadOnlyShapeInfo;
    tadOffsetsZ = tadOffsets;
  }

  auto x = reinterpret_cast<X const*>(vx);
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<X*>(vz);

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
    tadLength = shape::length(tadOnlyShapeInfo);  // shape::tadLength(xShapeInfo, dimension, dimensionLength);
    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
    numTads = shape::length(xShapeInfo) / tadLength;
    yEWS = shape::elementWiseStride(yShapeInfo);
    zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);
  }
  __syncthreads();

  __shared__ X* rZ;
  __shared__ X const* rX;

  for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
    if (threadIdx.x == 0) {
      rZ = z + tadOffsetsZ[r];
      rX = x + tadOffsets[r];
    }
    __syncthreads();

    if (tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {
      for (int i = threadIdx.x; i < tadLength; i += blockDim.x) rZ[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS]);
    } else {
      // it is expected that x and z tads and y array all have the same length
      for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
        auto xOffset = shape::getIndexOffset(i, tadOnlyShapeInfo);
        auto yOffset = shape::getIndexOffset(i, yShapeInfo);
        auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ);

        rZ[zOffset] = OpType::op(rX[xOffset], y[yOffset]);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void BroadcastInt<X>::transformCuda(const void* vx, const sd::LongType const* xShapeInfo, const void* vy,
                                              const sd::LongType const* yShapeInfo, void* vz,
                                              const sd::LongType const* zShapeInfo) {
  const X* x = reinterpret_cast<const X*>(vx);
  const X* y = reinterpret_cast<const X*>(vy);
  X* z = reinterpret_cast<X*>(vz);

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

    z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
  }
}

BUILD_SINGLE_TEMPLATE(template class BroadcastInt, , SD_INTEGER_TYPES);
}  // namespace broadcast
}  // namespace functions
