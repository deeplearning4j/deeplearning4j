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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/addBias.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL static void addBiasCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                  const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                  const bool isNCHW) {
  // bias [oC]

  // if(input_rank == 4)
  // input and output have same shapes: [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
  // if(input_rank == 5)
  // input and output have same shapes: [bS, oD, oH, oW, oC] (NHWC) or [bS, oD, oC, oH, oW] (NCHW)

  const X* x = reinterpret_cast<const X*>(vx);
  const Y* y = reinterpret_cast<const Y*>(vy);
  X* z = reinterpret_cast<X*>(vz);

  __shared__ LongType rank, channelPosition, posOfNonUnityDim;
  __shared__ LongType len, *sharedMem;
  __shared__ bool xzSameOffsets, xzAreSame;
  __shared__ const LongType *xShape;
  __shared__ const LongType *xStride;
  __shared__ const LongType *zStride;
  __shared__ const LongType *yStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(xShapeInfo);  // xRank == zRank
    xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    len = shape::length(xShapeInfo);
    channelPosition = isNCHW ? 1 : rank - 1;  // second or last
    xzAreSame = x == z;

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
    yStride = shape::stride(yShapeInfo);

    shape::isCommonVector(yShapeInfo, posOfNonUnityDim);
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    INDEX2COORDS(i, rank, xShape, coords);

    LongType xOffsets;
    COORDS2INDEX(rank, xStride, coords, xOffsets);
    LongType zOffsets;
    COORDS2INDEX(rank, zStride, coords, zOffsets);
    LongType yOffsets = coords[channelPosition] * yStride[posOfNonUnityDim];

    if (xzAreSame)
      z[zOffsets] += static_cast<X>(y[yOffsets]);
    else
      z[zOffsets] = static_cast<X>(x[xOffsets]) + static_cast<X>(y[yOffsets]);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void addBiasCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                const void* vy, const LongType* yShapeInfo, void* vz,
                                const LongType* zShapeInfo, const bool isNCHW) {
  addBiasCuda<X, Y>
  <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, isNCHW);
  sd::DebugHelper::checkGlobalErrorCode("addbias  failed");

}

template <typename X, typename Y>
SD_KERNEL static void addBias2DCuda(const void* vx, const void* vy, void* vz, uint32_t blocks, uint32_t length) {
  auto y = reinterpret_cast<const Y*>(vy);

  for (uint32_t b = blockIdx.x; b < blocks; b += gridDim.x) {
    auto x = reinterpret_cast<const X*>(vx) + length * b;
    auto z = reinterpret_cast<X*>(vz) + length * b;

    for (uint32_t e = threadIdx.x; e < length; e += blockDim.x) {
      z[e] = x[e] + y[e];
    }
  }
}

template <typename X, typename Y>
static void addBias2DCudaLauncher(const cudaStream_t* stream, const void* vx, const void* vy, void* vz, uint32_t blocks,
                                  uint32_t length) {
  dim3 dims = getAddBiasDims(2, 2);

  addBias2DCuda<X, Y><<<dims.x, dims.y, dims.z, *stream>>>(vx, vy, vz, blocks, length);
  sd::DebugHelper::checkGlobalErrorCode("addbias 2d  failed");

}

//////////////////////////////////////////////////////////////////////////
void addBias(graph::Context& block, NDArray& input, NDArray& bias, NDArray& output, const bool isNCHW) {
  PointersManager manager(block.launchContext(), "addBias");
  NDArray::prepareSpecialUse({&output}, {&input, &bias});

  if (input.rankOf() == 2 && bias.rankOf() == 1 && input.ordering() == 'c' && output.ordering() == 'c' &&
  input.sizeAt(1) == bias.sizeAt(0)) {
    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBias2DCudaLauncher,
                          (block.launchContext()->getCudaStream(), input.specialBuffer(), bias.specialBuffer(),
                              output.specialBuffer(), input.sizeAt(0), bias.sizeAt(0)),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  } else {
    // default case
    dim3 dims = getAddBiasDims(input.rankOf(), input.rankOf());
    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBiasCudaLauncher,
                          (dims.x, dims.y, dims.z, block.launchContext()->getCudaStream(),
                              input.specialBuffer(), input.specialShapeInfo(), bias.specialBuffer(),
                              bias.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), isNCHW),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  }
  NDArray::registerSpecialUse({&output}, {&input, &bias});
  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
