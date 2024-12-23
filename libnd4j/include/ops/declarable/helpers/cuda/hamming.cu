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
// @author raver119@gmail.com
//
#include <ops/declarable/helpers/hamming.h>
#include <ops/declarable/helpers/helpers.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {
template <typename X, typename Z>
static SD_KERNEL void _hammingKernel(const void *vx, const LongType *xShapeInfo, const void *vy,
                                     const LongType *yShapeInfo, void *vz, void *reductionBuffer, LongType length) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);

  __shared__ LongType shared[SD_CUDA_BLOCK_SIZE];
  __shared__ LongType xRank, yRank;
  __shared__ const LongType *xShapePtr, *xStridePtr;
  __shared__ const LongType *yShapePtr, *yStridePtr;

  if (threadIdx.x == 0) {
    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    xShapePtr = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);
    yShapePtr = shape::shapeOf(yShapeInfo);
    yStridePtr = shape::stride(yShapeInfo);
  }
  __syncthreads();

  // Initialize shared memory for intermediate results
  shared[threadIdx.x] = 0;

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (LongType e = tid; e < length; e += blockDim.x * gridDim.x) {
    LongType xCoords[SD_MAX_RANK], yCoords[SD_MAX_RANK];
    LongType xOffset, yOffset;

    // Calculate coordinates and offsets using cached values
    INDEX2COORDS(e, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);
    INDEX2COORDS(e, yRank, yShapePtr, yCoords);
    COORDS2INDEX(yRank, yStridePtr, yCoords, yOffset);

    auto _x = static_cast<unsigned long long>(x[xOffset]);
    auto _y = static_cast<unsigned long long>(y[yOffset]);

    // Save intermediate results into shared memory
    shared[threadIdx.x] += __popcll(_x ^ _y);
  }
  __syncthreads();

  // Reduction within a block
  auto numItems = sd::math::sd_min<LongType>(blockDim.x, length);
  auto floorPow2 = numItems;
  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) floorPow2 &= floorPow2 - 1;

    if (threadIdx.x >= floorPow2)
      shared[threadIdx.x - floorPow2] += shared[threadIdx.x];

    __syncthreads();
  }

  for (LongType activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
    if (threadIdx.x < activeThreads && threadIdx.x + activeThreads < numItems)
      shared[threadIdx.x] += shared[threadIdx.x + activeThreads];

    __syncthreads();
  }

  // Write the final result to global memory
  if (threadIdx.x == 0 && shared[0] > 0) {
    math::atomics::sd_atomicAdd<Z>(&z[0], static_cast<Z>(shared[0]));
  }
}


template <typename X, typename Z>
static void _hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &z) {
  dim3 launchDims = getLaunchDims("hamming");
  _hammingKernel<X, Z><<<launchDims.x,launchDims.y, launchDims.z, *context->getCudaStream()>>>(
      x.specialBuffer(), x.specialShapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.specialBuffer(), nullptr,
      x.lengthOf());
  DebugHelper::checkErrorCode(context->getCudaStream(),"_hammingKernel failed");

}

void hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &output) {
  NDArray::prepareSpecialUse({&output}, {&x, &y});
  BUILD_DOUBLE_SELECTOR(x.dataType(), output.dataType(), _hamming, (context, x, y, output), SD_INTEGER_TYPES,
                        SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&x, &y});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
