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
// @author Yurii Shyrma, created on 10.06.2019
//

#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/cross.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void crossCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo) {
  __shared__ const T* x;
  __shared__ const T* y;
  __shared__ T* z;
  __shared__ LongType rank;
  __shared__ LongType lenWithoutLastDim, totalThreads;
  __shared__ const LongType *xShape, *xStride, *yShape, *yStride, *zStride;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<const T*>(vx);
    y = reinterpret_cast<const T*>(vy);
    z = reinterpret_cast<T*>(vz);

    rank = shape::rank(xShapeInfo);
    lenWithoutLastDim = shape::length(xShapeInfo) / xShapeInfo[rank];
    totalThreads = gridDim.x * blockDim.x;

    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  extern __shared__ LongType sharedMem[];
  auto coords = sharedMem + threadIdx.x * rank;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < lenWithoutLastDim; i += totalThreads) {
    // Compute coordinates without the last dimension
    INDEX2COORDS(i, rank - 1, xShape, coords);
    coords[rank - 1] = 0;

    LongType xOffset, yOffset, zOffset;
    COORDS2INDEX(rank, xStride, coords, xOffset);
    COORDS2INDEX(rank, yStride, coords, yOffset);

    // Fetch elements for cross product
    const auto x0 = x[xOffset];
    const auto y0 = y[yOffset];

    xOffset += xStride[rank - 1];
    yOffset += yStride[rank - 1];

    const auto x1 = x[xOffset];
    const auto y1 = y[yOffset];

    xOffset += xStride[rank - 1];
    yOffset += yStride[rank - 1];

    const auto x2 = x[xOffset];
    const auto y2 = y[yOffset];

    // Compute offsets for output
    COORDS2INDEX(rank, zStride, coords, zOffset);
    z[zOffset] = x1 * y2 - x2 * y1;

    zOffset += zStride[rank - 1];
    z[zOffset] = x2 * y0 - x0 * y2;

    zOffset += zStride[rank - 1];
    z[zOffset] = x0 * y1 - x1 * y0;
  }
}


template <typename T>
SD_HOST static void crossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                      const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                      const void* vy, const LongType* yShapeInfo, void* vz,
                                      const LongType* zShapeInfo) {
  crossCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream),"crossCuda failed");

}
BUILD_SINGLE_TEMPLATE(template void crossCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, const void* vy,
                       const sd::LongType* yShapeInfo, void* vz, const sd::LongType* zShapeInfo),
                      SD_NUMERIC_TYPES);

void crossBatched(LaunchContext* context, NDArray* x, NDArray* y, NDArray* z) {
  dim3 launchDims = getCross(x->lengthOf(),x->rankOf(),x->sizeAt(-1));
  PointersManager manager(context, "cross");

  NDArray::prepareSpecialUse({z}, {x, y});
  BUILD_SINGLE_SELECTOR(
      x->dataType(), crossCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), x->specialBuffer(), x->specialShapeInfo(),
       y->specialBuffer(), y->specialShapeInfo(), z->specialBuffer(), z->specialShapeInfo()),
      SD_NUMERIC_TYPES);
  NDArray::registerSpecialUse({z}, {x, y});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
