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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {
///////////////////////////////////////////////////////////////////
// x - input, y - indices, z - output
template <typename X, typename Y>
SD_KERNEL static void gatherNDCuda(const void *vx, const LongType *xShapeInfo, const void *vy,
                                   const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<X *>(vz);

  __shared__ int xRank, yRank, zRank, maxRank, yLastDim;
  __shared__ LongType zLen, totalThreads;
  __shared__ const LongType *xShapePtr, *xStridePtr;
  __shared__ const LongType *yShapePtr, *yStridePtr;
  __shared__ const LongType *zShapePtr, *zStridePtr;

  if (threadIdx.x == 0) {
    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);
    maxRank = sd::math::sd_max<int>(yRank, sd::math::sd_max<int>(xRank, zRank));

    zLen = shape::length(zShapeInfo);
    yLastDim = shape::shapeOf(yShapeInfo)[yRank - 1];

    totalThreads = gridDim.x * blockDim.x;

    xShapePtr = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);
    yShapePtr = shape::shapeOf(yShapeInfo);
    yStridePtr = shape::stride(yShapeInfo);
    zShapePtr = shape::shapeOf(zShapeInfo);
    zStridePtr = shape::stride(zShapeInfo);
  }
  __syncthreads();

  extern __shared__ unsigned char shmem[];
  auto coord = reinterpret_cast<LongType *>(shmem) + threadIdx.x * maxRank;

  LongType *zCoordStart, *xCoordStart;

  if (yLastDim == xRank) {
    zCoordStart = coord;
    xCoordStart = coord;
  } else if (zRank >= xRank) {
    zCoordStart = coord;
    xCoordStart = coord + zRank - xRank;
  } else {
    zCoordStart = coord + xRank - zRank;
    xCoordStart = coord;
  }

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < zLen; i += totalThreads) {
    // Compute z coordinates and offset
    INDEX2COORDS(i, zRank, zShapePtr, zCoordStart);
    LongType zOffset;
    COORDS2INDEX(zRank, zStridePtr, zCoordStart, zOffset);

    // Save and modify last y coordinate
    int coordToRestore = (yLastDim != xRank) ? static_cast<int>(zCoordStart[yRank - 1]) : 0;
    zCoordStart[yRank - 1] = 0;

    // Compute y offset
    LongType yOffset;
    COORDS2INDEX(yRank, yStridePtr, zCoordStart, yOffset);

    // Restore z coordinate
    if (yLastDim != xRank) zCoordStart[yRank - 1] = coordToRestore;

    // Compute x coordinates
    for (LongType j = 0; j < yLastDim; ++j) {
      xCoordStart[j] = y[yOffset + j * yStridePtr[yRank - 1]];
    }

    // Compute x offset
    LongType xOffset;
    COORDS2INDEX(xRank, xStridePtr, xCoordStart, xOffset);

    // Assign value to z
    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void gatherNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                 const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo,
                                 const void *vy, const LongType *yShapeInfo, void *vz,
                                 const LongType *zShapeInfo) {
  gatherNDCuda<X, Y>
      <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream),"gatherNDCuda failed");

}

///////////////////////////////////////////////////////////////////
void gatherND(LaunchContext *context, NDArray &input, NDArray &indices, NDArray &output) {
  const int maxRank = sd::math::sd_max<int>(indices.rankOf(), sd::math::sd_max<int>(input.rankOf(), output.rankOf()));


  dim3 gatherNdDims = getGatherNd(output.lengthOf(),maxRank);
  const auto xType = input.dataType();
  const auto yType = indices.dataType();

  PointersManager manager(context, "gatherND");

  NDArray::prepareSpecialUse({&output}, {&input, &indices});
  BUILD_DOUBLE_SELECTOR(xType, yType, gatherNDCudaLauncher,
                        (gatherNdDims.y, gatherNdDims.x, gatherNdDims.z, context->getCudaStream(), input.specialBuffer(),
                         input.specialShapeInfo(), indices.specialBuffer(), indices.specialShapeInfo(),
                         output.specialBuffer(), output.specialShapeInfo()),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&input, &indices});

  manager.synchronize();
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
