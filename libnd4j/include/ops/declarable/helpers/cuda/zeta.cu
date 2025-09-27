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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 26.04.2019
//
#include <ops/declarable/helpers/zeta.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void zetaCuda(const void *vx, const LongType *xShapeInfo, const void *vq, const LongType *qShapeInfo,
                               void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const T *>(vx);
  const auto q = reinterpret_cast<const T *>(vq);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType len;
  __shared__ LongType xRank, qRank, zRank;
  __shared__ LongType *sharedMem;
  __shared__ const LongType *xShape, *qShape, *zShape;
  __shared__ const LongType *xStride, *qStride, *zStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    len = shape::length(xShapeInfo);

    // Cache ranks
    xRank = shape::rank(xShapeInfo);
    qRank = shape::rank(qShapeInfo);
    zRank = shape::rank(zShapeInfo);

    // Cache shape pointers
    xShape = shape::shapeOf(xShapeInfo);
    qShape = shape::shapeOf(qShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    // Cache stride pointers
    xStride = shape::stride(xShapeInfo);
    qStride = shape::stride(qShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto totalThreads = gridDim.x * blockDim.x;

  // Use shared memory for coordinates
  auto coords = sharedMem + threadIdx.x * SD_MAX_RANK;

  for (LongType i = tid; i < len; i += totalThreads) {
    LongType xOffset, qOffset, zOffset;

    INDEX2COORDS(i, xRank, xShape, coords);
    COORDS2INDEX(xRank, xStride, coords, xOffset);

    INDEX2COORDS(i, qRank, qShape, coords);
    COORDS2INDEX(qRank, qStride, coords, qOffset);

    INDEX2COORDS(i, zRank, zShape, coords);
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    z[zOffset] = zetaScalar<T>(x[xOffset], q[qOffset]);
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
static void zetaCudaLauncher(const int blocksPerGrid, const int sharedMemory, const int threadsPerBlock,
                             const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, const void *vq,
                             const LongType *qShapeInfo, void *vz, const LongType *zShapeInfo) {
  zetaCuda<T>
      <<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(vx, xShapeInfo, vq, qShapeInfo, vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "zetaCuda failed");
}

void zeta(LaunchContext *context, NDArray&x, NDArray&q, NDArray &z) {
  if (!x.isActualOnDeviceSide()) x.syncToDevice();
  if (!q.isActualOnDeviceSide()) q.syncToDevice();

  dim3 launchDims = zetaDims(x.lengthOf());
  BUILD_SINGLE_SELECTOR(
      x.dataType(), zetaCudaLauncher,
      (launchDims.x, launchDims.z, launchDims.y, context->getCudaStream(), x.specialBuffer(), x.specialShapeInfo(),
       q.specialBuffer(), q.specialShapeInfo(), z.specialBuffer(), z.specialShapeInfo()),
      SD_FLOAT_TYPES);

  x.tickReadHost();
  q.tickReadHost();
  z.tickWriteDevice();
}

BUILD_SINGLE_TEMPLATE( void zetaCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMmemory,
                       const cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo, const void *vq,
                       const sd::LongType *qShapeInfo, void *vz, const sd::LongType *zShapeInfo),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
