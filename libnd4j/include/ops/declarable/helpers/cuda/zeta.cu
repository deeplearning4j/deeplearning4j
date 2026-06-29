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
  __shared__ const LongType *xShape, *qShape, *zShape;
  __shared__ const LongType *xStride, *qStride, *zStride;

  if (threadIdx.x == 0) {
    len = shape::length(xShapeInfo);

    xRank = shape::rank(xShapeInfo);
    qRank = shape::rank(qShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    qShape = shape::shapeOf(qShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    xStride = shape::stride(xShapeInfo);
    qStride = shape::stride(qShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto totalThreads = gridDim.x * blockDim.x;

  // Stack-allocated coordinate arrays per thread (no shared memory needed for coords)
  LongType xCoords[SD_MAX_RANK], qCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];

  for (LongType i = tid; i < len; i += totalThreads) {
    LongType xOffset, qOffset, zOffset;

    INDEX2COORDS(i, xRank, xShape, xCoords);
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);

    INDEX2COORDS(i, qRank, qShape, qCoords);
    COORDS2INDEX(qRank, qStride, qCoords, qOffset);

    INDEX2COORDS(i, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

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
  NDArray::prepareSpecialUse({&z}, {&x, &q});

  dim3 launchDims = zetaDims(x.lengthOf());
  // Pin device shape pointers in named locals before the async kernel launch.
  // specialShapeInfo() returns _shapeInfoD whose backing ConstantShapeBuffer is
  // reference-counted and can be freed if evaluated lazily inside the
  // BUILD_SINGLE_SELECTOR argument list (C++ argument evaluation order is
  // unspecified).  Storing in locals keeps the device pointer valid through
  // the launcher call and the cudaStreamSynchronize inside checkErrorCode.
  const LongType* xShapeInfoD = x.specialShapeInfo();
  const LongType* qShapeInfoD = q.specialShapeInfo();
  const LongType* zShapeInfoD = z.specialShapeInfo();
  BUILD_SINGLE_SELECTOR(
      x.dataType(), zetaCudaLauncher,
      (launchDims.x, launchDims.z, launchDims.y, context->getCudaStream(), x.specialBuffer(), xShapeInfoD,
       q.specialBuffer(), qShapeInfoD, z.specialBuffer(), zShapeInfoD),
      SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({&z}, {&x, &q});
}

BUILD_SINGLE_TEMPLATE( void zetaCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMmemory,
                       const cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo, const void *vq,
                       const sd::LongType *qShapeInfo, void *vz, const sd::LongType *zShapeInfo),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
