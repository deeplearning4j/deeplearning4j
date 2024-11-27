/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/gammaMathFunc.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void diGammaCuda(const void *vx, const LongType *xShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType len;
  __shared__ bool sameOffset;

  if (threadIdx.x == 0) {
    len = shape::length(xShapeInfo);
    sameOffset = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
  }
  __syncthreads();

  LongType xCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xOffset;
  LongType zOffset;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
    INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, xCoords);
    COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), xCoords, xOffset);
    if (sameOffset) {
      zOffset = xOffset;
    } else {
      INDEX2COORDS(i, shape::rank(zShapeInfo), zShapeInfo, zCoords);
      COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords, zOffset);
    }

    z[zOffset] = diGammaScalar<T>(x[xOffset]);
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void diGammaCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                                const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, void *vz,
                                const LongType *zShapeInfo) {
  diGammaCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(vx, xShapeInfo, vz, zShapeInfo);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "crossCuda failed");
}

///////////////////////////////////////////////////////////////////
void diGamma(LaunchContext *context, NDArray&x, NDArray &z) {
  dim3 digammaDims2 = digammaDims(z.lengthOf());
  NDArray::prepareSpecialUse({&z}, {&x});
  BUILD_SINGLE_SELECTOR(x.dataType(), diGammaCudaLauncher,
                        (digammaDims2.y, digammaDims2.x, digammaDims2.z, context->getCudaStream(), x.specialBuffer(),
                         x.specialShapeInfo(), z.specialBuffer(), z.specialShapeInfo()),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&z}, {&x});
}

BUILD_SINGLE_TEMPLATE(template void diGammaCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                       const cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo, void *vz,
                       const sd::LongType *zShapeInfo),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
