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
#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/gammaMathFunc.h>
#include <ops/declarable/helpers/zeta.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void polyGammaCuda(const void *vn, const LongType *nShapeInfo, const void *vx,
                                    const LongType *xShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto n = reinterpret_cast<const T *>(vn);
  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType len;
  __shared__ bool sameOffsetNX, sameOffsetNZ;

  if (threadIdx.x == 0) {
    len = shape::length(nShapeInfo);
    sameOffsetNX = shape::haveSameShapeAndStrides(xShapeInfo, nShapeInfo);
    sameOffsetNZ = shape::haveSameShapeAndStrides(zShapeInfo, nShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto totalThreads = gridDim.x * blockDim.x;

  for (int i = tid; i < len; i += totalThreads) {
    const auto nOffset = shape::getIndexOffset(i, nShapeInfo);
    const auto xOffset = sameOffsetNX ? nOffset : shape::getIndexOffset(i, xShapeInfo);
    const auto zOffset = sameOffsetNZ ? nOffset : shape::getIndexOffset(i, zShapeInfo);

    const T order = n[nOffset];

    int sign = (static_cast<int>(order) + 1) % 2 ? -1 : 1;

    if (order != static_cast<int>(order)) {
      z[zOffset] = DataTypeUtils::nanOrZero<T>();
    } else if (order == 0) {
      z[zOffset] = diGammaScalar<T>(x[xOffset]);
    } else {
      T factorial = 1;
      for (int i = 2; i <= order; ++i) factorial *= i;

      z[zOffset] = sign * factorial * zetaScalar<T>(order + 1, x[xOffset]);
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void polyGammaCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                                  const cudaStream_t *stream, const void *vn, const LongType *nShapeInfo,
                                  const void *vx, const LongType *xShapeInfo, void *vz,
                                  const LongType *zShapeInfo) {
  polyGammaCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(vn, nShapeInfo, vx, xShapeInfo, vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "print_device failed");

}

///////////////////////////////////////////////////////////////////
void polyGamma(LaunchContext *context, const NDArray &n, const NDArray &x, NDArray &z) {
  NDArray::prepareSpecialUse({&z}, {&n, &x});

  dim3 launchDims = polygammaDims(z.lengthOf());
  BUILD_SINGLE_SELECTOR(
      n.dataType(), polyGammaCudaLauncher,
      (launchDims.y,launchDims.x,launchDims.z, context->getCudaStream(), n.specialBuffer(), n.specialShapeInfo(),
       x.specialBuffer(), x.specialShapeInfo(), z.specialBuffer(), z.specialShapeInfo()),
      SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({&z}, {&n, &x});
}

BUILD_SINGLE_TEMPLATE(template void polyGammaCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,const cudaStream_t *stream, const void *vn,
                       const sd::LongType *nShapeInfo, const void *vx, const sd::LongType *xShapeInfo, void *vz,
                       const sd::LongType *zShapeInfo),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
