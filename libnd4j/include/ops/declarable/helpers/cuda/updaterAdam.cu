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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <helpers/PointersManager.h>
#include <math/platformmath.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/updatersHelpers.h>
#include <system/op_boilerplate.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void adamUpdaterCuda(const void* vx, const LongType* xShapeInfo, const void* vinv,
                               const LongType* invShapeInfo, const void* vinm, const LongType* inmShapeInfo,
                               void* vz, const LongType* zShapeInfo, void* vstV, const LongType* stvShapeInfo,
                               void* vstM, const LongType* stmShapeInfo, const T lr, const T beta1, const T beta2,
                               const T epsilon, const T iteration) {
  const auto grad = reinterpret_cast<const T*>(vx);
  const auto initU = reinterpret_cast<const T*>(vinv);
  const auto initM = reinterpret_cast<const T*>(vinm);
  auto up = reinterpret_cast<T*>(vz);
  auto stU = reinterpret_cast<T*>(vstV);
  auto stM = reinterpret_cast<T*>(vstM);

  __shared__ LongType xLen, xRank, zRank, invRank, inmRank, stvRank, stmRank;
  __shared__ T epsilonT;
  __shared__ bool bOrdering, bXZsame, bXInUSame, bXStUSame, bXInMSame, bXStMSame;
  __shared__ LongType *sharedMem;
  __shared__ const LongType *xShape, *zShape, *invShape, *inmShape, *stvShape, *stmShape;
  __shared__ const LongType *xStride, *zStride, *invStride, *inmStride, *stvStride, *stmStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    xLen = shape::length(xShapeInfo);

    T beta1T = math::sd_pow<T, T, T>(beta1, (iteration + 1));
    T beta2T = math::sd_pow<T, T, T>(beta2, (iteration + 1));
    epsilonT = lr * math::sd_sqrt<T, T>(1. - beta2T) / (1.0 - beta1T);
    if (math::sd_isnan(epsilonT) || 0 == epsilonT || math::sd_isinf(epsilonT)) epsilonT = epsilon;

    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);
    invRank = shape::rank(invShapeInfo);
    inmRank = shape::rank(inmShapeInfo);
    stvRank = shape::rank(stvShapeInfo);
    stmRank = shape::rank(stmShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
    invShape = shape::shapeOf(invShapeInfo);
    invStride = shape::stride(invShapeInfo);
    inmShape = shape::shapeOf(inmShapeInfo);
    inmStride = shape::stride(inmShapeInfo);
    stvShape = shape::shapeOf(stvShapeInfo);
    stvStride = shape::stride(stvShapeInfo);
    stmShape = shape::shapeOf(stmShapeInfo);
    stmStride = shape::stride(stmShapeInfo);

    bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) &&
                shape::order(zShapeInfo) == shape::order(stmShapeInfo) &&
                shape::order(stmShapeInfo) == shape::order(inmShapeInfo) &&
                shape::order(inmShapeInfo) == shape::order(stvShapeInfo) &&
                shape::order(stvShapeInfo) == shape::order(invShapeInfo);

    bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    bXInUSame = shape::haveSameShapeAndStrides(xShapeInfo, invShapeInfo);
    bXStUSame = shape::haveSameShapeAndStrides(xShapeInfo, stvShapeInfo);
    bXInMSame = shape::haveSameShapeAndStrides(xShapeInfo, inmShapeInfo);
    bXStMSame = shape::haveSameShapeAndStrides(xShapeInfo, stmShapeInfo);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    LongType xOffset = i, zOffset = i, initMOffset = i, initUOffset = i, stMOffset = i, stUOffset = i;

    if (!bOrdering) {
      INDEX2COORDS(i, xRank, xShape, coords);
      COORDS2INDEX(xRank, xStride, coords, xOffset);

      if (!bXZsame)
        COORDS2INDEX(zRank, zStride, coords, zOffset);
      if (!bXInUSame)
        COORDS2INDEX(invRank, invStride, coords, initUOffset);
      if (!bXStUSame)
        COORDS2INDEX(stvRank, stvStride, coords, stUOffset);
      if (!bXInMSame)
        COORDS2INDEX(inmRank, inmStride, coords, initMOffset);
      if (!bXStMSame)
        COORDS2INDEX(stmRank, stmStride, coords, stMOffset);
    }

    stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * (1 - beta1);
    stU[stUOffset] = beta2 * initU[initUOffset] + grad[xOffset] * grad[xOffset] * (1 - beta2);
    up[zOffset] = (stM[stMOffset] * epsilonT) / (math::sd_sqrt<T, T>(stU[stUOffset]) + epsilon);
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void adamUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                             const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                             const void* vinv, const LongType* invShapeInfo, const void* vinm,
                             const LongType* inmShapeInfo, void* vz, const LongType* zShapeInfo, void* vstV,
                             const LongType* stvShapeInfo, void* vstM, const LongType* stmShapeInfo,
                             const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon,
                             const int nIteration) {
  const T lr = static_cast<T>(dLr);
  const T beta1 = static_cast<T>(dBeta1);
  const T beta2 = static_cast<T>(dBeta2);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  const T iteration = static_cast<T>(nIteration);
  adamUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(
      vx, xShapeInfo, vinv, invShapeInfo, vinm, inmShapeInfo, vz, zShapeInfo, vstV, stvShapeInfo, vstM, stmShapeInfo,
      lr, beta1, beta2, epsilon, iteration);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "adamUpdaterCuda failed");

}

///////////////////////////////////////////////////////////////////
void updaterAdam(LaunchContext* context, NDArray& gradient, NDArray& initStateU,
                 NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr,
                 const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
  PointersManager manager(context, "adamUpdater");

  dim3 launchDims = updaterDims(gradient.lengthOf());
  NDArray::prepareSpecialUse({&update, &stateU, &stateM}, {&gradient, &initStateU, &initStateM});

  BUILD_SINGLE_SELECTOR(gradient.dataType(), adamUpdaterCudaLauncher,
                        (launchDims.y, launchDims.x,launchDims.z, context->getCudaStream(), gradient.specialBuffer(),
                         gradient.specialShapeInfo(), initStateU.specialBuffer(), initStateU.specialShapeInfo(),
                         initStateM.specialBuffer(), initStateM.specialShapeInfo(), update.specialBuffer(),
                         update.specialShapeInfo(), stateU.specialBuffer(), stateU.specialShapeInfo(),
                         stateM.specialBuffer(), stateM.specialShapeInfo(), dLr, dBeta1, dBeta2, dEpsilon, nIteration),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({&update, &stateU, &stateM}, {&gradient, &initStateU, &initStateM});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
