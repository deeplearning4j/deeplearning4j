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
SD_KERNEL void amsGradUpdaterCuda(const void* vx, const LongType* xShapeInfo, const void* vinv,
                                  const LongType* invShapeInfo, const void* vinm, const LongType* inmShapeInfo,
                                  const void* vinh, const LongType* inhShapeInfo, void* vz,
                                  const LongType* zShapeInfo, void* vstV, const LongType* stvShapeInfo,
                                  void* vstM, const LongType* stmShapeInfo, void* vstH,
                                  const LongType* sthShapeInfo, const T lr, const T beta1, const T beta2,
                                  const T epsilon, const T iteration) {
  const auto grad = reinterpret_cast<const T*>(vx);
  const auto initV = reinterpret_cast<const T*>(vinv);
  const auto initM = reinterpret_cast<const T*>(vinm);
  const auto initH = reinterpret_cast<const T*>(vinh);
  auto up = reinterpret_cast<T*>(vz);
  auto stV = reinterpret_cast<T*>(vstV);
  auto stM = reinterpret_cast<T*>(vstM);
  auto stH = reinterpret_cast<T*>(vstH);

  __shared__ LongType xLen, xRank, zRank, invRank, inmRank, inhRank, stvRank, stmRank, sthRank;
  __shared__ T mbeta1, mbeta2, epsilonT;
  __shared__ bool bOrdering, bXZsame, bXInUSame, bXStUSame, bXInMSame, bXStMSame, bXInHSame, bXStHSame;
  __shared__ LongType *sharedMem;
  __shared__ const LongType *xShape, *zShape, *invShape, *inmShape, *inhShape, *stvShape, *stmShape, *sthShape;
  __shared__ const LongType *xStride, *zStride, *invStride, *inmStride, *inhStride, *stvStride, *stmStride, *sthStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    xLen = shape::length(xShapeInfo);

    epsilonT = lr * math::sd_sqrt<T, T>(1.0 - math::sd_pow<T, T, T>(beta2, (iteration + 1))) /
               (1.0 - math::sd_pow<T, T, T>(beta1, (iteration + 1)));

    if (math::sd_isnan(epsilonT) || 0 == epsilonT || math::sd_isinf(epsilonT)) epsilonT = epsilon;

    mbeta1 = (1 - beta1);
    mbeta2 = (1 - beta2);

    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);
    invRank = shape::rank(invShapeInfo);
    inmRank = shape::rank(inmShapeInfo);
    inhRank = shape::rank(inhShapeInfo);
    stvRank = shape::rank(stvShapeInfo);
    stmRank = shape::rank(stmShapeInfo);
    sthRank = shape::rank(sthShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
    invShape = shape::shapeOf(invShapeInfo);
    invStride = shape::stride(invShapeInfo);
    inmShape = shape::shapeOf(inmShapeInfo);
    inmStride = shape::stride(inmShapeInfo);
    inhShape = shape::shapeOf(inhShapeInfo);
    inhStride = shape::stride(inhShapeInfo);
    stvShape = shape::shapeOf(stvShapeInfo);
    stvStride = shape::stride(stvShapeInfo);
    stmShape = shape::shapeOf(stmShapeInfo);
    stmStride = shape::stride(stmShapeInfo);
    sthShape = shape::shapeOf(sthShapeInfo);
    sthStride = shape::stride(sthShapeInfo);

    bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) &&
                shape::order(zShapeInfo) == shape::order(stmShapeInfo) &&
                shape::order(stmShapeInfo) == shape::order(inmShapeInfo) &&
                shape::order(inmShapeInfo) == shape::order(stvShapeInfo) &&
                shape::order(stvShapeInfo) == shape::order(invShapeInfo) &&
                shape::order(invShapeInfo) == shape::order(sthShapeInfo) &&
                shape::order(sthShapeInfo) == shape::order(inhShapeInfo);

    bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    bXInUSame = shape::haveSameShapeAndStrides(xShapeInfo, invShapeInfo);
    bXStUSame = shape::haveSameShapeAndStrides(xShapeInfo, stvShapeInfo);
    bXInMSame = shape::haveSameShapeAndStrides(xShapeInfo, inmShapeInfo);
    bXStMSame = shape::haveSameShapeAndStrides(xShapeInfo, stmShapeInfo);
    bXInHSame = shape::haveSameShapeAndStrides(xShapeInfo, inhShapeInfo);
    bXStHSame = shape::haveSameShapeAndStrides(xShapeInfo, sthShapeInfo);
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * SD_MAX_RANK;

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    LongType xOffset = i, zOffset = i, initMOffset = i, initVOffset = i, initHOffset = i, stMOffset = i, stVOffset = i,
             stHOffset = i;

    if (!bOrdering) {
      INDEX2COORDS(i, xRank, xShape, coords);
      COORDS2INDEX(xRank, xStride, coords, xOffset);

      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(zRank, zStride, coords, zOffset);
      }

      if (bXInMSame) {
        initMOffset = xOffset;
      } else {
        COORDS2INDEX(inmRank, inmStride, coords, initMOffset);
      }

      if (bXStMSame) {
        stMOffset = xOffset;
      } else {
        COORDS2INDEX(stmRank, stmStride, coords, stMOffset);
      }

      if (bXInUSame) {
        initVOffset = xOffset;
      } else {
        COORDS2INDEX(invRank, invStride, coords, initVOffset);
      }

      if (bXStUSame) {
        stVOffset = xOffset;
      } else {
        COORDS2INDEX(stvRank, stvStride, coords, stVOffset);
      }

      if (bXInHSame) {
        initHOffset = xOffset;
      } else {
        COORDS2INDEX(inhRank, inhStride, coords, initHOffset);
      }

      if (bXStHSame) {
        stHOffset = xOffset;
      } else {
        COORDS2INDEX(sthRank, sthStride, coords, stHOffset);
      }
    }

    stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * mbeta1;
    stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;
    stH[stHOffset] = math::sd_max(initH[initHOffset], stV[stVOffset]);
    up[zOffset] = epsilonT * stM[stMOffset] / (math::sd_sqrt<T, T>(stH[stHOffset]) + epsilon);
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void amsGradUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                                const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                const void* vinv, const LongType* invShapeInfo, const void* vinm,
                                const LongType* inmShapeInfo, const void* vinh, const LongType* inhShapeInfo,
                                void* vz, const LongType* zShapeInfo, void* vstV, const LongType* stvShapeInfo,
                                void* vstM, const LongType* stmShapeInfo, void* vstH,
                                const LongType* sthShapeInfo, const double dLr, const double dBeta1,
                                const double dBeta2, const double dEpsilon, const int nIteration) {
  const T lr = static_cast<T>(dLr);
  const T beta1 = static_cast<T>(dBeta1);
  const T beta2 = static_cast<T>(dBeta2);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  const T iteration = static_cast<T>(nIteration);

  amsGradUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(
      vx, xShapeInfo, vinv, invShapeInfo, vinm, inmShapeInfo, vinh, inhShapeInfo, vz, zShapeInfo, vstV, stvShapeInfo,
      vstM, stmShapeInfo, vstH, sthShapeInfo, lr, beta1, beta2, epsilon, iteration);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "amsGradUpdaterCudaLauncher failed");

}

///////////////////////////////////////////////////////////////////
void updaterAmsGrad(LaunchContext* context, NDArray& gradient, NDArray& initStateV,
                    NDArray& initStateM, NDArray& initStateH, NDArray& update, NDArray& stateV,
                    NDArray& stateM, NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2,
                    const double dEpsilon, const int nIteration) {
  PointersManager manager(context, "amsGradUpdater");
  dim3 launchDims = updaterDims(gradient.lengthOf());
  NDArray::prepareSpecialUse({&update, &stateV, &stateM, &stateH}, {&gradient, &initStateV, &initStateM, &initStateH});
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), amsGradUpdaterCudaLauncher,
      (launchDims.y, launchDims.x,launchDims.z, context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
          initStateV.specialBuffer(), initStateV.specialShapeInfo(), initStateM.specialBuffer(),
          initStateM.specialShapeInfo(), initStateH.specialBuffer(), initStateH.specialShapeInfo(), update.specialBuffer(),
          update.specialShapeInfo(), stateV.specialBuffer(), stateV.specialShapeInfo(), stateM.specialBuffer(),
          stateM.specialShapeInfo(), stateH.specialBuffer(), stateH.specialShapeInfo(), dLr, dBeta1, dBeta2, dEpsilon,
          nIteration),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&update, &stateV, &stateM, &stateH}, {&gradient, &initStateV, &initStateM, &initStateH});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
