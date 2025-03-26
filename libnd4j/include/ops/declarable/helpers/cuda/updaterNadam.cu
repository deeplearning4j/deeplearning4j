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
SD_KERNEL void nadamUpdaterCuda(const void* vx, const LongType* xShapeInfo, const void* vinv,
                                const LongType* invShapeInfo, const void* vinm, const LongType* inmShapeInfo,
                                void* vz, const LongType* zShapeInfo, void* vstV, const LongType* stvShapeInfo,
                                void* vstM, const LongType* stmShapeInfo, const T lr, const T beta1, const T beta2,
                                const T epsilon, const T iteration) {
  const auto grad = reinterpret_cast<const T*>(vx);
  const auto initV = reinterpret_cast<const T*>(vinv);
  const auto initM = reinterpret_cast<const T*>(vinm);
  auto up = reinterpret_cast<T*>(vz);
  auto stV = reinterpret_cast<T*>(vstV);
  auto stM = reinterpret_cast<T*>(vstM);

  __shared__ LongType xLen, xRank, zRank, invRank, inmRank, stvRank, stmRank;
  __shared__ T mbeta1T, mbeta1, mbeta2;
  __shared__ bool bOrdering, bXZsame, bXInUSame, bXStUSame, bXInMSame, bXStMSame;
  __shared__ LongType *sharedMem;
  __shared__ const LongType *xShape, *zShape, *invShape, *inmShape, *stvShape, *stmShape;
  __shared__ const LongType *xStride, *zStride, *invStride, *inmStride, *stvStride, *stmStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    xLen = shape::length(xShapeInfo);
    mbeta1T = 1.0 - math::sd_pow<T, T, T>(beta1, (iteration + 1));
    mbeta1 = (1 - beta1);
    mbeta2 = (1 - beta2);

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
    LongType xOffset, zOffset, initMOffset, initUOffset, stMOffset, stUOffset;

    INDEX2COORDS(i, xRank, xShape, coords);
    COORDS2INDEX(xRank, xStride, coords, xOffset);

    if (bXZsame) {
      zOffset = xOffset;
    } else {
      COORDS2INDEX(zRank, zStride, coords, zOffset);
    }

    if (bXInUSame) {
      initUOffset = xOffset;
    } else {
      COORDS2INDEX(invRank, invStride, coords, initUOffset);
    }

    if (bXStUSame) {
      stUOffset = xOffset;
    } else {
      COORDS2INDEX(stvRank, stvStride, coords, stUOffset);
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

    auto oneMinusBeta1Grad = grad[xOffset] * mbeta1;
    stM[stMOffset] = beta1 * initM[initMOffset] + oneMinusBeta1Grad;
    stV[stUOffset] = beta2 * initV[initUOffset] + grad[xOffset] * grad[xOffset] * mbeta2;
    up[zOffset] = (lr * ((stM[stMOffset] * beta1 + oneMinusBeta1Grad) / mbeta1T)) /
                  (math::sd_sqrt<T, T>(stV[stUOffset]) + epsilon);
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void nadamUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
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

  nadamUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(
      vx, xShapeInfo, vinv, invShapeInfo, vinm, inmShapeInfo, vz, zShapeInfo, vstV, stvShapeInfo, vstM, stmShapeInfo,
      lr, beta1, beta2, epsilon, iteration);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "nadamUpdaterCuda failed");

}

///////////////////////////////////////////////////////////////////
void updaterNadam(LaunchContext* context, NDArray& gradient, NDArray& initStateV,
                  NDArray& initStateM, NDArray& update, NDArray& stateV, NDArray& stateM, const double dLr,
                  const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
  PointersManager manager(context, "nadamUpdater");


  dim3 launchDims = updaterDims(gradient.lengthOf());
  NDArray::prepareSpecialUse({&update, &stateV, &stateM}, {&gradient, &initStateV, &initStateM});
  BUILD_SINGLE_SELECTOR(gradient.dataType(), nadamUpdaterCudaLauncher,
                        (launchDims.y, launchDims.x,launchDims.z, context->getCudaStream(), gradient.specialBuffer(),
                            gradient.specialShapeInfo(), initStateV.specialBuffer(), initStateV.specialShapeInfo(),
                            initStateM.specialBuffer(), initStateM.specialShapeInfo(), update.specialBuffer(),
                            update.specialShapeInfo(), stateV.specialBuffer(), stateV.specialShapeInfo(),
                            stateM.specialBuffer(), stateM.specialShapeInfo(), dLr, dBeta1, dBeta2, dEpsilon, nIteration),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&update, &stateV, &stateM}, {&gradient, &initStateV, &initStateM});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
