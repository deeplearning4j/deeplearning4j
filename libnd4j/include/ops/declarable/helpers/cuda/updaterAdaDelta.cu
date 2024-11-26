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
SD_KERNEL void adaDeltaUpdaterCuda(const void* vx, const LongType* xShapeInfo, const void* vinMsg,
                                   const LongType* inMsgShapeInfo, const void* vinMsdx,
                                   const LongType* inMsdxShapeInfo, void* vz, const LongType* zShapeInfo,
                                   void* vstMsg, const LongType* stMsgShapeInfo, void* vstMsdx,
                                   const LongType* stMsdxShapeInfo, const T rho, const T epsilon) {
  const auto grad = reinterpret_cast<const T*>(vx);
  const auto initMsg = reinterpret_cast<const T*>(vinMsg);
  const auto initMsdx = reinterpret_cast<const T*>(vinMsdx);

  auto up = reinterpret_cast<T*>(vz);
  auto stMsg = reinterpret_cast<T*>(vstMsg);
  auto stMsdx = reinterpret_cast<T*>(vstMsdx);

  __shared__ LongType xLen;
  __shared__ T rhoT;
  __shared__ bool bOrdering, bXZsame, bXInMsgSame, bXStMsgSame, bXInMsdxSame, bXStMsdxSame;

  if (threadIdx.x == 0) {
    xLen = shape::length(xShapeInfo);

    rhoT = (1 - rho);

    bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) &&
                shape::order(zShapeInfo) == shape::order(stMsgShapeInfo) &&
                shape::order(stMsgShapeInfo) == shape::order(inMsgShapeInfo) &&
                shape::order(inMsgShapeInfo) == shape::order(stMsdxShapeInfo) &&
                shape::order(stMsdxShapeInfo) == shape::order(inMsdxShapeInfo);

    bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    bXInMsgSame = shape::haveSameShapeAndStrides(xShapeInfo, inMsgShapeInfo);
    bXStMsgSame = shape::haveSameShapeAndStrides(xShapeInfo, stMsgShapeInfo);
    bXInMsdxSame = shape::haveSameShapeAndStrides(xShapeInfo, inMsdxShapeInfo);
    bXStMsdxSame = shape::haveSameShapeAndStrides(xShapeInfo, stMsdxShapeInfo);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    LongType xOffset, zOffset, initMsgOffset, initMsdxOffset, stMsgOffset, stMsdxOffset;

    INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, coords);
    COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), coords, xOffset);
    zOffset = bXZsame ? xOffset : COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), coords, zOffset);
    initMsgOffset = bXInMsgSame ? xOffset : COORDS2INDEX(shape::rank(inMsgShapeInfo), shape::shapeOf(inMsgShapeInfo), coords, initMsgOffset);
    stMsgOffset = bXStMsgSame ? xOffset : COORDS2INDEX(shape::rank(stMsgShapeInfo), shape::shapeOf(stMsgShapeInfo), coords, stMsgOffset);
    initMsdxOffset = bXInMsdxSame ? xOffset : COORDS2INDEX(shape::rank(inMsdxShapeInfo), shape::shapeOf(inMsdxShapeInfo), coords, initMsdxOffset);
    stMsdxOffset = bXStMsdxSame ? xOffset : COORDS2INDEX(shape::rank(stMsdxShapeInfo), shape::shapeOf(stMsdxShapeInfo), coords, stMsdxOffset);

    stMsg[stMsgOffset] = rho * initMsg[initMsgOffset] + grad[xOffset] * grad[xOffset] * rhoT;

    up[zOffset] = grad[xOffset] * (math::sd_sqrt<T, T>(initMsdx[initMsdxOffset] + epsilon) /
                                   math::sd_sqrt<T, T>(stMsg[stMsgOffset] + epsilon));

    stMsdx[stMsdxOffset] = rho * initMsdx[initMsdxOffset] + up[zOffset] * up[zOffset] * rhoT;
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void adaDeltaUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                                 const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                 const void* vinMsg, const LongType* inMsgShapeInfo, const void* vinMsdx,
                                 const LongType* inMsdxShapeInfo, void* vz, const LongType* zShapeInfo,
                                 void* vstMsg, const LongType* stMsgShapeInfo, void* vstMsdx,
                                 const LongType* stMsdxShapeInfo, const double dRho, const double dEpsilon) {
  const T rho = static_cast<T>(dRho);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  adaDeltaUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(
      vx, xShapeInfo, vinMsg, inMsgShapeInfo, vinMsdx, inMsdxShapeInfo, vz, zShapeInfo, vstMsg, stMsgShapeInfo, vstMsdx,
      stMsdxShapeInfo, rho, epsilon);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "adaDeltaUpdaterCuda failed");

}

///////////////////////////////////////////////////////////////////
void updaterAdaDelta(LaunchContext* context, NDArray& gradient, NDArray& initStateMsg,
                     NDArray& initStateMsdx, NDArray& update, NDArray& stateMsg, NDArray& stateMsdx,
                     const double dRho, const double dEpsilon) {
  PointersManager manager(context, "adaDeltaUpdater");
  dim3 updater2Dims = updaterDims(gradient.lengthOf());

  NDArray::prepareSpecialUse({&update, &stateMsg, &stateMsdx}, {&gradient, &initStateMsg, &initStateMsdx});
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), adaDeltaUpdaterCudaLauncher,
      (updater2Dims.y, updater2Dims.x,updater2Dims.z, context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
          initStateMsg.specialBuffer(), initStateMsg.specialShapeInfo(), initStateMsdx.specialBuffer(),
          initStateMsdx.specialShapeInfo(), update.specialBuffer(), update.specialShapeInfo(), stateMsg.specialBuffer(),
          stateMsg.specialShapeInfo(), stateMsdx.specialBuffer(), stateMsdx.specialShapeInfo(), dRho, dEpsilon),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&update, &stateMsg, &stateMsdx}, {&gradient, &initStateMsg, &initStateMsdx});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
