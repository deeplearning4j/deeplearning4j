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
SD_KERNEL void adaGradUpdaterCuda(const void* vx, const LongType* xShapeInfo, const void* vin,
                                  const LongType* inShapeInfo, void* vz, const LongType* zShapeInfo, void* vst,
                                  const LongType* stShapeInfo, const T lr, const T epsilon) {
  const auto x = reinterpret_cast<const T*>(vx);
  const auto init = reinterpret_cast<const T*>(vin);

  auto up = reinterpret_cast<T*>(vz);
  auto st = reinterpret_cast<T*>(vst);

  __shared__ bool bEWS, bOrdering, bXZsame, bXInSame, bXStSame;
  __shared__ LongType xLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(xShapeInfo);

    bEWS = 1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
           1 == shape::elementWiseStride(stShapeInfo) && 1 == shape::elementWiseStride(inShapeInfo);
    bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) &&
                shape::order(xShapeInfo) == shape::order(stShapeInfo) &&
                shape::order(xShapeInfo) == shape::order(inShapeInfo);

    bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    bXInSame = shape::haveSameShapeAndStrides(xShapeInfo, inShapeInfo);
    bXStSame = shape::haveSameShapeAndStrides(xShapeInfo, stShapeInfo);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    LongType xOffset, zOffset, initOffset, stOffset;

    if (!bEWS || !bOrdering) {
      INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, coords);
      COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), coords, xOffset);
      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), coords, zOffset);
      }

      if (bXInSame) {
        initOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(inShapeInfo), shape::shapeOf(inShapeInfo), coords, initOffset);
      }

      if (bXStSame) {
        stOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(stShapeInfo), shape::shapeOf(stShapeInfo), coords, stOffset);
      }
    } else {
      xOffset = zOffset = initOffset = stOffset = i;
    }

    st[stOffset] = init[initOffset] + x[xOffset] * x[xOffset];
    up[zOffset] = (lr * x[xOffset]) / (math::sd_sqrt<T, T>(st[stOffset]) + epsilon);
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void adaGradUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMemory,
                                const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                const void* vin, const LongType* inShapeInfo, void* vz,
                                const LongType* zShapeInfo, void* vst, const LongType* stShapeInfo,
                                const double dLr, const double dEpsilon) {
  const T lr = static_cast<T>(dLr);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  adaGradUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(vx, xShapeInfo, vin, inShapeInfo, vz,
                                                                                   zShapeInfo, vst, stShapeInfo, lr, epsilon);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "adaGradUpdaterCuda failed");

}

///////////////////////////////////////////////////////////////////
void updaterAdaGrad(LaunchContext* context, NDArray& gradient, NDArray& initState, NDArray& update,
                    NDArray& stateH, const double dLr, const double dEpsilon) {
  PointersManager manager(context, "adaGradUpdater");

  dim3 launchDims = updaterDims(gradient.lengthOf());
  NDArray::prepareSpecialUse({&update, &stateH}, {&gradient, &initState});
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), adaGradUpdaterCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z,context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
          initState.specialBuffer(), initState.specialShapeInfo(), update.specialBuffer(), update.specialShapeInfo(),
          stateH.specialBuffer(), stateH.specialShapeInfo(), dLr, dEpsilon),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&update, &stateH}, {&gradient, &initState});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
