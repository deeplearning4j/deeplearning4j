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

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void adaGradUpdaterCuda(const void* vx, const sd::LongType* xShapeInfo, const void* vin,
                                  const sd::LongType* inShapeInfo, void* vz, const sd::LongType* zShapeInfo, void* vst,
                                  const sd::LongType* stShapeInfo, const T lr, const T epsilon) {
  const auto x = reinterpret_cast<const T*>(vx);
  const auto init = reinterpret_cast<const T*>(vin);

  auto up = reinterpret_cast<T*>(vz);
  auto st = reinterpret_cast<T*>(vst);

  __shared__ bool bEWS, bOrdering, bXZsame, bXInSame, bXStSame;
  __shared__ sd::LongType xLen;

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

  int coords[SD_MAX_RANK];

  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    auto xOffset = i, zOffset = i, initOffset = i, stOffset = i;

    if (!bEWS || !bOrdering) {
      shape::index2coords(i, xShapeInfo, coords);
      xOffset = shape::getOffset(xShapeInfo, coords);
      zOffset = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
      initOffset = bXInSame ? xOffset : shape::getOffset(inShapeInfo, coords);
      stOffset = bXStSame ? xOffset : shape::getOffset(stShapeInfo, coords);
    }

    st[stOffset] = init[initOffset] + x[xOffset] * x[xOffset];
    up[zOffset] = (lr * x[xOffset]) / (math::sd_sqrt<T, T>(st[stOffset]) + epsilon);
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void adaGradUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream,
                                const void* vx, const sd::LongType* xShapeInfo, const void* vin,
                                const sd::LongType* inShapeInfo, void* vz, const sd::LongType* zShapeInfo, void* vst,
                                const sd::LongType* stShapeInfo, const double dLr, const double dEpsilon) {
  const T lr = static_cast<T>(dLr);
  const T epsilon = static_cast<T>(dEpsilon);

  adaGradUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, vin, inShapeInfo, vz,
                                                                          zShapeInfo, vst, stShapeInfo, lr, epsilon);
}

///////////////////////////////////////////////////////////////////
void updaterAdaGrad(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState, NDArray& update,
                    NDArray& stateH, const double dLr, const double dEpsilon) {
  PointersManager manager(context, "adaGradUpdater");

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

  NDArray::prepareSpecialUse({&update, &stateH}, {&gradient, &initState});
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), adaGradUpdaterCudaLauncher,
      (blocksPerGrid, threadsPerBlock, context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
       initState.specialBuffer(), initState.specialShapeInfo(), update.specialBuffer(), update.specialShapeInfo(),
       stateH.specialBuffer(), stateH.specialShapeInfo(), dLr, dEpsilon),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&update, &stateH}, {&gradient, &initState});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
