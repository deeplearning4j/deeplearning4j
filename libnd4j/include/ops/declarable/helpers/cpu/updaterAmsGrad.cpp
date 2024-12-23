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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <execution/Threads.h>
#include <math/platformmath.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/updatersHelpers.h>
#if NOT_EXCLUDED(OP_amsgrad_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void amsGradUpdater_(NDArray& gradient, NDArray& initStateV, NDArray& initStateM,
                            NDArray& initStateH, NDArray& update, NDArray& stateV, NDArray& stateM,
                            NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2,
                            const double dEpsilon, const int nIteration) {
  const T* grad = gradient.bufferAsT<T>();
  const T* initV = initStateV.bufferAsT<T>();
  const T* initM = initStateM.bufferAsT<T>();
  const T* initH = initStateH.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* stV = stateV.bufferAsT<T>();
  T* stM = stateM.bufferAsT<T>();
  T* stH = stateH.bufferAsT<T>();

  const T lr = static_cast<T>(dLr);
  const T beta1 = static_cast<T>(dBeta1);
  const T beta2 = static_cast<T>(dBeta2);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  const T iteration = static_cast<T>(nIteration);

  T epsilonT = lr * sd::math::sd_sqrt<T, T>(1.0 - sd::math::sd_pow<T, T, T>(beta2, (iteration + 1))) /
               (1.0 - sd::math::sd_pow<T, T, T>(beta1, (iteration + 1)));

  if (sd::math::sd_isnan(epsilonT) || 0 == epsilonT || sd::math::sd_isinf(epsilonT)) epsilonT = epsilon;

  const T mbeta1 = (1 - beta1);
  const T mbeta2 = (1 - beta2);

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() &&
               1 == stateV.ews() && 1 == initStateV.ews() && 1 == stateH.ews() && 1 == initStateH.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateV.ordering() &&
                       stateV.ordering() == initStateV.ordering() && stateV.ordering() == initStateM.ordering() &&
                       stateM.ordering() == initStateM.ordering() && stateM.ordering() == initStateH.ordering() &&
                       stateH.ordering() == initStateH.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        stM[i] = beta1 * initM[i] + grad[i] * mbeta1;
        stV[i] = beta2 * initV[i] + grad[i] * grad[i] * mbeta2;
        stH[i] = sd::math::sd_max(initH[i], stV[i]);

        up[i] = epsilonT * stM[i] / (sd::math::sd_sqrt<T, T>(stH[i]) + epsilon);
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  // Cache shape and stride information
  sd::LongType gradRank = gradient.rankOf();
  sd::LongType updateRank = update.rankOf();
  sd::LongType initStateVRank = initStateV.rankOf();
  sd::LongType stateVRank = stateV.rankOf();
  sd::LongType initStateMRank = initStateM.rankOf();
  sd::LongType stateMRank = stateM.rankOf();
  sd::LongType initStateHRank = initStateH.rankOf();
  sd::LongType stateHRank = stateH.rankOf();

  sd::LongType *gradShape = shape::shapeOf(gradient.shapeInfo());
  sd::LongType *gradStride = shape::stride(gradient.shapeInfo());
  sd::LongType *updateStride = shape::stride(update.shapeInfo());
  sd::LongType *initStateVStride = shape::stride(initStateV.shapeInfo());
  sd::LongType *stateVStride = shape::stride(stateV.shapeInfo());
  sd::LongType *initStateMStride = shape::stride(initStateM.shapeInfo());
  sd::LongType *stateMStride = shape::stride(stateM.shapeInfo());
  sd::LongType *initStateHStride = shape::stride(initStateH.shapeInfo());
  sd::LongType *stateHStride = shape::stride(stateH.shapeInfo());

  bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
  bool bXInVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateV.shapeInfo());
  bool bXStVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateV.shapeInfo());
  bool bXInMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateM.shapeInfo());
  bool bXStMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateM.shapeInfo());
  bool bXInHSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateH.shapeInfo());
  bool bXStHSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateH.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, gradRank, gradShape, coords);

      sd::LongType xOffset;
      COORDS2INDEX(gradRank, gradStride, coords, xOffset);

      sd::LongType zOffset;
      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(updateRank, updateStride, coords, zOffset);
      }

      sd::LongType initVOffset;
      if (bXInVSame) {
        initVOffset = xOffset;
      } else {
        COORDS2INDEX(initStateVRank, initStateVStride, coords, initVOffset);
      }

      sd::LongType stVOffset;
      if (bXStVSame) {
        stVOffset = xOffset;
      } else {
        COORDS2INDEX(stateVRank, stateVStride, coords, stVOffset);
      }

      sd::LongType initMOffset;
      if (bXInMSame) {
        initMOffset = xOffset;
      } else {
        COORDS2INDEX(initStateMRank, initStateMStride, coords, initMOffset);
      }

      sd::LongType stMOffset;
      if (bXStMSame) {
        stMOffset = xOffset;
      } else {
        COORDS2INDEX(stateMRank, stateMStride, coords, stMOffset);
      }

      sd::LongType initHOffset;
      if (bXInHSame) {
        initHOffset = xOffset;
      } else {
        COORDS2INDEX(initStateHRank, initStateHStride, coords, initHOffset);
      }

      sd::LongType stHOffset;
      if (bXStHSame) {
        stHOffset = xOffset;
      } else {
        COORDS2INDEX(stateHRank, stateHStride, coords, stHOffset);
      }

      stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * mbeta1;
      stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;
      stH[stHOffset] = sd::math::sd_max(initH[initHOffset], stV[stVOffset]);

      up[zOffset] = epsilonT * stM[stMOffset] / (sd::math::sd_sqrt<T, T>(stH[stHOffset]) + epsilon);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterAmsGrad(sd::LaunchContext* context, NDArray& gradient, NDArray& initStateV,
                    NDArray& initStateM, NDArray& initStateH, NDArray& update, NDArray& stateV,
                    NDArray& stateM, NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2,
                    const double dEpsilon, const int nIteration) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), amsGradUpdater_,
                        (gradient, initStateV, initStateM, initStateH, update, stateV, stateM, stateH, dLr, dBeta1,
                            dBeta2, dEpsilon, nIteration),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif