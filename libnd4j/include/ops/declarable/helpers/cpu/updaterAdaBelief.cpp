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
// @author Abdelrauf (rauf@konduit.ai)

// https://arxiv.org/pdf/2010.07468.pdf
#include <execution/Threads.h>
#include <math/platformmath.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/updatersHelpers.h>
#if NOT_EXCLUDED(OP_adabelief_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void adaBeliefUpdater_(NDArray& gradient, NDArray& initStateU, NDArray& initStateM,
                              NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr, const double dBeta1,
                              const double dBeta2, const double dEpsilon, const int nIteration) {
  // Cache shape information
  const auto gradientShapeInfo = gradient.shapeInfo();
  const auto updateShapeInfo = update.shapeInfo();
  const auto initStateUShapeInfo = initStateU.shapeInfo();
  const auto stateUShapeInfo = stateU.shapeInfo();
  const auto initStateMShapeInfo = initStateM.shapeInfo();
  const auto stateMShapeInfo = stateM.shapeInfo();
  
  const auto gradRank = shape::rank(gradientShapeInfo);
  const auto* gradShape = shape::shapeOf(gradientShapeInfo);
  const auto* gradStride = shape::stride(gradientShapeInfo);
  const auto* updateStride = shape::stride(updateShapeInfo);
  const auto* initStateUStride = shape::stride(initStateUShapeInfo);
  const auto* stateUStride = shape::stride(stateUShapeInfo);
  const auto* initStateMStride = shape::stride(initStateMShapeInfo);
  const auto* stateMStride = shape::stride(stateMShapeInfo);

  const T* grad = gradient.bufferAsT<T>();
  const T* initU = initStateU.bufferAsT<T>();
  const T* initM = initStateM.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* stU = stateU.bufferAsT<T>();
  T* stM = stateM.bufferAsT<T>();

  const T lr = static_cast<T>(dLr);
  const T beta1 = static_cast<T>(dBeta1);
  const T beta2 = static_cast<T>(dBeta2);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  const T iteration = static_cast<T>(nIteration);

  const T beta1T = sd::math::sd_pow<T, T, T>(beta1, (iteration + 1));
  const T beta2T = sd::math::sd_pow<T, T, T>(beta2, (iteration + 1));

  T epsilonT = lr * sd::math::sd_sqrt<T, T>(1. - beta2T) / (1.0 - beta1T);
  if (sd::math::sd_isnan(epsilonT) || 0 == epsilonT || sd::math::sd_isinf(epsilonT)) epsilonT = epsilon;

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() &&
               1 == stateU.ews() && 1 == initStateU.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateU.ordering() &&
                       stateU.ordering() == initStateU.ordering() && stateU.ordering() == initStateM.ordering() &&
                       stateM.ordering() == initStateM.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        stM[i] = beta1 * initM[i] + grad[i] * (1 - beta1);
        stU[i] = beta2 * initU[i] + (grad[i] - stM[i]) * (grad[i] - stM[i]) * (1 - beta2) + epsilon;

        up[i] = (stM[i] * epsilonT) / (sd::math::sd_sqrt<T, T>(stU[i]) + epsilon);
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradientShapeInfo, updateShapeInfo);
  bool bXInVSame = shape::haveSameShapeAndStrides(gradientShapeInfo, initStateUShapeInfo);
  bool bXStVSame = shape::haveSameShapeAndStrides(gradientShapeInfo, stateUShapeInfo);
  bool bXInMSame = shape::haveSameShapeAndStrides(gradientShapeInfo, initStateMShapeInfo);
  bool bXStMSame = shape::haveSameShapeAndStrides(gradientShapeInfo, stateMShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (sd::LongType  i = start; i < stop; i++) {
      INDEX2COORDS(i, gradRank, gradShape, coords);
      sd::LongType xOffset, zOffset, initUOffset, stUOffset, initMOffset, stMOffset;
      COORDS2INDEX(gradRank, gradStride, coords, xOffset);
      COORDS2INDEX(gradRank, updateStride, coords, zOffset);
      COORDS2INDEX(gradRank, initStateUStride, coords, initUOffset);
      COORDS2INDEX(gradRank, stateUStride, coords, stUOffset);
      COORDS2INDEX(gradRank, initStateMStride, coords, initMOffset);
      COORDS2INDEX(gradRank, stateMStride, coords, stMOffset);
      
      stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * (1 - beta1);
      stU[stUOffset] = beta2 * initU[initUOffset] +
                       (grad[xOffset] - stM[stMOffset]) * (grad[xOffset] - stM[stMOffset]) * (1 - beta2) + epsilon;

      up[zOffset] = (stM[stMOffset] * epsilonT) / (sd::math::sd_sqrt<T, T>(stU[stUOffset]) + epsilon);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterAdaBelief(sd::LaunchContext* context, NDArray& gradient, NDArray& initStateU,
                      NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr,
                      const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), adaBeliefUpdater_,
      (gradient, initStateU, initStateM, update, stateU, stateM, dLr, dBeta1, dBeta2, dEpsilon, nIteration),
      SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif