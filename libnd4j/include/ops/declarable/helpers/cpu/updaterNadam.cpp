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
#if NOT_EXCLUDED(OP_nadam_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void nadamUpdater_(NDArray& gradient, NDArray& initStateV, NDArray& initStateM,
                          NDArray& update, NDArray& stateV, NDArray& stateM, const double dLr, const double dBeta1,
                          const double dBeta2, const double dEpsilon, const int nIteration) {
  // Cache shape information
  const auto gradientShapeInfo = gradient.shapeInfo();
  const auto updateShapeInfo = update.shapeInfo();
  const auto initStateVShapeInfo = initStateV.shapeInfo();
  const auto stateVShapeInfo = stateV.shapeInfo();
  const auto initStateMShapeInfo = initStateM.shapeInfo();
  const auto stateMShapeInfo = stateM.shapeInfo();
  
  const auto gradRank = shape::rank(gradientShapeInfo);
  const auto* gradShape = shape::shapeOf(gradientShapeInfo);
  const auto* gradStride = shape::stride(gradientShapeInfo);
  const auto* updateStride = shape::stride(updateShapeInfo);
  const auto* initStateVStride = shape::stride(initStateVShapeInfo);
  const auto* stateVStride = shape::stride(stateVShapeInfo);
  const auto* initStateMStride = shape::stride(initStateMShapeInfo);
  const auto* stateMStride = shape::stride(stateMShapeInfo);

  const T* grad = gradient.bufferAsT<T>();
  const T* initV = initStateV.bufferAsT<T>();
  const T* initM = initStateM.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* stV = stateV.bufferAsT<T>();
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

  const T mbeta1T = 1.0 - sd::math::sd_pow<T, T, T>(beta1, (iteration + 1));
  const T mbeta1 = (1 - beta1);
  const T mbeta2 = (1 - beta2);

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() &&
               1 == stateV.ews() && 1 == initStateV.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateV.ordering() &&
                       stateV.ordering() == initStateV.ordering() && stateV.ordering() == initStateM.ordering() &&
                       stateM.ordering() == initStateM.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto oneMinusBeta1Grad = grad[i] * mbeta1;

        stM[i] = beta1 * initM[i] + oneMinusBeta1Grad;
        stV[i] = beta2 * initV[i] + grad[i] * grad[i] * mbeta2;

        up[i] = (lr * ((stM[i] * beta1 + oneMinusBeta1Grad) / mbeta1T)) / (sd::math::sd_sqrt<T, T>(stV[i]) + epsilon);
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradientShapeInfo, updateShapeInfo);
  bool bXInVSame = shape::haveSameShapeAndStrides(gradientShapeInfo, initStateVShapeInfo);
  bool bXStVSame = shape::haveSameShapeAndStrides(gradientShapeInfo, stateVShapeInfo);
  bool bXInMSame = shape::haveSameShapeAndStrides(gradientShapeInfo, initStateMShapeInfo);
  bool bXStMSame = shape::haveSameShapeAndStrides(gradientShapeInfo, stateMShapeInfo);

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
        COORDS2INDEX(gradRank, updateStride, coords, zOffset);
      }

      sd::LongType initVOffset;
      if (bXInVSame) {
        initVOffset = xOffset;
      } else {
        COORDS2INDEX(gradRank, initStateVStride, coords, initVOffset);
      }

      sd::LongType stVOffset;
      if (bXStVSame) {
        stVOffset = xOffset;
      } else {
        COORDS2INDEX(gradRank, stateVStride, coords, stVOffset);
      }

      sd::LongType initMOffset;
      if (bXInMSame) {
        initMOffset = xOffset;
      } else {
        COORDS2INDEX(gradRank, initStateMStride, coords, initMOffset);
      }

      sd::LongType stMOffset;
      if (bXStMSame) {
        stMOffset = xOffset;
      } else {
        COORDS2INDEX(gradRank, stateMStride, coords, stMOffset);
      }

      auto oneMinusBeta1Grad = grad[xOffset] * mbeta1;

      stM[stMOffset] = beta1 * initM[initMOffset] + oneMinusBeta1Grad;
      stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;

      up[zOffset] = (lr * ((stM[stMOffset] * beta1 + oneMinusBeta1Grad) / mbeta1T)) /
                    (sd::math::sd_sqrt<T, T>(stV[stVOffset]) + epsilon);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterNadam(sd::LaunchContext* context, NDArray& gradient, NDArray& initStateV,
                  NDArray& initStateM, NDArray& update, NDArray& stateV, NDArray& stateM, const double dLr,
                  const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
  BUILD_SINGLE_SELECTOR(
      gradient.dataType(), nadamUpdater_,
      (gradient, initStateV, initStateM, update, stateV, stateM, dLr, dBeta1, dBeta2, dEpsilon, nIteration),
      SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif