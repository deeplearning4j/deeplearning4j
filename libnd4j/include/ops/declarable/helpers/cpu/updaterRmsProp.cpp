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
#if NOT_EXCLUDED(OP_rms_prop_updater)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rmsPropUpdater_(NDArray& gradient, NDArray& initState, NDArray& update, NDArray& stateG,
                            const double dLr, const double dRmsDecay, const double dEpsilon) {
  const T* grad = gradient.bufferAsT<T>();
  const T* init = initState.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* st = stateG.bufferAsT<T>();

  const T lr = static_cast<T>(dLr);
  const T rmsDecay = static_cast<T>(dRmsDecay);
  const T epsilon = static_cast<T>(dEpsilon);

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateG.ews() && 1 == initState.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateG.ordering() &&
                       stateG.ordering() == initState.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        st[i] = init[i] * rmsDecay + grad[i] * grad[i] * (1 - rmsDecay);
        up[i] = (lr * grad[i]) / (math::sd_sqrt<T, T>(st[i]) + epsilon);
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
  bool bXInSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initState.shapeInfo());
  bool bXStSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateG.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, shape::rank(gradient.shapeInfo()), shape::shapeOf(gradient.shapeInfo()), coords);
      sd::LongType xOffset, zOffset, initOffset, stOffset;
      COORDS2INDEX(shape::rank(gradient.shapeInfo()), shape::stride(gradient.shapeInfo()), coords, xOffset);
      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(update.shapeInfo()), shape::stride(update.shapeInfo()), coords, zOffset);
      }
      if (bXInSame) {
        initOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(initState.shapeInfo()), shape::stride(initState.shapeInfo()), coords, initOffset);
      }
      if (bXStSame) {
        stOffset = xOffset;
      } else {
        COORDS2INDEX(shape::rank(stateG.shapeInfo()), shape::stride(stateG.shapeInfo()), coords, stOffset);
      }

      st[stOffset] = init[initOffset] * rmsDecay + grad[xOffset] * grad[xOffset] * (1 - rmsDecay);
      up[zOffset] = (lr * grad[xOffset]) / (math::sd_sqrt<T, T>(st[stOffset]) + epsilon);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterRmsProp(sd::LaunchContext* context, NDArray& gradient, NDArray& initState, NDArray& update,
                    NDArray& stateG, const double dLr, const double dRmsDecay, const double dEpsilon) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), rmsPropUpdater_,
                        (gradient, initState, update, stateG, dLr, dRmsDecay, dEpsilon), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif