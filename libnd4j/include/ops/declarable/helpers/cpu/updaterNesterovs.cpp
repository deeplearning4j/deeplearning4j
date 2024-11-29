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
#if NOT_EXCLUDED(OP_nesterovs_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void nesterovsUpdater_(NDArray& gradient, NDArray& initState, NDArray& update, NDArray& stateV,
                              const double dLr, const double dMomentum) {
  const T* grad = gradient.bufferAsT<T>();
  const T* init = initState.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* st = stateV.bufferAsT<T>();

  const T lr = static_cast<T>(dLr);
  const T momentum = static_cast<T>(dMomentum);
  const T momentumT = (-momentum - 1);

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateV.ews() && 1 == initState.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateV.ordering() &&
                       stateV.ordering() == initState.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        T prevState = momentum * init[i];
        st[i] = prevState - lr * grad[i];
        up[i] = prevState + momentumT * st[i];
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
  bool bXInSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initState.shapeInfo());
  bool bXStSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateV.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, gradient.rankOf(), gradient.shapeInfo(), coords);

      sd::LongType xOffset;
      COORDS2INDEX(gradient.rankOf(), shape::stride(gradient.shapeInfo()), coords, xOffset);

      sd::LongType zOffset;
      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(update.rankOf(), shape::stride(update.shapeInfo()), coords, zOffset);
      }

      sd::LongType initOffset;
      if (bXInSame) {
        initOffset = xOffset;
      } else {
        COORDS2INDEX(initState.rankOf(), shape::stride(initState.shapeInfo()), coords, initOffset);
      }

      sd::LongType stOffset;
      if (bXStSame) {
        stOffset = xOffset;
      } else {
        COORDS2INDEX(stateV.rankOf(), shape::stride(stateV.shapeInfo()), coords, stOffset);
      }

      T prevState = momentum * init[initOffset];
      st[stOffset] = prevState - lr * grad[xOffset];
      up[zOffset] = prevState + momentumT * st[stOffset];
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterNesterovs(sd::LaunchContext* context, NDArray& gradient, NDArray& initState, NDArray& update,
                      NDArray& stateV, const double dLr, const double dMomentum) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), nesterovsUpdater_, (gradient, initState, update, stateV, dLr, dMomentum),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif