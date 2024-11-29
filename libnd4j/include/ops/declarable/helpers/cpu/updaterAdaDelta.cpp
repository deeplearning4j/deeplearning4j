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
#if NOT_EXCLUDED(OP_adadelta_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void adaDeltaUpdater_(NDArray& gradient, NDArray& initStateMsg, NDArray& initStateMsdx,
                             NDArray& update, NDArray& stateMsg, NDArray& stateMsdx, const double dRho,
                             const double dEpsilon) {
  const T* grad = gradient.bufferAsT<T>();
  const T* initMsg = initStateMsg.bufferAsT<T>();
  const T* initMsdx = initStateMsdx.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* stMsg = stateMsg.bufferAsT<T>();
  T* stMsdx = stateMsdx.bufferAsT<T>();

  const T rho = static_cast<T>(dRho);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  const T rhoT = (1 - rho);

  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateMsg.ews() && 1 == initStateMsg.ews() &&
               1 == stateMsdx.ews() && 1 == initStateMsdx.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateMsdx.ordering() &&
                       stateMsdx.ordering() == initStateMsdx.ordering() &&
                       stateMsdx.ordering() == initStateMsg.ordering() &&
                       stateMsg.ordering() == initStateMsg.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        stMsg[i] = rho * initMsg[i] + grad[i] * grad[i] * rhoT;

        up[i] =
            grad[i] * (sd::math::sd_sqrt<T, T>(initMsdx[i] + epsilon) / sd::math::sd_sqrt<T, T>(stMsg[i] + epsilon));

        stMsdx[i] = rho * initMsdx[i] + up[i] * up[i] * rhoT;
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
  bool bXInMsgSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateMsg.shapeInfo());
  bool bXStMsgSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateMsg.shapeInfo());
  bool bXInMsdxSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateMsdx.shapeInfo());
  bool bXStMsdxSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateMsdx.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (sd::LongType i = start; i < gradient.lengthOf(); i++) {
      INDEX2COORDS(i, gradient.rankOf(), gradient.shapeInfo(), coords);

      sd::LongType xOffset;
      COORDS2INDEX(gradient.rankOf(), shape::stride(gradient.shapeInfo()), coords, xOffset);

      sd::LongType zOffset;
      if (bXZsame) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(update.rankOf(), shape::stride(update.shapeInfo()), coords, zOffset);
      }

      sd::LongType initMsgOffset;
      if (bXInMsgSame) {
        initMsgOffset = xOffset;
      } else {
        COORDS2INDEX(initStateMsg.rankOf(), shape::stride(initStateMsg.shapeInfo()), coords, initMsgOffset);
      }

      sd::LongType stMsgOffset;
      if (bXStMsgSame) {
        stMsgOffset = xOffset;
      } else {
        COORDS2INDEX(stateMsg.rankOf(), shape::stride(stateMsg.shapeInfo()), coords, stMsgOffset);
      }

      sd::LongType initMsdxOffset;
      if (bXInMsdxSame) {
        initMsdxOffset = xOffset;
      } else {
        COORDS2INDEX(initStateMsdx.rankOf(), shape::stride(initStateMsdx.shapeInfo()), coords, initMsdxOffset);
      }

      sd::LongType stMsdxOffset;
      if (bXStMsdxSame) {
        stMsdxOffset = xOffset;
      } else {
        COORDS2INDEX(stateMsdx.rankOf(), shape::stride(stateMsdx.shapeInfo()), coords, stMsdxOffset);
      }

      stMsg[stMsgOffset] = rho * initMsg[initMsgOffset] + grad[xOffset] * grad[xOffset] * rhoT;

      up[zOffset] = grad[xOffset] * (sd::math::sd_sqrt<T, T>(initMsdx[initMsdxOffset] + epsilon) /
                                     sd::math::sd_sqrt<T, T>(stMsg[stMsgOffset] + epsilon));

      stMsdx[stMsdxOffset] = rho * initMsdx[initMsdxOffset] + up[zOffset] * up[zOffset] * rhoT;
    }
  };
  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterAdaDelta(sd::LaunchContext* context, NDArray& gradient, NDArray& initStateMsg,
                     NDArray& initStateMsdx, NDArray& update, NDArray& stateMsg, NDArray& stateMsdx,
                     const double dRho, const double dEpsilon) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), adaDeltaUpdater_,
                        (gradient, initStateMsg, initStateMsdx, update, stateMsg, stateMsdx, dRho, dEpsilon),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif