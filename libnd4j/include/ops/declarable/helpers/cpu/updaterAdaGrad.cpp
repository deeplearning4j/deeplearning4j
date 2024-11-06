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
#if NOT_EXCLUDED(OP_adagrad_updater)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void adaGradUpdater_(NDArray& gradient, NDArray& initState, NDArray& update, NDArray& stateH,
                            const double dLr, const double dEpsilon) {
  const T* grad = gradient.bufferAsT<T>();
  const T* init = initState.bufferAsT<T>();

  T* up = update.bufferAsT<T>();
  T* st = stateH.bufferAsT<T>();

  const T lr = static_cast<T>(dLr);
  T epsilon = static_cast<T>(dEpsilon);
  //fp16 to prevent underflow
  if(epsilon == 0.0) {
    epsilon = static_cast<T>(1e-7);
  }
  bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateH.ews() && 1 == initState.ews();
  bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateH.ordering() &&
                       stateH.ordering() == initState.ordering();

  if (bEws1 && bSameOrdering) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        st[i] = init[i] + grad[i] * grad[i];
        up[i] = (lr * grad[i]) / (math::sd_sqrt<T, T>(st[i]) + epsilon);
      }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
  }

  bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
  bool bXInSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initState.shapeInfo());
  bool bXStSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateH.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (sd::LongType  i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, gradient.shapeInfo(), coords);

      const auto xOffset = shape::getOffset(gradient.shapeInfo(), coords);

      const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.shapeInfo(), coords);
      const auto initOffset = bXInSame ? xOffset : shape::getOffset(initState.shapeInfo(), coords);
      const auto stOffset = bXStSame ? xOffset : shape::getOffset(stateH.shapeInfo(), coords);

      st[stOffset] = init[initOffset] + grad[xOffset] * grad[xOffset];
      up[zOffset] = (lr * grad[xOffset]) / (math::sd_sqrt<T, T>(st[stOffset]) + epsilon);
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
  return;
}

void updaterAdaGrad(sd::LaunchContext* context, NDArray& gradient, NDArray& initState, NDArray& update,
                    NDArray& stateH, const double dLr, const double dEpsilon) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), adaGradUpdater_, (gradient, initState, update, stateH, dLr, dEpsilon),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif