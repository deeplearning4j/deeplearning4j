/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/updatersHelpers.h>
#include <execution/Threads.h>
#include <math/platformmath.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void nesterovsUpdater_(const NDArray& gradient, const NDArray& initState, NDArray& update, NDArray& stateV, const double dLr, const double dMomentum) {

    const T* grad = gradient.bufferAsT<T>();
    const T* init = initState.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* st = stateV.bufferAsT<T>();

    const T lr = static_cast<T>(dLr);
    const T momentum = static_cast<T>(dMomentum);
    const T momentumT = (-momentum - 1);

    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateV.ews() && 1 == initState.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateV.ordering() && stateV.ordering() == initState.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                     T prevState = momentum * init[i];
                     st[i] = prevState - lr * grad[i];
                     up[i] = prevState + momentumT * st[i];
                 }
            };

           samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
           return;
    }
    
    bool bXZsame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), update.getShapeInfo());
    bool bXInSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), initState.getShapeInfo());
    bool bXStSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), stateV.getShapeInfo());

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < stop; i++) {
            shape::index2coordsCPU(start, i, gradient.getShapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.getShapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.getShapeInfo(), coords);
            const auto initOffset = bXInSame ? xOffset : shape::getOffset(initState.getShapeInfo(), coords);
            const auto stOffset = bXStSame ? xOffset : shape::getOffset(stateV.getShapeInfo(), coords);
            
            T prevState = momentum * init[initOffset];
            st[stOffset] = prevState - lr * grad[xOffset];
            up[zOffset] = prevState + momentumT * st[stOffset];
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterNesterovs(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState, NDArray& update, NDArray& stateV, const double dLr, const double dMomentum) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), nesterovsUpdater_, (gradient, initState, update, stateV, dLr, dMomentum), FLOAT_TYPES);
}

}
}
}
