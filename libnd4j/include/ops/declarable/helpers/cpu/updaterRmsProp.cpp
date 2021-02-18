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

#include <ops/declarable/helpers/updatersHelpers.h>
#include <execution/Threads.h>
#include <math/platformmath.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rmsPropUpdater_(const NDArray& gradient, const NDArray& initState, NDArray& update, NDArray& stateG, 
                            const double dLr, const double dRmsDecay, const double dEpsilon) {

    const T* grad = gradient.bufferAsT<T>();
    const T* init = initState.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* st = stateG.bufferAsT<T>();
        
    const T lr = static_cast<T>(dLr);
    const T rmsDecay = static_cast<T>(dRmsDecay);
    const T epsilon = static_cast<T>(dEpsilon);
    
    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateG.ews() && 1 == initState.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() && update.ordering() == stateG.ordering() && stateG.ordering() == initState.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                     st[i] =  init[i] * rmsDecay + grad[i] * grad[i] * (1 - rmsDecay) ;
                     up[i] = (lr * grad[i]) / ( math::nd4j_sqrt<T, T>(st[i]) + epsilon);
                 }
            };

           samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
           return;
    }
    
    bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
    bool bXInSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initState.shapeInfo());
    bool bXStSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateG.shapeInfo());

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < stop; i++) {
            shape::index2coordsCPU(start, i, gradient.shapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.shapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.shapeInfo(), coords);
            const auto initOffset = bXInSame ? xOffset : shape::getOffset(initState.shapeInfo(), coords);
            const auto stOffset = bXStSame ? xOffset : shape::getOffset(stateG.shapeInfo(), coords);
            
            st[stOffset] =  init[initOffset] * rmsDecay + grad[xOffset] * grad[xOffset] * (1 - rmsDecay) ;
            up[zOffset] = (lr * grad[xOffset]) / ( math::nd4j_sqrt<T, T>(st[stOffset]) + epsilon);
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterRmsProp(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState, NDArray& update, NDArray& stateG,
                    const double dLr, const double dRmsDecay, const double dEpsilon) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), rmsPropUpdater_, (gradient, initState, update, stateG, dLr, dRmsDecay, dEpsilon), FLOAT_TYPES);
}


}
}
}
