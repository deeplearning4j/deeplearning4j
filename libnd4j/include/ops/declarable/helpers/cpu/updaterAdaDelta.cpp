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
static void adaDeltaUpdater_(const NDArray& gradient, const NDArray& initStateMsg, const NDArray& initStateMsdx, 
                             NDArray& update, NDArray& stateMsg, NDArray& stateMsdx, const double dRho, const double dEpsilon) {

    const T* grad = gradient.bufferAsT<T>();
    const T* initMsg = initStateMsg.bufferAsT<T>();
    const T* initMsdx = initStateMsdx.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* stMsg = stateMsg.bufferAsT<T>();
    T* stMsdx = stateMsdx.bufferAsT<T>();

    const T rho = static_cast<T>(dRho);
    const T epsilon = static_cast<T>(dEpsilon);
    const T rhoT = (1 - rho);
    
    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateMsg.ews() && 1 == initStateMsg.ews() && 1 == stateMsdx.ews() && 1 == initStateMsdx.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() &&
        update.ordering() == stateMsdx.ordering() &&
        stateMsdx.ordering() == initStateMsdx.ordering() &&
        stateMsdx.ordering() == initStateMsg.ordering() && stateMsg.ordering() == initStateMsg.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                      stMsg[i] = rho * initMsg[i] + grad[i] * grad[i] * rhoT;
                     
                      up[i] = grad[i] * (sd::math::nd4j_sqrt<T, T>(initMsdx[i] + epsilon) / sd::math::nd4j_sqrt<T, T>(stMsg[i] + epsilon));

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

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < gradient.lengthOf(); i++) {
            shape::index2coordsCPU(start, i, gradient.shapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.shapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.shapeInfo(), coords);
            const auto initMsgOffset = bXInMsgSame ? xOffset : shape::getOffset(initStateMsg.shapeInfo(), coords);
            const auto stMsgOffset = bXStMsgSame ? xOffset : shape::getOffset(stateMsg.shapeInfo(), coords);
            const auto initMsdxOffset = bXInMsdxSame ? xOffset : shape::getOffset(initStateMsdx.shapeInfo(), coords);
            const auto stMsdxOffset = bXStMsdxSame ? xOffset : shape::getOffset(stateMsdx.shapeInfo(), coords);
            
            
            stMsg[stMsgOffset] = rho * initMsg[initMsgOffset] + grad[xOffset] * grad[xOffset] * rhoT;
            
            up[zOffset] = grad[xOffset] * (sd::math::nd4j_sqrt<T, T>(initMsdx[initMsdxOffset] + epsilon) / sd::math::nd4j_sqrt<T, T>(stMsg[stMsgOffset] + epsilon));
            
            stMsdx[stMsdxOffset] = rho * initMsdx[initMsdxOffset] + up[zOffset] * up[zOffset] * rhoT;
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterAdaDelta(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateMsg, const NDArray& initStateMsdx, 
                      NDArray& update, NDArray& stateMsg, NDArray& stateMsdx, const double dRho, const double dEpsilon) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), adaDeltaUpdater_, (gradient, initStateMsg, initStateMsdx, update, stateMsg, stateMsdx, dRho, dEpsilon), FLOAT_TYPES);
}

}
}
}
