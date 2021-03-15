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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void nadamUpdater_(const NDArray& gradient, const NDArray& initStateV, const NDArray& initStateM, 
                          NDArray& update, NDArray& stateV, NDArray& stateM, const double dLr, const double dBeta1, 
                          const double dBeta2, const double dEpsilon, const int nIteration) {

    const T* grad = gradient.bufferAsT<T>();
    const T* initV = initStateV.bufferAsT<T>();
    const T* initM = initStateM.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* stV = stateV.bufferAsT<T>();
    T* stM = stateM.bufferAsT<T>();

    const T lr = static_cast<T>(dLr);
    const T beta1 = static_cast<T>(dBeta1);
    const T beta2 = static_cast<T>(dBeta2);
    const T epsilon = static_cast<T>(dEpsilon);
    const T iteration = static_cast<T>(nIteration);

    const T mbeta1T = 1.0 - sd::math::nd4j_pow<T, T, T>(beta1, (iteration + 1));
    const T mbeta1 = (1 - beta1);
    const T mbeta2 = (1 - beta2);

    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() && 1 == stateV.ews() && 1 == initStateV.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() &&
        update.ordering() == stateV.ordering() &&
        stateV.ordering() == initStateV.ordering() &&
        stateV.ordering() == initStateM.ordering() && stateM.ordering() == initStateM.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                     auto oneMinusBeta1Grad = grad[i] * mbeta1;

                     stM[i] = beta1 * initM[i] + oneMinusBeta1Grad;
                     stV[i] = beta2 * initV[i] + grad[i] * grad[i] * mbeta2;

                     up[i] = (lr * ((stM[i] * beta1 + oneMinusBeta1Grad) / mbeta1T)) / (sd::math::nd4j_sqrt<T, T>(stV[i]) + epsilon);
                 }
            };

           samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
           return;
    }
    
    bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
    bool bXInVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateV.shapeInfo());
    bool bXStVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateV.shapeInfo());
    bool bXInMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateM.shapeInfo());
    bool bXStMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateM.shapeInfo());

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < stop; i++) {
            shape::index2coordsCPU(start, i, gradient.shapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.shapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.shapeInfo(), coords);
            const auto initVOffset = bXInVSame ? xOffset : shape::getOffset(initStateV.shapeInfo(), coords);
            const auto stVOffset = bXStVSame ? xOffset : shape::getOffset(stateV.shapeInfo(), coords);
            const auto initMOffset = bXInMSame ? xOffset : shape::getOffset(initStateM.shapeInfo(), coords);
            const auto stMOffset = bXStMSame ? xOffset : shape::getOffset(stateM.shapeInfo(), coords);
            
            auto oneMinusBeta1Grad = grad[xOffset] * mbeta1;

            stM[stMOffset] = beta1 * initM[initMOffset] + oneMinusBeta1Grad;
            stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;

            up[zOffset] = (lr * ((stM[stMOffset] * beta1 + oneMinusBeta1Grad) / mbeta1T)) / (sd::math::nd4j_sqrt<T, T>(stV[stVOffset]) + epsilon);
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterNadam(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateV, const NDArray& initStateM,
                 NDArray& update, NDArray& stateV, NDArray& stateM, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), nadamUpdater_, (gradient, initStateV, initStateM, update, stateV, stateM, dLr, dBeta1, dBeta2, dEpsilon, nIteration), FLOAT_TYPES);
}


}
}
}
