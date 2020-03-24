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
static void amsGradUpdater_(const NDArray& gradient, const NDArray& initStateV, const NDArray& initStateM, const NDArray& initStateH, 
                            NDArray& update, NDArray& stateV, NDArray& stateM, NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {

    const T* grad = gradient.bufferAsT<T>();
    const T* initV = initStateV.bufferAsT<T>();
    const T* initM = initStateM.bufferAsT<T>();
    const T* initH = initStateH.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* stV = stateV.bufferAsT<T>();
    T* stM = stateM.bufferAsT<T>();
    T* stH = stateH.bufferAsT<T>();

    const T lr = static_cast<T>(dLr);
    const T beta1 = static_cast<T>(dBeta1);
    const T beta2 = static_cast<T>(dBeta2);
    const T epsilon = static_cast<T>(dEpsilon);
    const T iteration = static_cast<T>(nIteration);

    T epsilonT = lr * sd::math::nd4j_sqrt<T, T>(1.0 - sd::math::nd4j_pow<T, T, T>(beta2, (iteration + 1))) / (1.0 - sd::math::nd4j_pow<T, T, T>(beta1, (iteration + 1)));

    if (sd::math::nd4j_isnan(epsilonT) || 0 == epsilonT || sd::math::nd4j_isinf(epsilonT))
        epsilonT = epsilon;
    
    const T mbeta1 = (1 - beta1);
    const T mbeta2 = (1 - beta2);

    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() &&
                 1 == stateV.ews() && 1 == initStateV.ews() && 1 == stateH.ews() && 1 == initStateH.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() &&
        update.ordering() == stateV.ordering() &&
        stateV.ordering() == initStateV.ordering() &&
        stateV.ordering() == initStateM.ordering() &&
        stateM.ordering() == initStateM.ordering() &&
        stateM.ordering() == initStateH.ordering() && stateH.ordering() == initStateH.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                     stM[i] = beta1 * initM[i] + grad[i] * mbeta1;
                     stV[i] = beta2 * initV[i] + grad[i] * grad[i] * mbeta2;
                     stH[i] = sd::math::nd4j_max(initH[i], stV[i]);

                     up[i] = epsilonT * stM[i] / (sd::math::nd4j_sqrt<T, T>(stH[i]) + epsilon);
                 }
            };

           samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
           return;
    }
    
    bool bXZsame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), update.getShapeInfo());
    bool bXInVSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), initStateV.getShapeInfo());
    bool bXStVSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), stateV.getShapeInfo());
    bool bXInMSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), initStateM.getShapeInfo());
    bool bXStMSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), stateM.getShapeInfo());
    bool bXInHSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), initStateH.getShapeInfo());
    bool bXStHSame = shape::haveSameShapeAndStrides(gradient.getShapeInfo(), stateH.getShapeInfo());

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < stop; i++) {
            shape::index2coordsCPU(start, i, gradient.getShapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.getShapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.getShapeInfo(), coords);
            const auto initVOffset = bXInVSame ? xOffset : shape::getOffset(initStateV.getShapeInfo(), coords);
            const auto stVOffset = bXStVSame ? xOffset : shape::getOffset(stateV.getShapeInfo(), coords);
            const auto initMOffset = bXInMSame ? xOffset : shape::getOffset(initStateM.getShapeInfo(), coords);
            const auto stMOffset = bXStMSame ? xOffset : shape::getOffset(stateM.getShapeInfo(), coords);
            const auto initHOffset = bXInHSame ? xOffset : shape::getOffset(initStateH.getShapeInfo(), coords);
            const auto stHOffset = bXStHSame ? xOffset : shape::getOffset(stateH.getShapeInfo(), coords);
            
            stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * mbeta1;
            stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;
            stH[stHOffset] = sd::math::nd4j_max(initH[initHOffset], stV[stVOffset]);

            up[zOffset] = epsilonT * stM[stMOffset] / (sd::math::nd4j_sqrt<T, T>(stH[stHOffset]) + epsilon);
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterAmsGrad(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateV, const NDArray& initStateM, const NDArray& initStateH, 
                   NDArray& update, NDArray& stateV, NDArray& stateM, NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), amsGradUpdater_, (gradient, initStateV, initStateM, initStateH, update, stateV, stateM, stateH, dLr, dBeta1, dBeta2, dEpsilon, nIteration), FLOAT_TYPES);
}


}
}
}
