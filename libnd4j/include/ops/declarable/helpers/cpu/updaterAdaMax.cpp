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
static void adaMaxUpdater_(const NDArray& gradient, const NDArray& initStateU, const NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {

    const T* grad = gradient.bufferAsT<T>();
    const T* initU = initStateU.bufferAsT<T>();
    const T* initM = initStateM.bufferAsT<T>();

    T* up = update.bufferAsT<T>();
    T* stU = stateU.bufferAsT<T>();
    T* stM = stateM.bufferAsT<T>();

    const T lr = static_cast<T>(dLr);
    const T beta1 = static_cast<T>(dBeta1);
    const T beta2 = static_cast<T>(dBeta2);   
    const T epsilon = static_cast<T>(dEpsilon);
    const T iteration = static_cast<T>(nIteration);
    const T beta1T = sd::math::nd4j_pow<T, T, T>(beta1, (iteration + 1));
    T epsilonT = lr / (1.0 - beta1T);
    if (sd::math::nd4j_isnan(epsilonT) || 0 == epsilonT || sd::math::nd4j_isinf(epsilonT))
        epsilonT = epsilon;
        
    
    bool bEws1 = 1 == gradient.ews() && 1 == update.ews() && 1 == stateM.ews() && 1 == initStateM.ews() && 1 == stateU.ews() && 1 == initStateU.ews();
    bool bSameOrdering = gradient.ordering() == update.ordering() &&
        update.ordering() == stateU.ordering() &&
        stateU.ordering() == initStateU.ordering() &&
        stateU.ordering() == initStateM.ordering() && stateM.ordering() == initStateM.ordering();

    if (bEws1 && bSameOrdering) {
            
            auto func = PRAGMA_THREADS_FOR{
                 for (auto i = start; i < stop; i++) {
                      //m = B_1 * m + (1-B_1)*grad
                      stM[i] = beta1 * initM[i] + grad[i] * (1 - beta1);
                      //u = max(B_2 * u, |grad|)
                      stU[i] = sd::math::nd4j_max((beta2 * initU[i]), sd::math::nd4j_abs(grad[i])) + 1e-32;

                      up[i] = stM[i] * epsilonT / stU[i];
                 }
            };

           samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
           return;
    }
    
    bool bXZsame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), update.shapeInfo());
    bool bXInVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateU.shapeInfo());
    bool bXStVSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateU.shapeInfo());
    bool bXInMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), initStateM.shapeInfo());
    bool bXStMSame = shape::haveSameShapeAndStrides(gradient.shapeInfo(), stateM.shapeInfo());

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
        for (auto i = start; i < stop; i++) {
            shape::index2coordsCPU(start, i, gradient.shapeInfo(), coords);
            const auto xOffset =  shape::getOffset(gradient.shapeInfo(), coords);
            const auto zOffset = bXZsame ? xOffset : shape::getOffset(update.shapeInfo(), coords);
            const auto initUOffset = bXInVSame ? xOffset : shape::getOffset(initStateU.shapeInfo(), coords);
            const auto stUOffset = bXStVSame ? xOffset : shape::getOffset(stateU.shapeInfo(), coords);
            const auto initMOffset = bXInMSame ? xOffset : shape::getOffset(initStateM.shapeInfo(), coords);
            const auto stMOffset = bXStMSame ? xOffset : shape::getOffset(stateM.shapeInfo(), coords);
            
            //m = B_1 * m + (1-B_1)*grad
            stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * (1 - beta1);
            //u = max(B_2 * u, |grad|)
            stU[stUOffset] = sd::math::nd4j_max((beta2 * initU[initUOffset]), sd::math::nd4j_abs(grad[xOffset])) + 1e-32;

            up[zOffset] = stM[stMOffset] * epsilonT / stU[stUOffset];
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf(), 1);
    return;
}

void updaterAdaMax(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateU, const NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), adaMaxUpdater_, (gradient, initStateU, initStateM, update, stateU, stateM, dLr, dBeta1, dBeta2, dEpsilon, nIteration), FLOAT_TYPES);
}

}
}
}
