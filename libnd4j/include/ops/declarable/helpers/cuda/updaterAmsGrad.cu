/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#include <ops/declarable/helpers/updatersHelpers.h>
#include <helpers/PointersManager.h>
#include <math/platformmath.h>
#include <math/templatemath.h>

namespace sd    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ void amsGradUpdaterCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vinv, const Nd4jLong* invShapeInfo, 
                                  const void* vinm, const Nd4jLong* inmShapeInfo, const void* vinh, const Nd4jLong* inhShapeInfo,
                                  void* vz, const Nd4jLong* zShapeInfo,  void* vstV, const Nd4jLong* stvShapeInfo, void* vstM, 
                                  const Nd4jLong* stmShapeInfo, void* vstH, const Nd4jLong* sthShapeInfo,
                                  const T lr, const T beta1, const T beta2, const T epsilon, const T iteration) {

    const auto grad = reinterpret_cast<const T*>(vx);
    const auto initV = reinterpret_cast<const T*>(vinv);
    const auto initM = reinterpret_cast<const T*>(vinm);
    const auto initH = reinterpret_cast<const T*>(vinh);
    
    auto up = reinterpret_cast<T*>(vz);
    auto stV = reinterpret_cast<T*>(vstV);
    auto stM = reinterpret_cast<T*>(vstM);
    auto stH = reinterpret_cast<T*>(vstH);

    __shared__ Nd4jLong xLen;
    __shared__ T mbeta1, mbeta2, epsilonT;
    __shared__ bool bEWS, bOrdering, bXZsame, bXInUSame, bXStUSame, bXInMSame, bXStMSame, bXInHSame, bXStHSame;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        
        epsilonT = lr * sd::math::nd4j_sqrt<T, T>(1.0 - sd::math::nd4j_pow<T, T, T>(beta2, (iteration + 1))) / (1.0 - sd::math::nd4j_pow<T, T, T>(beta1, (iteration + 1)));

        if (sd::math::nd4j_isnan(epsilonT) || 0 == epsilonT || sd::math::nd4j_isinf(epsilonT))
            epsilonT = epsilon;

        mbeta1 = (1 - beta1);
        mbeta2 = (1 - beta2);

        bEWS = 1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
               1 == shape::elementWiseStride(stmShapeInfo) && 1 == shape::elementWiseStride(inmShapeInfo) &&
               1 == shape::elementWiseStride(stvShapeInfo) && 1 == shape::elementWiseStride(invShapeInfo) &&
               1 == shape::elementWiseStride(sthShapeInfo) && 1 == shape::elementWiseStride(inhShapeInfo);

        bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) && shape::order(zShapeInfo) == shape::order(stmShapeInfo) &&
                    shape::order(stmShapeInfo) == shape::order(inmShapeInfo) && shape::order(inmShapeInfo) == shape::order(stvShapeInfo) &&
                    shape::order(stvShapeInfo) == shape::order(invShapeInfo) && shape::order(invShapeInfo) == shape::order(sthShapeInfo) &&
                    shape::order(sthShapeInfo) == shape::order(inhShapeInfo);

        bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        bXInUSame = shape::haveSameShapeAndStrides(xShapeInfo, invShapeInfo);
        bXStUSame = shape::haveSameShapeAndStrides(xShapeInfo, stvShapeInfo);
        bXInMSame = shape::haveSameShapeAndStrides(xShapeInfo, inmShapeInfo);
        bXStMSame = shape::haveSameShapeAndStrides(xShapeInfo, stmShapeInfo);
        bXInHSame = shape::haveSameShapeAndStrides(xShapeInfo, inhShapeInfo);
        bXStHSame = shape::haveSameShapeAndStrides(xShapeInfo, sthShapeInfo);
    }
    __syncthreads();

    int coords[MAX_RANK];

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {

        auto xOffset = i, zOffset = i, initMOffset = i, initVOffset = i, initHOffset = i, stMOffset = i, stVOffset = i, stHOffset = i;

        if (!bEWS || !bOrdering){

            shape::index2coords(i, xShapeInfo, coords);
            xOffset  = shape::getOffset(xShapeInfo, coords);
            zOffset  = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
            initMOffset = bXInMSame ? xOffset : shape::getOffset(inmShapeInfo, coords);
            stMOffset = bXStMSame ? xOffset : shape::getOffset(stmShapeInfo, coords);
            initVOffset = bXInUSame ? xOffset : shape::getOffset(invShapeInfo, coords);
            stVOffset = bXStUSame ? xOffset : shape::getOffset(stvShapeInfo, coords);
            initHOffset = bXInHSame ? xOffset : shape::getOffset(inhShapeInfo, coords);
            stHOffset = bXStHSame ? xOffset : shape::getOffset(sthShapeInfo, coords);
        }

        stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * mbeta1;
        stV[stVOffset] = beta2 * initV[initVOffset] + grad[xOffset] * grad[xOffset] * mbeta2;
        stH[stHOffset] = sd::math::nd4j_max(initH[initHOffset], stV[stVOffset]);

        up[zOffset] = epsilonT * stM[stMOffset] / (sd::math::nd4j_sqrt<T, T>(stH[stHOffset]) + epsilon);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void amsGradUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx, const Nd4jLong* xShapeInfo,
    const void* vinv, const Nd4jLong* invShapeInfo, const void* vinm, const Nd4jLong* inmShapeInfo,
    const void* vinh, const Nd4jLong* inhShapeInfo,  void* vz, const Nd4jLong* zShapeInfo,
    void* vstV, const Nd4jLong* stvShapeInfo, void* vstM, const Nd4jLong* stmShapeInfo, 
    void* vstH, const Nd4jLong* sthShapeInfo, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
    
    const T lr = static_cast<T>(dLr);
    const T beta1 = static_cast<T>(dBeta1);
    const T beta2 = static_cast<T>(dBeta2);
    const T epsilon = static_cast<T>(dEpsilon);
    const T iteration = static_cast<T>(nIteration);

    amsGradUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, * stream>>>(vx, xShapeInfo, vinv, invShapeInfo, vinm, inmShapeInfo,
        vinh, inhShapeInfo, vz, zShapeInfo, vstV, stvShapeInfo, vstM, stmShapeInfo, vstH, sthShapeInfo, lr, beta1, beta2, epsilon, iteration);
}

///////////////////////////////////////////////////////////////////
void updaterAmsGrad(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateV, const NDArray& initStateM, const NDArray& initStateH, 
     NDArray& update, NDArray& stateV, NDArray& stateM, NDArray& stateH, const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {

    PointersManager manager(context, "amsGradUpdater");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({ &update, &stateV, &stateM, &stateH }, { &gradient, &initStateV, &initStateM, &initStateH });
    BUILD_SINGLE_SELECTOR(gradient.dataType(), amsGradUpdaterCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
        initStateV.specialBuffer(), initStateV.specialShapeInfo(), initStateM.specialBuffer(), initStateM.specialShapeInfo(),
        initStateH.specialBuffer(), initStateH.specialShapeInfo(), update.specialBuffer(), update.specialShapeInfo(),
        stateV.specialBuffer(), stateV.specialShapeInfo(), stateM.specialBuffer(), stateM.specialShapeInfo(),
        stateH.specialBuffer(), stateH.specialShapeInfo(), dLr, dBeta1, dBeta2, dEpsilon, nIteration), FLOAT_TYPES);
    NDArray::registerSpecialUse({ &update, &stateV, &stateM , &stateH }, { &gradient, &initStateV, &initStateM, &initStateH });

    manager.synchronize();
}


}
}
}
