/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
__global__ void adaDeltaUpdaterCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vinMsg, const Nd4jLong* inMsgShapeInfo, 
    const void* vinMsdx, const Nd4jLong* inMsdxShapeInfo, void* vz, const Nd4jLong* zShapeInfo, void* vstMsg, 
    const Nd4jLong* stMsgShapeInfo, void* vstMsdx, const Nd4jLong* stMsdxShapeInfo, const T rho, const T epsilon) {

    const auto grad = reinterpret_cast<const T*>(vx);
    const auto initMsg= reinterpret_cast<const T*>(vinMsg);
    const auto initMsdx = reinterpret_cast<const T*>(vinMsdx);
   
    auto up = reinterpret_cast<T*>(vz);
    auto stMsg = reinterpret_cast<T*>(vstMsg);
    auto stMsdx = reinterpret_cast<T*>(vstMsdx);

    __shared__ Nd4jLong xLen;
    __shared__ T rhoT;
    __shared__ bool bEWS, bOrdering, bXZsame, bXInMsgSame, bXStMsgSame, bXInMsdxSame, bXStMsdxSame;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        
        rhoT = (1 - rho);

        bEWS =  1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
                1 == shape::elementWiseStride(stMsgShapeInfo) && 1 == shape::elementWiseStride(inMsgShapeInfo) &&
                1 == shape::elementWiseStride(stMsdxShapeInfo) && 1 == shape::elementWiseStride(inMsdxShapeInfo);
        bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) && shape::order(zShapeInfo) == shape::order(stMsgShapeInfo) && 
                    shape::order(stMsgShapeInfo) == shape::order(inMsgShapeInfo) && shape::order(inMsgShapeInfo) == shape::order(stMsdxShapeInfo) &&
                    shape::order(stMsdxShapeInfo) == shape::order(inMsdxShapeInfo);

        bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        bXInMsgSame = shape::haveSameShapeAndStrides(xShapeInfo, inMsgShapeInfo);
        bXStMsgSame = shape::haveSameShapeAndStrides(xShapeInfo, stMsgShapeInfo);
        bXInMsdxSame = shape::haveSameShapeAndStrides(xShapeInfo, inMsdxShapeInfo);
        bXStMsdxSame = shape::haveSameShapeAndStrides(xShapeInfo, stMsdxShapeInfo);
    }
    __syncthreads();

    int coords[MAX_RANK];

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {

        auto xOffset = i, zOffset = i, initMsgOffset = i, initMsdxOffset = i, stMsgOffset = i, stMsdxOffset = i;

        if (!bEWS || !bOrdering){

            shape::index2coords(i, xShapeInfo, coords);
            xOffset  = shape::getOffset(xShapeInfo, coords);
            zOffset  = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
            initMsgOffset = bXInMsgSame ? xOffset : shape::getOffset(inMsgShapeInfo, coords);
            stMsgOffset = bXStMsgSame ? xOffset : shape::getOffset(stMsgShapeInfo, coords);
            initMsdxOffset = bXInMsdxSame ? xOffset : shape::getOffset(inMsdxShapeInfo, coords);
            stMsdxOffset = bXStMsdxSame ? xOffset : shape::getOffset(stMsdxShapeInfo, coords);
        }

        stMsg[stMsgOffset] = rho * initMsg[initMsgOffset] + grad[xOffset] * grad[xOffset] * rhoT;

        up[zOffset] = grad[xOffset] * (sd::math::nd4j_sqrt<T, T>(initMsdx[initMsdxOffset] + epsilon) / sd::math::nd4j_sqrt<T, T>(stMsg[stMsgOffset] + epsilon));

        stMsdx[stMsdxOffset] = rho * initMsdx[initMsdxOffset] + up[zOffset] * up[zOffset] * rhoT;
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void adaDeltaUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx, const Nd4jLong* xShapeInfo,
    const void* vinMsg, const Nd4jLong* inMsgShapeInfo, const void* vinMsdx, const Nd4jLong* inMsdxShapeInfo,
    void* vz, const Nd4jLong* zShapeInfo, void* vstMsg, const Nd4jLong* stMsgShapeInfo, 
    void* vstMsdx, const Nd4jLong* stMsdxShapeInfo, const double dRho, const double dEpsilon) {

    const T rho = static_cast<T>(dRho);
    const T epsilon = static_cast<T>(dEpsilon);

    adaDeltaUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, * stream>>>(vx, xShapeInfo, vinMsg, inMsgShapeInfo,
        vinMsdx, inMsdxShapeInfo, vz, zShapeInfo, vstMsg, stMsgShapeInfo, vstMsdx, stMsdxShapeInfo, rho, epsilon);
}

///////////////////////////////////////////////////////////////////
void updaterAdaDelta(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateMsg, const NDArray& initStateMsdx, 
                    NDArray& update, NDArray& stateMsg, NDArray& stateMsdx, const double dRho, const double dEpsilon) {

    PointersManager manager(context, "adaDeltaUpdater");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({ &update, &stateMsg, &stateMsdx }, { &gradient, &initStateMsg, &initStateMsdx });
    BUILD_SINGLE_SELECTOR(gradient.dataType(), adaDeltaUpdaterCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
        initStateMsg.specialBuffer(), initStateMsg.specialShapeInfo(), initStateMsdx.specialBuffer(), initStateMsdx.specialShapeInfo(),
        update.specialBuffer(), update.specialShapeInfo(),stateMsg.specialBuffer(), stateMsg.specialShapeInfo(),
        stateMsdx.specialBuffer(), stateMsdx.specialShapeInfo(), dRho, dEpsilon), FLOAT_TYPES);
    NDArray::registerSpecialUse({ &update, &stateMsg, &stateMsdx }, { &gradient, &initStateMsg, &initStateMsdx });

    manager.synchronize();
}

}
}
}
