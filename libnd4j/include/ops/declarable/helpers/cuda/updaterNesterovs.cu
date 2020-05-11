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
__global__ void nesterovsUpdaterCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vin, const Nd4jLong* inShapeInfo, 
                                     void* vz, const Nd4jLong* zShapeInfo, void* vst, const Nd4jLong* stShapeInfo, const T lr, const T momentum) {

    const auto grad = reinterpret_cast<const T*>(vx);
    const auto init = reinterpret_cast<const T*>(vin);
    auto up = reinterpret_cast<T*>(vz);
    auto st = reinterpret_cast<T*>(vst);

    __shared__ Nd4jLong xLen;
    __shared__ T momentumT;
    __shared__ bool bEWS, bOrdering, bXZsame, bXInSame, bXStSame;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        momentumT = (-momentum - 1);

        bEWS =  1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
                1 == shape::elementWiseStride(stShapeInfo) && 1 == shape::elementWiseStride(inShapeInfo);
        bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) && shape::order(xShapeInfo) == shape::order(inShapeInfo) &&
                    shape::order(xShapeInfo) == shape::order(stShapeInfo);

        bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        bXInSame = shape::haveSameShapeAndStrides(xShapeInfo, inShapeInfo);
        bXStSame = shape::haveSameShapeAndStrides(xShapeInfo, stShapeInfo);
    }
    __syncthreads();

    int coords[MAX_RANK];
    
    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {

        auto xOffset = i, zOffset = i, initOffset = i, stOffset = i;

        if (!bEWS || !bOrdering) {

            shape::index2coords(i, xShapeInfo, coords);
            xOffset  = shape::getOffset(xShapeInfo, coords);
            zOffset  = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
            initOffset = bXInSame ? xOffset : shape::getOffset(inShapeInfo, coords);
            stOffset = bXStSame ? xOffset : shape::getOffset(stShapeInfo, coords);
        }

        T prevState =  momentum * init[initOffset];
        st[stOffset] = prevState - lr * grad[xOffset];
        up[zOffset] = prevState + momentumT * st[stOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void nesterovsUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, 
                                          const void* vx, const Nd4jLong* xShapeInfo, const void* vin, const Nd4jLong* inShapeInfo, 
                                          void* vz, const Nd4jLong* zShapeInfo, void* vst, const Nd4jLong* stShapeInfo,
                                          const double dLr, const double dMomentum) {
    
     const T lr = static_cast<T>(dLr);
     const T momentum = static_cast<T>(dMomentum);
     nesterovsUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, * stream>>>(vx, xShapeInfo, vin, inShapeInfo,
                                             vz, zShapeInfo, vst, stShapeInfo, lr, momentum);
}

///////////////////////////////////////////////////////////////////
void updaterNesterovs(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState, 
                      NDArray& update, NDArray& stateV, const double dLr, const double dMomentum) {

    PointersManager manager(context, "nesterovsUpdater");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({ &update, &stateV }, { &gradient, &initState });
    BUILD_SINGLE_SELECTOR(gradient.dataType(), nesterovsUpdaterCudaLauncher, (blocksPerGrid, threadsPerBlock, 
        context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
        initState.specialBuffer(), initState.specialShapeInfo(),
        update.specialBuffer(), update.specialShapeInfo(),
        stateV.specialBuffer(), stateV.specialShapeInfo(), dLr, dMomentum), FLOAT_TYPES);
    NDArray::registerSpecialUse({ &update, &stateV }, { &gradient, &initState });

    manager.synchronize();
}

}
}
}
