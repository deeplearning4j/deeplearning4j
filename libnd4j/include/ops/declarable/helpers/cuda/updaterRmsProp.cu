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
__global__ void rmsPropUpdaterCuda(const void *vx, const Nd4jLong *xShapeInfo, const void *vin, const Nd4jLong *inShapeInfo, 
                                   void *vz, const Nd4jLong *zShapeInfo, void* vst, const Nd4jLong* stShapeInfo,
                                   const T lr, const T rmsDecay, const T epsilon) {

    const auto x = reinterpret_cast<const T*>(vx);
    const auto init = reinterpret_cast<const T*>(vin);
    
          auto up = reinterpret_cast<T*>(vz);
          auto st = reinterpret_cast<T*>(vst);

    __shared__ Nd4jLong xLen;   
    __shared__ bool bEWS, bOrdering, bXZsame, bXInSame, bXStSame;

    if (threadIdx.x == 0) {

        xLen = shape::length(xShapeInfo);
        
        bEWS = 1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
               1 == shape::elementWiseStride(stShapeInfo) && 1 == shape::elementWiseStride(inShapeInfo);
        
        bOrdering = shape::order(zShapeInfo) == shape::order(xShapeInfo) && shape::order(xShapeInfo) == shape::order(stShapeInfo) &&
            shape::order(xShapeInfo) == shape::order(inShapeInfo);
        bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        bXInSame = shape::haveSameShapeAndStrides(xShapeInfo, inShapeInfo); 
        bXStSame = shape::haveSameShapeAndStrides(xShapeInfo, stShapeInfo);
    }
    __syncthreads();
    
    int coords[MAX_RANK];

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i +=  gridDim.x * blockDim.x) {
        
        auto xOffset = i, zOffset = i, initOffset = i, stOffset = i;

        if (!bEWS || !bOrdering) {

            shape::index2coords(i, xShapeInfo, coords);
            xOffset  = shape::getOffset(xShapeInfo, coords);
            zOffset  = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
            initOffset = bXInSame ? xOffset : shape::getOffset(inShapeInfo, coords);
            stOffset = bXStSame ? xOffset : shape::getOffset(stShapeInfo, coords);
        }

        st[stOffset] = init[initOffset] * rmsDecay + x[xOffset] * x[xOffset] * (1 - rmsDecay) ;
        up[zOffset] = (lr * x[xOffset]) / (  math::nd4j_sqrt<T, T>(st[stOffset]) + epsilon);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void rmsPropUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, 
                                        const void *vx, const Nd4jLong *xShapeInfo, const void *vin, const Nd4jLong *inShapeInfo, 
                                        void *vz, const Nd4jLong *zShapeInfo, void* vst, const Nd4jLong* stShapeInfo,
                                        const double dLr, const double dRmsDecay, const double dEpsilon) {
    
    const T lr = static_cast<T>(dLr);
    const T rmsDecay = static_cast<T>(dRmsDecay);
    const T epsilon = static_cast<T>(dEpsilon);

    rmsPropUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, vin, inShapeInfo, 
                             vz, zShapeInfo, vst, stShapeInfo, lr, rmsDecay, epsilon);
}

///////////////////////////////////////////////////////////////////
void updaterRmsProp(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState, NDArray& update, NDArray& stateG, 
                    const double dLr, const double dRmsDecay, const double dEpsilon) {

    PointersManager manager(context, "rmsPropUpdater");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({&update, &stateG}, {&gradient, &initState });

    BUILD_SINGLE_SELECTOR(gradient.dataType(), rmsPropUpdaterCudaLauncher, (blocksPerGrid, threadsPerBlock, 
                          context->getCudaStream(), gradient.specialBuffer(), gradient.specialShapeInfo(),
                          initState.specialBuffer(), initState.specialShapeInfo(),
                          update.specialBuffer(), update.specialShapeInfo(),
                          stateG.specialBuffer(), stateG.specialShapeInfo(),
                          dLr, dRmsDecay, dEpsilon ), FLOAT_TYPES);

    NDArray::registerSpecialUse({&update, &stateG}, {&gradient, &initState});

    manager.synchronize();
}

}
}
}
