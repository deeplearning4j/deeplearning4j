/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 27.11.2018
//

#include <loops/special_kernels.h>

namespace nd4j {

////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void flattenKernel(
                    Nd4jPointer *extraPointers,
                    int dOffset,
                    char order,
                    void *vz, Nd4jLong *zShapeInfo,
                    void *vy, Nd4jLong *yShapeInfo) {

    auto z = reinterpret_cast<T*>(vz);
    auto y = reinterpret_cast<T*>(vy);
    
    __shared__ Nd4jLong lenY, yOrder, zEWS, yEWS;

    if (threadIdx.x == 0) {                
        
        yEWS = shape::elementWiseStride(yShapeInfo);
        zEWS = shape::elementWiseStride(zShapeInfo);
        lenY = shape::length(yShapeInfo);
    }
    __syncthreads();

    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;        
        
    if (zEWS >= 1 && yEWS >= 1 && yOrder == order) {
 
        for (int i = tid; i < lenY; i += gridDim.x * blockDim.x)
            z[i * zEWS + dOffset] = y[i * yEWS];
    } 
    else {
        
        for(auto i = tid; i < lenY; i += gridDim.x * blockDim.x)
            z[i * zEWS + dOffset] = y[shape::getIndexOrderOffset(i, yShapeInfo, lenY, order)];
    } 
}

////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ void flattenKernelGeneric(dim3& launchDims, cudaStream_t *stream, 
                            Nd4jPointer *extraPointers,
                            int dOffset,
                            char order,
                            void *vz, Nd4jLong *zShapeInfo,
                            void *vy, Nd4jLong *yShapeInfo) {

    flattenKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(extraPointers, dOffset, order, vz, zShapeInfo, vy, yShapeInfo);
    nd4j::DebugHelper::checkErrorCode(stream, "flattenGeneric(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT flattenKernelGeneric, (dim3& launchDims, cudaStream_t *stream, Nd4jPointer *extraPointers, int dOffset, char order, void *vz, Nd4jLong *zShapeInfo, void *vy, Nd4jLong *yShapeInfo), LIBND4J_TYPES);


}