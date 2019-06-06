/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// @author GS <sgazeos@gmail.com>, created on 17.01.2019
//

#include <loops/special_kernels.h>

namespace nd4j {


    template <typename T>
    static __global__ void repeatKernel(void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength,
                                        Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
                                        Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets) {
        //auto tid = blockIdx.x * blockDim.x; // + threadIdx.x;
//        int totalThreads = gridDim.x * blockDim.x;
        int totalThreads = blockDim.x;
        //const auto resultLength = shape::length(outputShape);
        for (Nd4jLong i = blockIdx.x; i < numTads; i += gridDim.x) {
            auto yOffset = tadInputOffsets[i];
            auto xOffset = tadOutputOffsets[i];
            for (Nd4jLong j = threadIdx.x; j < inputLength; j += totalThreads) {
                auto outputOffset = shape::getIndexOrderOffset(j, tadOnlyOutputShapeInfo, inputLength, shape::order(tadOnlyInputShapeInfo));
                auto inputOffset  = shape::getIndexOrderOffset(j, tadOnlyInputShapeInfo,  inputLength, shape::order(tadOnlyInputShapeInfo));
                *(reinterpret_cast<T*>(outputBuffer) + xOffset + outputOffset) = *(reinterpret_cast<T const*>(inputBuffer) + yOffset + inputOffset);
            }
        }
    }
    BUILD_SINGLE_TEMPLATE(template __global__ void repeatKernel, (void const* inputBuffer, void* outputBuffer,
            Nd4jLong numTads, Nd4jLong inputLength, Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets), LIBND4J_TYPES);

    template <typename X, typename Y>
    static __global__ void repeatKernelDouble(void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength,
                                              Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
                                              Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets) {
        //auto tid = blockIdx.x * blockDim.x; // + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        //const auto resultLength = shape::length(outputShape);
        for (Nd4jLong i = blockIdx.x; i < numTads; i += gridDim.x) {
            auto yOffset = tadInputOffsets[i];
            auto xOffset = tadOutputOffsets[i];
            for (Nd4jLong j = threadIdx.x; j < inputLength; j += totalThreads) {
                auto outputOffset = shape::getIndexOrderOffset(j, tadOnlyOutputShapeInfo, inputLength, shape::order(tadOnlyInputShapeInfo));
                auto inputOffset  = shape::getIndexOrderOffset(j, tadOnlyInputShapeInfo,  inputLength, shape::order(tadOnlyInputShapeInfo));
                *(reinterpret_cast<X*>(outputBuffer) + xOffset + outputOffset) = static_cast<X>(*(reinterpret_cast<Y const*>(inputBuffer) + yOffset + inputOffset));
            }
        }
    }
    BUILD_DOUBLE_TEMPLATE(template __global__ void repeatKernelDouble, (void const* inputBuffer, void* outputBuffer,
            Nd4jLong numTads, Nd4jLong inputLength, Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets), LIBND4J_TYPES, LIBND4J_TYPES);

    template <typename T>
    void repeatKernelH(void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength, Nd4jLong outputLength,
                              Nd4jLong *tadOnlyInputShapeInfo, Nd4jLong *tadInputOffsets,
                              Nd4jLong *tadOnlyOutputShapeInfo,Nd4jLong *tadOutputOffsets,
                              cudaStream_t stream) {
        dim3 launchDims(256, 512, 8192);
        repeatKernel<T><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(inputBuffer, outputBuffer, numTads, inputLength, tadOnlyInputShapeInfo, tadInputOffsets, tadOnlyOutputShapeInfo, tadOutputOffsets);
    }
    BUILD_SINGLE_TEMPLATE(template void repeatKernelH, (void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength, Nd4jLong outputLength,
            Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets,
            cudaStream_t stream), LIBND4J_TYPES);


    template <typename X, typename Y>
    void repeatKernelHH(void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength,
                               Nd4jLong *tadOnlyInputShapeInfo, Nd4jLong *tadInputOffsets,
                               Nd4jLong *tadOnlyOutputShapeInfo,Nd4jLong *tadOutputOffsets,
                               cudaStream_t stream) {
        dim3 launchDims(256, 512, 8192);
        repeatKernelDouble<X,Y><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(inputBuffer, outputBuffer, numTads, inputLength, tadOnlyInputShapeInfo, tadInputOffsets, tadOnlyOutputShapeInfo, tadOutputOffsets);
    }
    BUILD_DOUBLE_TEMPLATE(template void repeatKernelHH, (void const* inputBuffer, void* outputBuffer, Nd4jLong numTads, Nd4jLong inputLength,
            Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets,
            cudaStream_t stream), LIBND4J_TYPES, LIBND4J_TYPES);


}