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
// @author GS <sgazeos@gmail.com>, created on 25.01.2019
//

#include <loops/special_kernels.h>

namespace nd4j {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void swapUnsafeKernel(void* theFirstBuffer, Nd4jLong* theFirstShape, void* theSecondBuffer, Nd4jLong* theSecondShape) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        Nd4jLong resultLength = shape::length(theFirstShape);
        //const auto resultLength = shape::length(outputShape);
//        if (shape::order(outputShape) == 'c') {           //  ews == 1 always here
        for (int i = tid; i < resultLength; i += totalThreads) {
            auto xEws = shape::order(theFirstShape)  == 'c'? shape::elementWiseStride(theFirstShape) :1;
            auto yEws = shape::order(theSecondShape) == 'c'? shape::elementWiseStride(theSecondShape):1;
            //if (shape::order(theFirstShape) ==)
            auto xOffset = shape::getIndexOffset(i * xEws, theFirstShape, resultLength);
            auto yOffset = shape::getIndexOffset(i * yEws, theSecondShape, resultLength);
            T temp = *(reinterpret_cast<T*>(theFirstBuffer) + xOffset);
            *(reinterpret_cast<T*>(theFirstBuffer) + xOffset) = *(reinterpret_cast<T*>(theSecondBuffer) + yOffset);
            *(reinterpret_cast<T*>(theSecondBuffer) + yOffset) = temp;
        }
    }

    BUILD_SINGLE_TEMPLATE(template __global__ void swapUnsafeKernel, (void* theFirstBuffer, Nd4jLong* theFirstShape, void* theSecondBuffer, Nd4jLong* theSecondShape), LIBND4J_TYPES);

    template <typename T>
    void templatedSwapUnsafe(void* theFirstBuffer, Nd4jLong* theFirstShape, void* theSecondBuffer, Nd4jLong* theSecondShape, cudaStream_t* theStream) {
        swapUnsafeKernel<T><<<256, 512, 8192, *theStream>>>(theFirstBuffer, theFirstShape, theSecondBuffer, theSecondShape);
    }
    BUILD_SINGLE_TEMPLATE(template void templatedSwapUnsafe, (void* theFirstBuffer, Nd4jLong* theFirstShape, void* theSecondBuffer, Nd4jLong* theSecondShape, cudaStream_t* theStream), LIBND4J_TYPES);

}