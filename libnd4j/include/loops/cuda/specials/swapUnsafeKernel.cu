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

namespace sd {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // kernel to swap two NDArrays vals as linear sequences
    // input - theSecondBuffer/Shape from input NDArray
    // output - theFirstBuffer/Shape from input NDArray
    template <typename T>
    static __global__ void swapUnsafeKernel(void* theFirstBuffer, Nd4jLong const* theFirstShape, void* theSecondBuffer, Nd4jLong const* theSecondShape) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;

        __shared__ Nd4jLong resultLength, xEws, yEws;
        __shared__ bool sameOffsets, sameOrders;
        __shared__ T* input;
        __shared__ T* output;

        if (0 == threadIdx.x) {
           resultLength = shape::length(theFirstShape);
           input = reinterpret_cast<T*>(theSecondBuffer);
           output = reinterpret_cast<T*>(theFirstBuffer);

           sameOffsets = shape::haveSameShapeAndStrides(theFirstShape, theSecondShape);
           sameOrders  = shape::order(theFirstShape) == shape::order(theSecondShape);

           xEws = shape::elementWiseStride(theFirstShape);
           yEws = shape::elementWiseStride(theSecondShape);
        }
        __syncthreads();

        for (int i = tid; i < resultLength; i += totalThreads) {
            if(sameOrders && xEws > 0 && yEws > 0) {
                sd::math::nd4j_swap(output[i*xEws], input[i*yEws]);
            }
            else if(sameOffsets) {
                const auto offset = shape::getIndexOffset(i, theFirstShape);
                sd::math::nd4j_swap(output[offset], input[offset]);
            }
            else{
                const auto xOffset = shape::getIndexOffset(i, theFirstShape);
                const auto yOffset = shape::getIndexOffset(i, theSecondShape);
                sd::math::nd4j_swap(output[xOffset], input[yOffset]);
            }
        }
    }

    BUILD_SINGLE_TEMPLATE(template __global__ void swapUnsafeKernel, (void* theFirstBuffer, Nd4jLong const* theFirstShape, void* theSecondBuffer, Nd4jLong const* theSecondShape), LIBND4J_TYPES);

    template <typename T>
    void templatedSwapUnsafe(void* theFirstBuffer, Nd4jLong const* theFirstShape, void* theSecondBuffer, Nd4jLong const* theSecondShape, cudaStream_t* theStream) {
        swapUnsafeKernel<T><<<256, 512, 8192, *theStream>>>(theFirstBuffer, theFirstShape, theSecondBuffer, theSecondShape);
    }
    BUILD_SINGLE_TEMPLATE(template void templatedSwapUnsafe, (void* theFirstBuffer, Nd4jLong const* theFirstShape, void* theSecondBuffer, Nd4jLong const* theSecondShape, cudaStream_t* theStream), LIBND4J_TYPES);

}