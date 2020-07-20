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
// @author GS <sgazeos@gmail.com>, created on 16.01.2019
//

#include <loops/special_kernels.h>

namespace sd {
    static Nd4jLong __device__ __noinline__ getIndexOffset_(Nd4jLong index, Nd4jLong const* shapeInfo) {
        return shape::getIndexOffset(index, shapeInfo);
    }

    static Nd4jLong __device__ __noinline__ subArrayOffset(Nd4jLong index, Nd4jLong const* shapeInfoA, Nd4jLong const* shapeInfoB) {
        return shape::subArrayOffset(index, shapeInfoA, shapeInfoB);
    }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  tileKernel:
//  input: (inputBuffer and inputShape) - NDArray buffer and shape to tile
//  output: (outputBuffer and outputShape) - NDArray to tile input
//  resultLength - length for output array
    template<typename T>
    static __global__ void
    tileKernel(void const *inputBuffer, Nd4jLong const* inputShape, void *outputBuffer, Nd4jLong const* outputShape,
               Nd4jLong resultLength) {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        Original code to transform in cuda-based
        auto tid = blockIdx.x * blockDim.x + threadIdx.x; // copy linear sequence of elements, so one-level threading
        int totalThreads = gridDim.x * blockDim.x;
        if (shape::order(outputShape) == 'c') {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<T *>(outputBuffer) + i) = *(reinterpret_cast<T const *>(inputBuffer) + yOffset);
            }
        } else {
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto xOffset = getIndexOffset_(i, outputShape);
                auto yOffset = subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<T *>(outputBuffer) + xOffset) = *(reinterpret_cast<T const *>(inputBuffer) + yOffset);
            }
        }

    }

    BUILD_SINGLE_TEMPLATE(template __global__ void tileKernel,(void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void tileKernelH(void const *inputBuffer, Nd4jLong const* inputShape, void *outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, cudaStream_t *stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernel<T> << < launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer, outputShape, resultLength);
    }

    BUILD_SINGLE_TEMPLATE(template void tileKernelH, (void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, cudaStream_t *stream), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// enhancement for tileKernel to different input and output data types: X - output type, Y - input type
    template<typename X, typename Y>
    static __global__ void
    tileKernelDouble(void const *inputBuffer, Nd4jLong const* inputShape, void *outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, Nd4jLong ews) {
        char ordering = shape::order(outputShape);
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;

        if (ordering == 'c' && ews == 1) {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + i) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        } else if (ordering == 'c' && ews > 1) {
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + i * ews) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        } else {

            for (int i = tid; i < resultLength; i += totalThreads) {

                auto xOffset = getIndexOffset_(i, outputShape);
                auto yOffset = subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + xOffset) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        }
    }

    BUILD_SINGLE_TEMPLATE_TWICE(template __global__ void tileKernelDouble, (void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, Nd4jLong ews), LIBND4J_TYPES);

    template<typename X, typename Y>
    void tileKernelHH(void const *inputBuffer, Nd4jLong const* inputShape, void *outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t *stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernelDouble<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer, outputShape, resultLength, ews);
    }

    BUILD_SINGLE_TEMPLATE_TWICE(template void tileKernelHH, (void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t *stream),LIBND4J_TYPES);
}