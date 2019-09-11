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

namespace nd4j {
    static Nd4jLong __device__ __noinline__ _getIndexOffset(Nd4jLong index, Nd4jLong *shapeInfo) {
        return shape::getIndexOffset(index, shapeInfo);
    }

    static Nd4jLong __device__ __noinline__ _subArrayOffset(Nd4jLong index, Nd4jLong *shapeInfoA, Nd4jLong *shapeInfoB) {
        return shape::subArrayOffset(index, shapeInfoA, shapeInfoB);
    }

    static Nd4jLong __device__ __noinline__ _length(Nd4jLong *shapeInfo) {
        return shape::length(shapeInfo);
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    static __global__ void
    tileKernel(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape,
               Nd4jLong resultLength) {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        Original code to transform in cuda-based
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        //const auto resultLength = shape::length(outputShape);
        if (shape::order(outputShape) == 'c') {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = _subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<T *>(outputBuffer) + i) = *(reinterpret_cast<T const *>(inputBuffer) + yOffset);
            }
        } else {
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto xOffset = _getIndexOffset(i, outputShape);
                auto yOffset = _subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<T *>(outputBuffer) + xOffset) = *(reinterpret_cast<T const *>(inputBuffer) + yOffset);
            }
        }

    }

    BUILD_SINGLE_TEMPLATE(template __global__ void tileKernel,(void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength), LIBND4J_TYPES);

    template<typename T>
    void tileKernelH(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape, Nd4jLong resultLength, cudaStream_t *stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernel<T> << < launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer, outputShape, resultLength);
    }

    BUILD_SINGLE_TEMPLATE(template void tileKernelH, (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, cudaStream_t *stream), LIBND4J_TYPES);

    template<typename X, typename Y>
    static __global__ void
    tileKernelDouble(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape, Nd4jLong resultLength, Nd4jLong ews) {
        char ordering = shape::order(outputShape);
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;

        if (ordering == 'c' && ews == 1) {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = _subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + i) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        } else if (ordering == 'c' && ews > 1) {
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = _subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + i * ews) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        } else {

            for (int i = tid; i < resultLength; i += totalThreads) {

                auto xOffset = _getIndexOffset(i, outputShape);
                auto yOffset = _subArrayOffset(i, outputShape, inputShape);
                *(reinterpret_cast<X *>(outputBuffer) + xOffset) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        }
    }

    BUILD_SINGLE_TEMPLATE_TWICE(template __global__ void tileKernelDouble, (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, Nd4jLong ews), LIBND4J_TYPES);

    template<typename X, typename Y>
    void tileKernelHH(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t *stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernelDouble<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer, outputShape, resultLength, ews);
    }

    BUILD_SINGLE_TEMPLATE_TWICE(template void tileKernelHH, (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t *stream),LIBND4J_TYPES);
}