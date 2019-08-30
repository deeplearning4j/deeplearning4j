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
// Created by GS <sgazeos@gmail.com> on 4/6/2018.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/diag.h>

namespace nd4j {
namespace ops {
namespace helpers {

template <typename T>
static __global__ void diagFunctorKernel(void* outputBuffer, Nd4jLong* outputShape, void const* inputBuffer, Nd4jLong* inputShape, Nd4jLong inputLength) {
    __shared__ T *z;
    __shared__ T const* x;
    __shared__ Nd4jLong outputLength;

    if (threadIdx.x == 0) {
        z = reinterpret_cast<T*>(outputBuffer);
        x = reinterpret_cast<T const*>(inputBuffer);

        outputLength = shape::length(outputShape);
    }
    __syncthreads();

    const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;
    for (int t = tid; t < inputLength; t += step) {
        z[shape::getIndexOffset(t * (inputLength + 1), outputShape, outputLength)] = x[shape::getIndexOffset(t, inputShape, inputLength)]; //tX];
    }

}

    template <typename T>
    static __global__ void diagPartFunctorKernel(void* outputBuffer, Nd4jLong* outputShape, void const* inputBuffer, Nd4jLong* inputShape, Nd4jLong outputLength, Nd4jLong inputLength) {
        __shared__ T *z;
        __shared__ T const* x;

        if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
            x = reinterpret_cast<T const*>(inputBuffer);

        }
        __syncthreads();

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        Nd4jLong i = threadIdx.x * (outputLength + 1);
        for (int t = tid; t < outputLength && i < inputLength; t += step) {
            z[shape::getIndexOffset(t, outputShape, outputLength)] = x[shape::getIndexOffset(i, inputShape, inputLength)]; //tX];
            i += outputLength + 1;
        }
    }

//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
    template <typename T>
    static void _diagFunctor(nd4j::LaunchContext * context, const NDArray* input, NDArray* output) {
        auto stream = context->getCudaStream();
        auto inputLength = input->lengthOf();
        dim3 launchDims(256, 512, 8192);
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();
        diagFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), inputLength);
    }

    void diagFunctor(nd4j::LaunchContext * context, const NDArray* input, NDArray* output) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, _diagFunctor, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _diagFunctor, (nd4j::LaunchContext * context, const NDArray* input, NDArray* output);, LIBND4J_TYPES);

    template <typename T>
    void _diagPartFunctor(nd4j::LaunchContext * context, NDArray const* input, NDArray* output) {
        const int outLen = output->lengthOf();
        const int inLen = input->lengthOf();
        auto stream = context->getCudaStream();

        dim3 launchDims(256, 512, 8192);
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();

        diagPartFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), outLen, inLen);
    }


    void diagPartFunctor(nd4j::LaunchContext * context, NDArray const* input, NDArray* output) {
        auto zType = output->dataType();
        BUILD_SINGLE_SELECTOR(zType, _diagPartFunctor, (context, input, output), NUMERIC_TYPES);

    }

}
}
}