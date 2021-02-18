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
// Created by GS <sgazeos@gmail.com> on 4/6/2018.
//

#include <array/ResultSet.h>
#include <ops/declarable/helpers/diag.h>

namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diag functor cuda kernel
// outputBuffer - output tensor buffer
// outputShape - output tensor shape
// inputBuffer - input tensor buffer - this tensor should be placed on diagonal position of output
// inputShape - input tensor shape
// inputLength - length for input tensor
//
template <typename T>
static __global__ void diagFunctorKernel(void* outputBuffer, const Nd4jLong* outputShape, void const* inputBuffer, const Nd4jLong* inputShape, Nd4jLong inputLength) {
    __shared__ T *z;
    __shared__ T const* x;
    __shared__ Nd4jLong outputLength;

    if (threadIdx.x == 0) {
        z = reinterpret_cast<T*>(outputBuffer);
        x = reinterpret_cast<T const*>(inputBuffer);

        outputLength = shape::length(outputShape);
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;

    for (int t = tid; t < inputLength; t += step) { // for all vals in input, put all on diagonal position to output
        z[shape::getIndexOffset(t * (inputLength + 1), outputShape)] = x[shape::getIndexOffset(t, inputShape)]; //tX];
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diag part functor cuda kernel
// outputBuffer - output tensor buffer - linear sequence of diagonal values
// outputShape - output tensor shape
// inputBuffer - input tensor buffer - this tensor should be placed on diagonal position of output
// inputShape - input tensor shape
// outputLength - given length of output
// inputLength - given length for input tensor
//
    template <typename T>
    static __global__ void diagPartFunctorKernel(void* outputBuffer, const Nd4jLong* outputShape, void const* inputBuffer, const Nd4jLong* inputShape, Nd4jLong outputLength, Nd4jLong inputLength) {
        __shared__ T *z;
        __shared__ T const* x;

        if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
            x = reinterpret_cast<T const*>(inputBuffer);

        }
        __syncthreads();

        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        Nd4jLong i = threadIdx.x * (outputLength + 1); // pos to diagonal value
        for (int t = tid; t < outputLength && i < inputLength; t += step) { // loop by output, but input matrix may not be square
            // put diagonal val from input onto output
            z[shape::getIndexOffset(t, outputShape)] = x[shape::getIndexOffset(i, inputShape)]; 
            i += outputLength + 1; // shift to next diagonal value
        }
    }

//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
    template <typename T>
    static void _diagFunctor(sd::LaunchContext * context, const NDArray* input, NDArray* output) {
        auto stream = context->getCudaStream();
        auto inputLength = input->lengthOf();
        dim3 launchDims(256, 512, 8192);
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();
        diagFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), inputLength);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagFunctor - caller for diag functor processor
    void diagFunctor(sd::LaunchContext * context, const NDArray* input, NDArray* output) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, _diagFunctor, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _diagFunctor, (sd::LaunchContext * context, const NDArray* input, NDArray* output);, LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagPartFunctor - caller for diag part functor kernel
    template <typename T>
    void _diagPartFunctor(sd::LaunchContext * context, NDArray const* input, NDArray* output) {
        const int outLen = output->lengthOf();
        const int inLen = input->lengthOf();
        auto stream = context->getCudaStream();

        dim3 launchDims(256, 512, 8192);
        if (!input->isActualOnDeviceSide())
            input->syncToDevice();

        diagPartFunctorKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), outLen, inLen);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diagPartFunctor - caller for diag part functor processor
    void diagPartFunctor(sd::LaunchContext * context, NDArray const* input, NDArray* output) {
        auto zType = output->dataType();
        BUILD_SINGLE_SELECTOR(zType, _diagPartFunctor, (context, input, output), NUMERIC_TYPES);

    }

}
}
}