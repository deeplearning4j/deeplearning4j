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
// Created by Yurii Shyrma on 07.12.2017.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j {
namespace ops {
namespace helpers {


    template <typename T>
    static __global__ void matrixSetDiagKernel(void* outputBuffer, Nd4jLong* outputShape, void const* diagonalBuffer, Nd4jLong* diagonalShape, Nd4jLong lastDimSize, Nd4jLong last2DimSize, Nd4jLong lastSmallDim, Nd4jLong batchSize) {
        __shared__ T* z;
        __shared__ T const* x;
        __shared__ Nd4jLong outLength, diagonalLen;
        if (threadIdx.x == 0) {
            z = reinterpret_cast<T*>(outputBuffer);
            x = reinterpret_cast<T const*>(diagonalBuffer);
            outLength = shape::length(outputShape);
            diagonalLen = shape::length(diagonalShape);
        }
        __syncthreads();

        for(int i = blockIdx.x; i < batchSize; i+= gridDim.x )
            for(int j = threadIdx.x; j < lastSmallDim; j += blockDim.x) {
//                z[i * last2DimSize + j * (lastDimSize + 1)] = x[i * lastSmallDim + j];
                z[shape::getIndexOffset(i * last2DimSize + j * (lastDimSize + 1), outputShape, outLength)] = x[shape::getIndexOffset(i * lastSmallDim + j, diagonalShape, diagonalLen)];
            }
    }
    //////////////////////////////////////////////////////////////////////////
    // Returns a batched matrix tensor with new batched diagonal values.
    // for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
    template <typename T>
    static void _matrixSetDiag(nd4j::LaunchContext * context, const NDArray* input, const NDArray* diagonal, NDArray* output) {
        *output = *input;

        const int lastDimSize = input->sizeAt(-1);
        const int last2DimSize = input->sizeAt(-1) * input->sizeAt(-2);
        const int lastSmallDim = diagonal->sizeAt(-1);
        const int batchSize = input->lengthOf()/last2DimSize;
        auto stream = context->getCudaStream();
        dim3 launchDims(256, 512, 8192);
        matrixSetDiagKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), diagonal->getSpecialBuffer(), diagonal->getSpecialShapeInfo(), lastDimSize, last2DimSize, lastSmallDim, batchSize);
//// #pragma omp parallel for if(batchSize > Environment::getInstance()->elementwiseThreshold()) schedule(static)
//        for(int i = 0; i < batchSize; ++i )
//            for(int j = 0; j < lastSmallDim; ++j) {
//                output->p(i*last2DimSize + j*(lastDimSize + 1), diagonal->e<T>(i*lastSmallDim + j));
//            }

    }

    void matrixSetDiag(nd4j::LaunchContext * context, const NDArray* input, const NDArray* diagonal, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _matrixSetDiag, (context, input, diagonal, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _matrixSetDiag, (nd4j::LaunchContext * context, const NDArray* input, const NDArray* diagonal, NDArray* output), LIBND4J_TYPES);

}
}
}