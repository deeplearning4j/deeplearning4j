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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/matrix_band.h>
#include <TAD.h>
#include <cuda_exception.h>
#include <ShapeUtils.h>
#include <helpers/ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// matrix band kernel
//
// inputBuffer - buffer of input tensor
// inputShape - shape of input tensor
// outputBuffer - buffer of output tensor
// outputShape - shape of output tensor
// lowerBand - lower band of matrix
// upperBand - upper band of matrix
// tadOnlyInputShapeInfo - TAD shape for input
// tadInputOffsets - TAD offsets for input
// tadOnlyOutputShapeInfo - TAD output shape
// tadOutputOffsets - TAD output offsets
// numTads - number of subarrays
// inputLength - input subarray length
//
    template <typename T>
    static __global__ void matrixBandKernel(void* inputBuffer, Nd4jLong* inputShape,
            void* outputBuffer, Nd4jLong* outputShape, Nd4jLong lowerBand, Nd4jLong upperBand, Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong* tadInputOffsets,
                                            Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong* tadOutputOffsets, Nd4jLong numTads, Nd4jLong inputLength) {
        int totalThreads = blockDim.x;
        Nd4jLong rows = shape::sizeAt(inputShape, -2);
        Nd4jLong cols = shape::sizeAt(inputShape, -1);
        for (Nd4jLong e = blockIdx.x; e < numTads; e += gridDim.x) {
            auto yOffset = tadInputOffsets[e];
            auto xOffset = tadOutputOffsets[e];
            for (Nd4jLong i = blockIdx.y; i < rows; i += gridDim.y) {
                for (Nd4jLong j = threadIdx.x; j < cols; j += totalThreads) {
                    Nd4jLong coords[2] = {i, j};
                    Nd4jLong tadOffsetOut = shape::getOffset(tadOnlyOutputShapeInfo, coords);
                    Nd4jLong tadOffsetIn = shape::getOffset(tadOnlyInputShapeInfo, coords);

                    if (i >= j) { // check lower diagonals
                        if (lowerBand > 0) {
                            if ((i - j) > lowerBand)
                                *(reinterpret_cast<T *>(outputBuffer) + xOffset + tadOffsetOut) = T(0);
                            else
                                *(reinterpret_cast<T *>(outputBuffer) + xOffset + tadOffsetOut) = *(
                                        reinterpret_cast<T const *>(inputBuffer) + yOffset + tadOffsetIn);
                        }
                    } else if (j > i) {
                        if (upperBand > 0)
                            if ((j - i) > upperBand)
                                *(reinterpret_cast<T *>(outputBuffer) + xOffset + tadOffsetOut) = T(0);
                            else
                                *(reinterpret_cast<T *>(outputBuffer) + xOffset + tadOffsetOut) = *(
                                        reinterpret_cast<T const *>(inputBuffer) + yOffset + tadOffsetIn);
                    }
                }
            }
        }

    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// matrixBandPart_ - main algorithm caller
//
    template <typename T>
    void matrixBandPart_(nd4j::LaunchContext * context, NDArray* input, NDArray* output, Nd4jLong lowerBand, Nd4jLong upperBand) {
        dim3 launchDims(256, 512, 8192);
        auto stream = context->getCudaStream();

        std::vector<int> lastDims({input->rankOf() - 2, input->rankOf() - 1});
        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), lastDims);

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), lastDims);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), lastDims);

        const Nd4jLong numTads = packX.numberOfTads();

        NDArray::prepareSpecialUse({output}, {input});
        matrixBandKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(input->getSpecialBuffer(),
                input->getSpecialShapeInfo(), output->getSpecialBuffer(), output->getSpecialShapeInfo(),
                lowerBand, upperBand, packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), numTads, input->lengthOf());
        NDArray::registerSpecialUse({output}, {input});
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void matrixBandPart(nd4j::LaunchContext * context, NDArray* input, NDArray* output, Nd4jLong lowerBand, Nd4jLong upperBand) {
        BUILD_SINGLE_SELECTOR(input->dataType(), matrixBandPart_, (context, input, output, lowerBand, upperBand), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void matrixBandPart_, (nd4j::LaunchContext * context, NDArray* input, NDArray* output, Nd4jLong lowerBand, Nd4jLong upperBand), FLOAT_TYPES);
}
}
}

