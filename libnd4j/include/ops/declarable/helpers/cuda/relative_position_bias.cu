/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/relative_position_bias.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_relative_position_bias)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__global__ static void relativePositionBiasLearnedKernel(
    const T* biasTable,
    const sd::LongType* relativePositionIndex,
    T* output,
    const sd::LongType numHeads,
    const sd::LongType windowSize,
    const sd::LongType windowArea,
    const bool hasIndex,
    const sd::LongType biasTableStride0,
    const sd::LongType biasTableStride1,
    const sd::LongType indexStride0,
    const sd::LongType indexStride1,
    const sd::LongType outHeadStride,
    const sd::LongType outRowStride,
    const sd::LongType outColStride) {

    const sd::LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const sd::LongType total = numHeads * windowArea * windowArea;

    if (idx >= total) return;

    const sd::LongType h = idx / (windowArea * windowArea);
    const sd::LongType i = (idx / windowArea) % windowArea;
    const sd::LongType j = idx % windowArea;

    sd::LongType relIdx;
    if (hasIndex && relativePositionIndex != nullptr) {
        relIdx = relativePositionIndex[i * indexStride0 + j * indexStride1];
    } else {
        // Compute on the fly
        const int maxRelDist = 2 * windowSize - 1;
        int iRow = i / windowSize;
        int iCol = i % windowSize;
        int jRow = j / windowSize;
        int jCol = j % windowSize;
        int rowDiff = iRow - jRow + windowSize - 1;
        int colDiff = iCol - jCol + windowSize - 1;
        relIdx = rowDiff * maxRelDist + colDiff;
    }

    T biasVal = biasTable[relIdx * biasTableStride0 + h * biasTableStride1];
    output[h * outHeadStride + i * outRowStride + j * outColStride] = biasVal;
}

template <typename T>
__global__ static void relativePositionBiasAlibiKernel(
    T* output,
    const T* slopes,
    const sd::LongType numHeads,
    const sd::LongType seqLen,
    const sd::LongType outHeadStride,
    const sd::LongType outRowStride,
    const sd::LongType outColStride) {

    const sd::LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const sd::LongType total = numHeads * seqLen * seqLen;

    if (idx >= total) return;

    const sd::LongType h = idx / (seqLen * seqLen);
    const sd::LongType i = (idx / seqLen) % seqLen;
    const sd::LongType j = idx % seqLen;

    T slope = slopes[h];
    T distance = static_cast<T>(j - i);

    output[h * outHeadStride + i * outRowStride + j * outColStride] = slope * distance;
}

template <typename T>
static void relativePositionBiasCuda_(sd::LaunchContext* context,
                                       NDArray* biasTable,
                                       NDArray* relativePositionIndex,
                                       NDArray* output,
                                       int numHeads,
                                       int windowSize,
                                       bool useAlibi) {

    PointersManager manager(context, "relativePositionBias");

    if (useAlibi) {
        // ALiBi mode
        sd::LongType seqLen = output->sizeAt(1);

        // Compute slopes on CPU and copy to GPU
        std::vector<T> slopesVec(numHeads);
        double ratio = std::pow(2.0, -8.0 / numHeads);
        double slope = 1.0;
        for (int h = 0; h < numHeads; h++) {
            slope *= ratio;
            slopesVec[h] = static_cast<T>(slope);
        }

        T* slopesDevice;
        cudaMalloc(&slopesDevice, numHeads * sizeof(T));
        cudaMemcpyAsync(slopesDevice, slopesVec.data(), numHeads * sizeof(T),
                        cudaMemcpyHostToDevice, *context->getCudaStream());

        const sd::LongType total = numHeads * seqLen * seqLen;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;

        T* outputBuffer = output->specialBufferasT<T>();
        const auto outHeadStride = output->strideAt(0);
        const auto outRowStride = output->strideAt(1);
        const auto outColStride = output->strideAt(2);

        relativePositionBiasAlibiKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
            outputBuffer, slopesDevice, numHeads, seqLen,
            outHeadStride, outRowStride, outColStride);

        cudaFree(slopesDevice);
    } else {
        // Learned bias mode
        const T* biasBuffer = biasTable->specialBufferasT<T>();
        T* outputBuffer = output->specialBufferasT<T>();

        const sd::LongType windowArea = windowSize * windowSize;
        const bool hasIndex = relativePositionIndex != nullptr && !relativePositionIndex->isEmpty();

        const sd::LongType* indexBuffer = hasIndex ?
            relativePositionIndex->specialBufferasT<sd::LongType>() : nullptr;

        const auto biasTableStride0 = biasTable->strideAt(0);
        const auto biasTableStride1 = biasTable->strideAt(1);

        sd::LongType indexStride0 = 0, indexStride1 = 0;
        if (hasIndex) {
            indexStride0 = relativePositionIndex->strideAt(0);
            indexStride1 = relativePositionIndex->strideAt(1);
        }

        const auto outHeadStride = output->strideAt(0);
        const auto outRowStride = output->strideAt(1);
        const auto outColStride = output->strideAt(2);

        const sd::LongType total = numHeads * windowArea * windowArea;
        const int blockSize = 256;
        const int numBlocks = (total + blockSize - 1) / blockSize;

        relativePositionBiasLearnedKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
            biasBuffer, indexBuffer, outputBuffer,
            numHeads, windowSize, windowArea, hasIndex,
            biasTableStride0, biasTableStride1,
            indexStride0, indexStride1,
            outHeadStride, outRowStride, outColStride);
    }

    manager.synchronize();
}

void relativePositionBias(sd::LaunchContext* context,
                           NDArray* biasTable,
                           NDArray* relativePositionIndex,
                           NDArray* output,
                           int numHeads,
                           int windowSize,
                           bool useAlibi) {

    BUILD_SINGLE_SELECTOR(output->dataType(), relativePositionBiasCuda_,
                          (context, biasTable, relativePositionIndex, output, numHeads, windowSize, useAlibi),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(void relativePositionBiasCuda_,
                      (sd::LaunchContext* context, NDArray* biasTable, NDArray* relativePositionIndex,
                       NDArray* output, int numHeads, int windowSize, bool useAlibi),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
