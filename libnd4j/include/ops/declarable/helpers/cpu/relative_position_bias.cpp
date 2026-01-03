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

#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/relative_position_bias.h>

#include <cmath>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_relative_position_bias)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
std::vector<T> computeAlibiSlopes(int numHeads) {
    std::vector<T> slopes(numHeads);

    // Compute slopes as powers of 2
    // For n heads: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    double ratio = std::pow(2.0, -8.0 / numHeads);
    double slope = 1.0;
    for (int h = 0; h < numHeads; h++) {
        slope *= ratio;
        slopes[h] = static_cast<T>(slope);
    }
    return slopes;
}

template <typename T>
static void relativePositionBiasLearned_(sd::LaunchContext* context,
                                          NDArray* biasTable,
                                          NDArray* relativePositionIndex,
                                          NDArray* output,
                                          int numHeads,
                                          int windowSize) {

    const T* biasBuffer = biasTable->bufferAsT<T>();
    T* outputBuffer = output->bufferAsT<T>();

    const auto numRelPos = biasTable->sizeAt(0);
    const auto windowArea = windowSize * windowSize;

    // Get strides
    const auto biasTableStride0 = biasTable->strideAt(0);
    const auto biasTableStride1 = biasTable->strideAt(1);

    const auto outHeadStride = output->strideAt(0);
    const auto outRowStride = output->strideAt(1);
    const auto outColStride = output->strideAt(2);

    if (relativePositionIndex != nullptr && !relativePositionIndex->isEmpty()) {
        // Use precomputed relative position index
        const auto* indexBuffer = relativePositionIndex->bufferAsT<sd::LongType>();
        const auto indexStride0 = relativePositionIndex->strideAt(0);
        const auto indexStride1 = relativePositionIndex->strideAt(1);

        auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
            for (sd::LongType h = start; h < stop; h += increment) {
                T* outHead = outputBuffer + h * outHeadStride;

                for (sd::LongType i = 0; i < windowArea; i++) {
                    for (sd::LongType j = 0; j < windowArea; j++) {
                        sd::LongType relIdx = indexBuffer[i * indexStride0 + j * indexStride1];
                        T biasVal = biasBuffer[relIdx * biasTableStride0 + h * biasTableStride1];
                        outHead[i * outRowStride + j * outColStride] = biasVal;
                    }
                }
            }
        };

        samediff::Threads::parallel_for(func, 0, numHeads, 1);
    } else {
        // Compute relative position index on-the-fly
        // For 2D window: relative position = (row_diff + windowSize - 1) * (2*windowSize - 1) + (col_diff + windowSize - 1)
        const int maxRelDist = 2 * windowSize - 1;

        auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
            for (sd::LongType h = start; h < stop; h += increment) {
                T* outHead = outputBuffer + h * outHeadStride;

                for (sd::LongType i = 0; i < windowArea; i++) {
                    int iRow = i / windowSize;
                    int iCol = i % windowSize;

                    for (sd::LongType j = 0; j < windowArea; j++) {
                        int jRow = j / windowSize;
                        int jCol = j % windowSize;

                        int rowDiff = iRow - jRow + windowSize - 1;
                        int colDiff = iCol - jCol + windowSize - 1;
                        sd::LongType relIdx = rowDiff * maxRelDist + colDiff;

                        T biasVal = biasBuffer[relIdx * biasTableStride0 + h * biasTableStride1];
                        outHead[i * outRowStride + j * outColStride] = biasVal;
                    }
                }
            }
        };

        samediff::Threads::parallel_for(func, 0, numHeads, 1);
    }
}

template <typename T>
static void relativePositionBiasAlibi_(sd::LaunchContext* context,
                                        NDArray* sequenceLengthTensor,
                                        NDArray* output,
                                        int numHeads) {

    // Infer sequence length from output shape or input
    sd::LongType seqLen = output->sizeAt(1);

    T* outputBuffer = output->bufferAsT<T>();

    const auto outHeadStride = output->strideAt(0);
    const auto outRowStride = output->strideAt(1);
    const auto outColStride = output->strideAt(2);

    // Compute ALiBi slopes
    std::vector<T> slopes = computeAlibiSlopes<T>(numHeads);

    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        for (sd::LongType h = start; h < stop; h += increment) {
            T slope = slopes[h];
            T* outHead = outputBuffer + h * outHeadStride;

            for (sd::LongType i = 0; i < seqLen; i++) {
                for (sd::LongType j = 0; j < seqLen; j++) {
                    // ALiBi: slope * (j - i) for causal, slope * |j - i| for bidirectional
                    // Using causal formulation: only j <= i contributes
                    T distance = static_cast<T>(j - i);
                    outHead[i * outRowStride + j * outColStride] = slope * distance;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, numHeads, 1);
}

template <typename T>
static void relativePositionBias_(sd::LaunchContext* context,
                                   NDArray* biasTable,
                                   NDArray* relativePositionIndex,
                                   NDArray* output,
                                   int numHeads,
                                   int windowSize,
                                   bool useAlibi) {

    if (useAlibi) {
        relativePositionBiasAlibi_<T>(context, biasTable, output, numHeads);
    } else {
        relativePositionBiasLearned_<T>(context, biasTable, relativePositionIndex, output, numHeads, windowSize);
    }
}

void relativePositionBias(sd::LaunchContext* context,
                           NDArray* biasTable,
                           NDArray* relativePositionIndex,
                           NDArray* output,
                           int numHeads,
                           int windowSize,
                           bool useAlibi) {

    BUILD_SINGLE_SELECTOR(output->dataType(), relativePositionBias_,
                          (context, biasTable, relativePositionIndex, output, numHeads, windowSize, useAlibi),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(void relativePositionBias_,
                      (sd::LaunchContext* context, NDArray* biasTable, NDArray* relativePositionIndex,
                       NDArray* output, int numHeads, int windowSize, bool useAlibi),
                      SD_FLOAT_TYPES);

// Explicit instantiations for computeAlibiSlopes
template std::vector<float> computeAlibiSlopes<float>(int numHeads);
template std::vector<double> computeAlibiSlopes<double>(int numHeads);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
