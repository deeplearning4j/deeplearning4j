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
#include <ops/declarable/helpers/ctc_decode.h>

#include <cmath>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_ctc_greedy_decoder)

namespace sd {
namespace ops {
namespace helpers {

template <typename T, typename I>
static void ctcGreedyDecode_(sd::LaunchContext* context,
                              NDArray* logits,
                              NDArray* sequenceLength,
                              NDArray* decoded,
                              NDArray* logProb,
                              sd::LongType blankIndex,
                              bool mergeRepeated) {

    const auto batchSize = logits->sizeAt(0);
    const auto maxTimeSteps = logits->sizeAt(1);
    const auto numClasses = logits->sizeAt(2);

    // Get raw buffers for efficient access
    const T* logitsBuffer = logits->bufferAsT<T>();
    const I* seqLenBuffer = sequenceLength != nullptr ? sequenceLength->bufferAsT<I>() : nullptr;
    sd::LongType* decodedBuffer = decoded->bufferAsT<sd::LongType>();
    T* logProbBuffer = logProb->bufferAsT<T>();

    // Get strides
    const auto logitsBatchStride = logits->strideAt(0);
    const auto logitsTimeStride = logits->strideAt(1);
    const auto logitsClassStride = logits->strideAt(2);

    const auto decodedBatchStride = decoded->strideAt(0);
    const auto decodedTimeStride = decoded->strideAt(1);

    // Process each batch element in parallel
    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        for (sd::LongType b = start; b < stop; b += increment) {
            // Get sequence length for this batch
            sd::LongType seqLen = maxTimeSteps;
            if (seqLenBuffer != nullptr) {
                seqLen = static_cast<sd::LongType>(seqLenBuffer[b]);
                if (seqLen > maxTimeSteps) seqLen = maxTimeSteps;
            }

            T totalLogProb = static_cast<T>(0);
            sd::LongType prevToken = -1;
            sd::LongType decodedIdx = 0;

            const T* batchLogits = logitsBuffer + b * logitsBatchStride;
            sd::LongType* batchDecoded = decodedBuffer + b * decodedBatchStride;

            // Initialize decoded output to -1 (invalid)
            for (sd::LongType t = 0; t < maxTimeSteps; t++) {
                batchDecoded[t * decodedTimeStride] = -1;
            }

            for (sd::LongType t = 0; t < seqLen; t++) {
                const T* timeLogits = batchLogits + t * logitsTimeStride;

                // Find argmax for this timestep
                sd::LongType maxIdx = 0;
                T maxVal = timeLogits[0];

                for (sd::LongType c = 1; c < numClasses; c++) {
                    T val = timeLogits[c * logitsClassStride];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = c;
                    }
                }

                // Accumulate log probability
                totalLogProb += maxVal;

                // Apply CTC decoding rules
                bool shouldOutput = false;

                if (maxIdx != blankIndex) {
                    if (mergeRepeated) {
                        // Only output if different from previous non-blank token
                        if (maxIdx != prevToken) {
                            shouldOutput = true;
                        }
                    } else {
                        shouldOutput = true;
                    }
                }

                if (shouldOutput) {
                    batchDecoded[decodedIdx * decodedTimeStride] = maxIdx;
                    decodedIdx++;
                    prevToken = maxIdx;
                } else if (maxIdx == blankIndex) {
                    // Reset previous token on blank (allows repeated tokens separated by blank)
                    prevToken = -1;
                }
            }

            logProbBuffer[b] = totalLogProb;
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1);
}

void ctcGreedyDecode(sd::LaunchContext* context,
                      NDArray* logits,
                      NDArray* sequenceLength,
                      NDArray* decoded,
                      NDArray* logProb,
                      int blankIndex,
                      bool mergeRepeated) {

    const auto numClasses = logits->sizeAt(2);
    sd::LongType actualBlankIndex = blankIndex < 0 ? numClasses - 1 : blankIndex;

    auto xType = logits->dataType();
    auto iType = sequenceLength != nullptr ? sequenceLength->dataType() : sd::DataType::INT32;

    // Build selector based on input types
    BUILD_DOUBLE_SELECTOR(xType, iType, ctcGreedyDecode_,
                          (context, logits, sequenceLength, decoded, logProb, actualBlankIndex, mergeRepeated),
                          SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
