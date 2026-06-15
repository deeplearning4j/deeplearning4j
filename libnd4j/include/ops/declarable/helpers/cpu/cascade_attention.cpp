/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/cascade_attention.h>
#include <system/op_boilerplate.h>
#include <cmath>
#include <vector>
#include <limits>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void cascadeAttentionCpu_(LaunchContext* context,
                                  NDArray* query, NDArray* key, NDArray* value,
                                  NDArray* output, int chunkSize, double scale) {
    auto batch = query->sizeAt(0);
    auto heads = query->sizeAt(1);
    auto queryLen = query->sizeAt(2);
    auto headDim = query->sizeAt(3);
    auto kvLen = key->sizeAt(2);

    T scaleF = static_cast<T>(scale);

    for (LongType b = 0; b < batch; b++) {
        for (LongType h = 0; h < heads; h++) {
            for (LongType q = 0; q < queryLen; q++) {
                // Read Q vector
                std::vector<T> qVec(headDim);
                for (LongType d = 0; d < headDim; d++) {
                    qVec[d] = query->e<T>(b, h, q, d);
                }

                // Per-query accumulation across chunks
                // Using log-sum-exp merge for numerical stability
                std::vector<T> outAccum(headDim, static_cast<T>(0));
                T globalMax = -std::numeric_limits<T>::infinity();
                T globalSumExp = static_cast<T>(0);

                // Process KV cache in chunks
                for (LongType chunkStart = 0; chunkStart < kvLen; chunkStart += chunkSize) {
                    LongType chunkEnd = std::min(chunkStart + (LongType)chunkSize, kvLen);
                    LongType chunkLen = chunkEnd - chunkStart;

                    // Compute attention scores for this chunk: Q @ K_chunk^T * scale
                    std::vector<T> scores(chunkLen);
                    T chunkMax = -std::numeric_limits<T>::infinity();

                    for (LongType k2 = 0; k2 < chunkLen; k2++) {
                        T dot = static_cast<T>(0);
                        for (LongType d = 0; d < headDim; d++) {
                            dot += qVec[d] * key->e<T>(b, h, chunkStart + k2, d);
                        }
                        scores[k2] = dot * scaleF;
                        if (scores[k2] > chunkMax) chunkMax = scores[k2];
                    }

                    // Compute chunk softmax numerators and sum
                    T chunkSumExp = static_cast<T>(0);
                    for (LongType k2 = 0; k2 < chunkLen; k2++) {
                        scores[k2] = std::exp(scores[k2] - chunkMax);
                        chunkSumExp += scores[k2];
                    }

                    // Compute chunk output: softmax_weights @ V_chunk
                    std::vector<T> chunkOut(headDim, static_cast<T>(0));
                    for (LongType k2 = 0; k2 < chunkLen; k2++) {
                        T w = scores[k2]; // unnormalized softmax weight
                        for (LongType d = 0; d < headDim; d++) {
                            chunkOut[d] += w * value->e<T>(b, h, chunkStart + k2, d);
                        }
                    }

                    // Merge with global accumulator using log-sum-exp
                    if (chunkStart == 0) {
                        // First chunk — initialize
                        globalMax = chunkMax;
                        globalSumExp = chunkSumExp;
                        outAccum = chunkOut;
                    } else {
                        // Merge: rescale both accumulators to common max
                        T newMax = std::max(globalMax, chunkMax);
                        T globalRescale = std::exp(globalMax - newMax);
                        T chunkRescale = std::exp(chunkMax - newMax);

                        T newSumExp = globalSumExp * globalRescale + chunkSumExp * chunkRescale;

                        for (LongType d = 0; d < headDim; d++) {
                            outAccum[d] = outAccum[d] * globalRescale + chunkOut[d] * chunkRescale;
                        }

                        globalMax = newMax;
                        globalSumExp = newSumExp;
                    }
                }

                // Normalize by total sum of exponentials
                if (globalSumExp > static_cast<T>(0)) {
                    for (LongType d = 0; d < headDim; d++) {
                        output->p(b, h, q, d, outAccum[d] / globalSumExp);
                    }
                }
            }
        }
    }
}

void cascadeAttentionCpu(LaunchContext* context,
                          NDArray* query, NDArray* key, NDArray* value,
                          NDArray* output, int chunkSize, double scale) {
    BUILD_SINGLE_SELECTOR(query->dataType(), cascadeAttentionCpu_,
                          (context, query, key, value, output, chunkSize, scale),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void cascadeAttentionCpu_,
                      (LaunchContext*, NDArray*, NDArray*, NDArray*, NDArray*, int, double),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
