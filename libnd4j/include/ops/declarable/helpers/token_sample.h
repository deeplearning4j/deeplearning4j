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

#ifndef LIBND4J_HELPERS_TOKEN_SAMPLE_H
#define LIBND4J_HELPERS_TOKEN_SAMPLE_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void tokenSample(NDArray* logits, NDArray* output,
                                double temperature, int topK, double topP,
                                LongType seed, LaunchContext* context);

/**
 * Extended token sampling with penalty and min-p support.
 *
 * Pipeline: penalties → temperature → top-k → min-p → top-p → sample
 *
 * @param logits       [batch, vocabSize] — will be modified in-place by penalties
 * @param output       [batch] INT64 — sampled token IDs
 * @param inputIds     [batch, seqLen] INT64 — prior tokens for penalty computation (nullable)
 * @param temperature  Temperature for logit scaling (<=0 = greedy)
 * @param topK         Top-K filtering (<=0 = off)
 * @param topP         Top-P nucleus filtering (<=0 or >=1 = off)
 * @param minP         Min-P adaptive filtering (<=0 = off)
 * @param repPenalty   Repetition penalty (1.0 = off)
 * @param freqPenalty  Frequency penalty (0.0 = off)
 * @param presPenalty  Presence penalty (0.0 = off)
 * @param seed         RNG seed (0 = random)
 */
SD_LIB_HIDDEN void tokenSampleWithPenalties(NDArray* logits, NDArray* output,
                                             NDArray* inputIds,
                                             double temperature, int topK,
                                             double topP, double minP,
                                             double repPenalty, double freqPenalty,
                                             double presPenalty,
                                             LongType seed, LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
