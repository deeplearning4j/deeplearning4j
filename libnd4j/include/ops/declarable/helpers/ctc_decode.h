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

#ifndef LIBND4J_HELPERS_CTC_DECODE_H
#define LIBND4J_HELPERS_CTC_DECODE_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief CTC Greedy Decoder - decodes CTC output using greedy/best path decoding
 *
 * Performs greedy decoding on CTC output by selecting the most likely
 * token at each timestep and optionally merging repeated tokens.
 *
 * @param context Launch context for execution
 * @param logits Input logits [batch, time, num_classes] - raw network outputs or log probabilities
 * @param sequenceLength Sequence lengths [batch] - actual sequence lengths (can be nullptr)
 * @param decoded Output decoded sequences [batch, time] - decoded token indices (-1 for invalid)
 * @param logProb Output log probabilities [batch] - log probability of decoded sequence
 * @param blankIndex Index of blank token (use num_classes-1 if < 0)
 * @param mergeRepeated Whether to merge consecutive identical non-blank tokens
 */
SD_LIB_HIDDEN void ctcGreedyDecode(sd::LaunchContext* context,
                                    NDArray* logits,
                                    NDArray* sequenceLength,
                                    NDArray* decoded,
                                    NDArray* logProb,
                                    int blankIndex,
                                    bool mergeRepeated);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_CTC_DECODE_H
