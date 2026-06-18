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

#ifndef LIBND4J_HELPERS_SHARED_KV_ATTENTION_H
#define LIBND4J_HELPERS_SHARED_KV_ATTENTION_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Shared KV Attention (Gemma 4 pattern).
 *
 * Grouped-query attention where K/V come from a donor layer rather than
 * being projected from the current hidden state.  Supports causal masking
 * and optional sliding window.
 *
 * @param context     Launch context (CPU or CUDA stream)
 * @param query       [batch, numHeads, qSeqLen, headDim]
 * @param sharedKey   [batch, numKvHeads, kvSeqLen, headDim]  — from donor layer
 * @param sharedValue [batch, numKvHeads, kvSeqLen, headDim]  — from donor layer
 * @param mask        Optional [batch, 1, qSeqLen, kvSeqLen] broadcastable mask
 *                    (additive; -inf for masked positions). May be nullptr.
 * @param output      [batch, numHeads, qSeqLen, headDim]
 * @param numHeads    Total number of query heads
 * @param numKvHeads  Number of KV heads (numHeads must be divisible by numKvHeads)
 * @param causal      If true, positions q < k are masked to -inf
 * @param slidingWindowSize  If > 0, positions where (q - k) > slidingWindowSize
 *                           are masked to -inf.  0 means no sliding window.
 * @param scale       Scaling factor applied to Q@K^T.  Typically 1/sqrt(headDim).
 */
SD_LIB_HIDDEN void sharedKvAttention(LaunchContext* context,
                                      NDArray* query,
                                      NDArray* sharedKey,
                                      NDArray* sharedValue,
                                      NDArray* mask,
                                      NDArray* output,
                                      int numHeads,
                                      int numKvHeads,
                                      bool causal,
                                      int slidingWindowSize,
                                      double scale);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
