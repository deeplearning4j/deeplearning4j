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

#ifndef LIBND4J_MERGE_ATTENTION_STATES_H
#define LIBND4J_MERGE_ATTENTION_STATES_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Merge two partial attention outputs using log-sum-exp (LSE) weighting.
 *
 * Used for chunked/disaggregated prefill and prefix caching:
 * when attention is computed over separate key/value chunks independently,
 * this kernel merges the partial outputs into a correct final result.
 *
 * Algorithm (from arxiv 2501.01005 section 2.2):
 *   For each position (batch, head, seq):
 *     m = max(lse_prefix, lse_suffix)
 *     w_prefix = exp(lse_prefix - m)
 *     w_suffix = exp(lse_suffix - m)
 *     output = (w_prefix * attn_prefix + w_suffix * attn_suffix) / (w_prefix + w_suffix)
 *     lse_output = m + log(w_prefix + w_suffix)
 *
 * @param context     launch context
 * @param attnPrefix  [batch, numHeads, seqLen, headDim] attention output from prefix chunk
 * @param lsePrefix   [batch, numHeads, seqLen] log-sum-exp from prefix
 * @param attnSuffix  [batch, numHeads, seqLen, headDim] attention output from suffix chunk
 * @param lseSuffix   [batch, numHeads, seqLen] log-sum-exp from suffix
 * @param output      [batch, numHeads, seqLen, headDim] merged attention output
 * @param lseOutput   [batch, numHeads, seqLen] merged log-sum-exp (may be nullptr)
 */
SD_LIB_HIDDEN void mergeAttentionStates(LaunchContext* context,
                                         NDArray* attnPrefix,
                                         NDArray* lsePrefix,
                                         NDArray* attnSuffix,
                                         NDArray* lseSuffix,
                                         NDArray* output,
                                         NDArray* lseOutput);

/**
 * Merge N partial attention outputs using log-sum-exp weighting.
 *
 * Generalizes the two-chunk merge to N chunks for multi-way disaggregation.
 *
 * @param context     launch context
 * @param attnChunks  vector of [batch, numHeads, seqLen, headDim] tensors
 * @param lseChunks   vector of [batch, numHeads, seqLen] LSE tensors
 * @param output      [batch, numHeads, seqLen, headDim] merged output
 * @param lseOutput   [batch, numHeads, seqLen] merged LSE (may be nullptr)
 */
SD_LIB_HIDDEN void mergeAttentionStatesMulti(LaunchContext* context,
                                              const std::vector<NDArray*>& attnChunks,
                                              const std::vector<NDArray*>& lseChunks,
                                              NDArray* output,
                                              NDArray* lseOutput);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MERGE_ATTENTION_STATES_H
