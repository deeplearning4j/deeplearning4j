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

#ifndef LIBND4J_FUSED_QK_NORM_ROPE_H
#define LIBND4J_FUSED_QK_NORM_ROPE_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Fused QK RMSNorm + Rotary Position Embedding.
 *
 * Combines three operations into one kernel pass:
 *   1. RMSNorm(Q) per head
 *   2. RMSNorm(K) per head
 *   3. RoPE rotation on both Q and K
 *
 * This eliminates 2 intermediate global memory round-trips vs. separate kernels.
 *
 * For each (batch, seq, head):
 *   q_norm = rms_norm(Q[b,s,h,:], gamma_q)
 *   k_norm = rms_norm(K[b,s,h,:], gamma_k)
 *   For each dimension pair (2i, 2i+1):
 *     theta = position * freq_base^(-2i/headDim)
 *     Q_out[2i]   = q_norm[2i]*cos(theta) - q_norm[2i+1]*sin(theta)
 *     Q_out[2i+1] = q_norm[2i]*sin(theta) + q_norm[2i+1]*cos(theta)
 *     (same for K)
 *
 * @param context       launch context
 * @param query         [batch, seqLen, numQHeads, headDim] query tensor
 * @param key           [batch, seqLen, numKVHeads, headDim] key tensor
 * @param gammaQ        [headDim] RMSNorm scale for queries
 * @param gammaK        [headDim] RMSNorm scale for keys
 * @param cosCache      [maxSeqLen, headDim/2] precomputed cos values (optional, nullptr to compute)
 * @param sinCache      [maxSeqLen, headDim/2] precomputed sin values (optional, nullptr to compute)
 * @param queryOut      [batch, seqLen, numQHeads, headDim] output normalized+rotated queries
 * @param keyOut        [batch, seqLen, numKVHeads, headDim] output normalized+rotated keys
 * @param epsilon       RMSNorm epsilon (default 1e-6)
 * @param freqBase      RoPE frequency base (default 10000.0)
 * @param isNeox        true for GPT-NeoX style (split-half), false for GPT-J (interleaved)
 */
SD_LIB_HIDDEN void fusedQkNormRope(LaunchContext* context,
                                    NDArray* query,
                                    NDArray* key,
                                    NDArray* gammaQ,
                                    NDArray* gammaK,
                                    NDArray* cosCache,
                                    NDArray* sinCache,
                                    NDArray* queryOut,
                                    NDArray* keyOut,
                                    float epsilon,
                                    double freqBase,
                                    bool isNeox);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_QK_NORM_ROPE_H
