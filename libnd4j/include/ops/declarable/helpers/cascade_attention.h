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

#ifndef LIBND4J_HELPERS_CASCADE_ATTENTION_H
#define LIBND4J_HELPERS_CASCADE_ATTENTION_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Cascade Attention — chunked attention for long-context KV cache.
 *
 * Splits the KV cache into fixed-size chunks. For each query position,
 * attention is computed separately per chunk, then results are merged
 * using the log-sum-exp trick for numerical stability:
 *
 *   For each chunk c:
 *     A_c = Q @ K_c^T * scale
 *     m_c = max(A_c)
 *     l_c = sum(exp(A_c - m_c))
 *     O_c = softmax(A_c) @ V_c
 *
 *   Merge:
 *     m = max(m_1, m_2, ..., m_C)
 *     O = sum_c( exp(m_c - m) * l_c * O_c ) / sum_c( exp(m_c - m) * l_c )
 *
 * This avoids materializing the full [queryLen x kvLen] attention matrix.
 *
 * @param context    Launch context
 * @param query      [batch, heads, queryLen, headDim]
 * @param key        [batch, heads, kvLen, headDim]
 * @param value      [batch, heads, kvLen, headDim]
 * @param output     [batch, heads, queryLen, headDim]
 * @param chunkSize  KV chunk size (default: 512)
 * @param scale      attention scale (typically 1/sqrt(headDim))
 */
SD_LIB_HIDDEN void cascadeAttention(LaunchContext* context,
                                     NDArray* query, NDArray* key, NDArray* value,
                                     NDArray* output, int chunkSize, double scale);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_CASCADE_ATTENTION_H
