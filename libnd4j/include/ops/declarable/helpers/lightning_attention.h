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

#ifndef LIBND4J_HELPERS_LIGHTNING_ATTENTION_H
#define LIBND4J_HELPERS_LIGHTNING_ATTENTION_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Lightning Attention — O(N) linear attention using cumulative sum recurrence.
 *
 * Instead of computing the full N×N attention matrix, linear attention
 * reformulates the computation as:
 *
 *   For each position i:
 *     S_i = sum_{j<=i} (K_j^T * V_j)         (running outer-product state)
 *     O_i = Q_i * S_i                          (query applied to accumulated state)
 *
 * This reduces complexity from O(N²d) to O(Nd²), which is faster for
 * long sequences when d < N.
 *
 * The kernel feature map is applied externally (caller's responsibility).
 * Common choices: elu(x)+1, softmax kernel, random feature maps.
 *
 * @param context   Launch context
 * @param query     [batch, heads, seqLen, headDim] — query after feature map
 * @param key       [batch, heads, seqLen, headDim] — key after feature map
 * @param value     [batch, heads, seqLen, headDim] — value
 * @param output    [batch, heads, seqLen, headDim] — attention output
 * @param scale     attention scale factor (typically 1/sqrt(headDim))
 */
SD_LIB_HIDDEN void lightningAttentionCpu(LaunchContext* context,
                                          NDArray* query, NDArray* key, NDArray* value,
                                          NDArray* output, double scale);

#if defined(SD_CUDA)
SD_LIB_HIDDEN void lightningAttentionCuda(LaunchContext* context,
                                           NDArray* query, NDArray* key, NDArray* value,
                                           NDArray* output, double scale);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_LIGHTNING_ATTENTION_H
