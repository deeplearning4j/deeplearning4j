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

#ifndef LIBND4J_DUAL_ROPE_H
#define LIBND4J_DUAL_ROPE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Dual RoPE (Rotary Position Embedding) from Gemma 4.
 *
 * Applies standard RoPE for sliding-window (local) attention layers and
 * proportional RoPE with a different freq_base for global attention layers.
 *
 * For each dimension pair (2i, 2i+1):
 *   theta_i = freq_base ^ (-2i / head_dim) * freq_scale
 *   cos_theta = cos(position * theta_i)
 *   sin_theta = sin(position * theta_i)
 *   output[2i]   = input[2i] * cos_theta - input[2i+1] * sin_theta
 *   output[2i+1] = input[2i] * sin_theta + input[2i+1] * cos_theta
 *
 * @param context       launch context
 * @param input         input tensor [batch, seq_len, num_heads, head_dim]
 * @param output        output tensor [batch, seq_len, num_heads, head_dim]
 * @param attentionType 0 = local (sliding window), 1 = global
 * @param positionOffset offset added to each sequence position index
 * @param localFreqBase  freq_base for local attention (default 10000)
 * @param globalFreqBase freq_base for global attention (default 1000000)
 * @param localFreqScale  frequency scaling for local attention (default 1.0)
 * @param globalFreqScale frequency scaling for global attention (default 1.0)
 */
SD_LIB_HIDDEN void dualRoPE(LaunchContext* context, NDArray* input, NDArray* output,
                             int attentionType, int positionOffset,
                             double localFreqBase, double globalFreqBase,
                             double localFreqScale, double globalFreqScale);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_DUAL_ROPE_H
