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
// Windowed/Local Attention for efficient transformers (Longformer, BigBird, Swin)
//

#ifndef LIBND4J_HELPERS_WINDOWED_ATTENTION_H
#define LIBND4J_HELPERS_WINDOWED_ATTENTION_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief Windowed/Local Attention forward pass
 *
 * Implements windowed attention mechanisms used in efficient transformers
 * like Longformer, BigBird, Swin Transformer, and SAM.
 *
 * @param context Launch context for execution
 * @param query Query tensor [batch, seq_len, num_heads, head_dim]
 * @param key Key tensor [batch, seq_len, num_heads, head_dim]
 * @param value Value tensor [batch, seq_len, num_heads, head_dim]
 * @param relativePositionBias Optional relative position bias [num_heads, window_size, window_size]
 * @param attentionMask Optional attention mask
 * @param output Output tensor [batch, seq_len, num_heads, head_dim]
 * @param attentionWeights Optional output attention weights (if returnWeights is true)
 * @param windowSize Size of attention window
 * @param numHeads Number of attention heads
 * @param shiftSize Shift size for shifted window attention (0 for no shift)
 * @param scale Attention scale factor (0 for auto = 1/sqrt(head_dim))
 * @param returnWeights Whether to return attention weights
 */
SD_LIB_HIDDEN void windowedAttention(sd::LaunchContext* context,
                                      NDArray* query,
                                      NDArray* key,
                                      NDArray* value,
                                      NDArray* relativePositionBias,
                                      NDArray* attentionMask,
                                      NDArray* output,
                                      NDArray* attentionWeights,
                                      int windowSize,
                                      int numHeads,
                                      int shiftSize,
                                      double scale,
                                      bool returnWeights);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_WINDOWED_ATTENTION_H
