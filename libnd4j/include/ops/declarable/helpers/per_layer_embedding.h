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

#ifndef LIBND4J_HELPERS_PER_LAYER_EMBEDDING_H
#define LIBND4J_HELPERS_PER_LAYER_EMBEDDING_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Per-Layer Embedding (Gemma 4)
 *
 * output = hidden_states + ple_weight[token_ids] * scale
 *
 * @param context       launch context
 * @param hiddenStates  [batch, seq_len, hidden_dim]
 * @param pleWeight     [vocab_size, hidden_dim]
 * @param tokenIds      [batch, seq_len] (INT64 or INT32)
 * @param output        [batch, seq_len, hidden_dim]
 * @param scale         scalar multiplier for the embedding lookup
 */
SD_LIB_HIDDEN void perLayerEmbedding(LaunchContext* context,
                                      NDArray* hiddenStates,
                                      NDArray* pleWeight,
                                      NDArray* tokenIds,
                                      NDArray* output,
                                      double scale);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_PER_LAYER_EMBEDDING_H
