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

#ifndef LIBND4J_APPLY_TOKEN_BITMASK_H
#define LIBND4J_APPLY_TOKEN_BITMASK_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Apply a token bitmask to logits in-place for grammar/constrained decoding.
 *
 * For each batch element:
 *   For each token position in the vocabulary:
 *     If the corresponding bit in the bitmask is 0 (disallowed),
 *       set logits[batch][token] = -infinity
 *
 * The bitmask is packed as uint32: bit j of word i corresponds to token (i*32 + j).
 * A set bit (1) means the token is ALLOWED; a clear bit (0) means DISALLOWED.
 *
 * This enables structured output generation (JSON schema, regex, grammar FSM).
 *
 * @param context     launch context
 * @param logits      [batchSize, vocabSize] logits tensor (modified in-place)
 * @param bitmask     [batchSize, ceil(vocabSize/32)] packed uint32 bitmask
 * @param indices     [batchSize] optional per-batch index mapping (may be nullptr).
 *                    If provided, indices[i] maps batch element i to bitmask row indices[i].
 *                    If nullptr, identity mapping (bitmask row i -> batch i).
 */
SD_LIB_HIDDEN void applyTokenBitmaskInplace(LaunchContext* context,
                                             NDArray* logits,
                                             NDArray* bitmask,
                                             NDArray* indices);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_APPLY_TOKEN_BITMASK_H
