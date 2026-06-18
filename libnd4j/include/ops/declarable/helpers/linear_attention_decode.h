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

//
// linear_attention_decode.h — single-token decode step for linear attention.
//
// Implements the recurrent form of linear attention:
//
//   state_new = exp(-decay[h]) * state_old + k ⊗ v   (rank-1 outer-product update)
//   output    = q @ state_new                          (matrix-vector product)
//
// State is always stored and computed as float32 regardless of the input dtype,
// matching the convention established by gated_delta_rule.cu / mamba2_ssm.cu.
//

#ifndef LIBND4J_HELPERS_LINEAR_ATTENTION_DECODE_H
#define LIBND4J_HELPERS_LINEAR_ATTENTION_DECODE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void linearAttentionDecode(
    LaunchContext* context,
    NDArray* query,       // [batch, 1, numHeads, headDimK]
    NDArray* key,         // [batch, 1, numHeads, headDimK]
    NDArray* value,       // [batch, 1, numHeads, headDimV]
    NDArray* decayRates,  // [numHeads] float32 — per-head log-decay magnitude
    NDArray* state,       // [batch, numHeads, headDimV, headDimK] float32, in-place update
    NDArray* output);     // [batch, 1, numHeads, headDimV]

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_LINEAR_ATTENTION_DECODE_H
