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

#ifndef LIBND4J_GATED_LINEAR_ATTN_H
#define LIBND4J_GATED_LINEAR_ATTN_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Gated Linear Attention (GLA). q, k, v are [B, T, H, S]; the optional gate is
 * [B, T, H, S] (null → no decay / pure linear attention); state is [B, H, S, S];
 * output is [B, T, H, S]. Per (b,h), with an S×S recurrent state, for each token:
 *   state[i,j] = g[i] * state[i,j] + k[i]*v[j]
 *   out[j] = scale * sum_i q[i] * state[i,j]
 */
SD_LIB_HIDDEN void gatedLinearAttn(LaunchContext* context, NDArray* q, NDArray* k, NDArray* v,
                                   NDArray* gate, NDArray* state, NDArray* output, double scale);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GATED_LINEAR_ATTN_H
