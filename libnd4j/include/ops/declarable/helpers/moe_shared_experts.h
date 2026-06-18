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
// Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern)
//

#ifndef LIBND4J_HELPERS_MOE_SHARED_EXPERTS_H
#define LIBND4J_HELPERS_MOE_SHARED_EXPERTS_H

#include <array/NDArray.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief MoE with shared experts forward pass
 *
 * Combines an always-on shared expert (SwiGLU) with top-K routed experts.
 * output = shared_expert(input) + weighted_sum(routed_experts(input))
 *
 * @param context Launch context for execution
 * @param input Input tensor [batch, seq_len, hidden_size]
 * @param routerWeights Router/gating weights [hidden_size, num_routed_experts]
 * @param routedExpertWeights Routed expert weights [num_routed_experts, hidden_size, expert_hidden]
 * @param sharedGateProj Shared expert gate projection [hidden_size, shared_intermediate_size]
 * @param sharedUpProj Shared expert up projection [hidden_size, shared_intermediate_size]
 * @param sharedDownProj Shared expert down projection [shared_intermediate_size, hidden_size]
 * @param routedExpertBias Routed expert biases (optional, may be nullptr)
 * @param output Output tensor [batch, seq_len, hidden_size]
 * @param routerProbs Output router probabilities [batch, seq_len, num_routed_experts]
 * @param expertIndices Output selected expert indices [batch, seq_len, top_k]
 * @param numRoutedExperts Number of routed experts
 * @param topK Number of experts to route to per token
 * @param normalizeProbs Whether to normalize router probs after top-K selection
 * @param capacityFactor Expert capacity factor
 */
SD_LIB_HIDDEN void moeSharedExperts(sd::LaunchContext* context,
                                     NDArray* input,
                                     NDArray* routerWeights,
                                     NDArray* routedExpertWeights,
                                     NDArray* sharedGateProj,
                                     NDArray* sharedUpProj,
                                     NDArray* sharedDownProj,
                                     NDArray* routedExpertBias,
                                     NDArray* output,
                                     NDArray* routerProbs,
                                     NDArray* expertIndices,
                                     int numRoutedExperts,
                                     int topK,
                                     bool normalizeProbs,
                                     double capacityFactor);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_MOE_SHARED_EXPERTS_H
