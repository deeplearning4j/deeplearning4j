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
// Sparse Mixture of Experts (MoE) for efficient large models
//

#ifndef LIBND4J_HELPERS_MIXTURE_OF_EXPERTS_H
#define LIBND4J_HELPERS_MIXTURE_OF_EXPERTS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief Mixture of Experts forward pass with Top-K routing
 *
 * Implements sparse MoE layer used in models like Mixtral, Switch Transformer, etc.
 * Routes each token to top-K experts and combines their outputs.
 *
 * @param context Launch context for execution
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param gatingWeights Router/gating weights [hidden_dim, num_experts]
 * @param expertWeights Expert weights - list of [hidden_dim, expert_dim] or single [num_experts, hidden_dim, expert_dim]
 * @param expertBias Expert biases (optional)
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param routerLogits Output router logits [batch, seq_len, num_experts] (optional)
 * @param expertIndices Output selected expert indices [batch, seq_len, topK] (optional)
 * @param numExperts Total number of experts
 * @param topK Number of experts to route to per token
 * @param expertCapacity Maximum tokens per expert (0 for unlimited)
 * @param useSoftmaxGating Whether to apply softmax to gating weights
 * @param useAuxLoss Whether to compute auxiliary load balancing loss
 */
SD_LIB_HIDDEN void mixtureOfExperts(sd::LaunchContext* context,
                                     NDArray* input,
                                     NDArray* gatingWeights,
                                     const std::vector<NDArray*>& expertWeights,
                                     const std::vector<NDArray*>& expertBias,
                                     NDArray* output,
                                     NDArray* routerLogits,
                                     NDArray* expertIndices,
                                     int numExperts,
                                     int topK,
                                     int expertCapacity,
                                     bool useSoftmaxGating,
                                     bool useAuxLoss);

/**
 * @brief Simple MoE with single expert weight tensor
 *
 * Simplified version where all expert weights are stored in a single tensor.
 *
 * @param context Launch context for execution
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param gatingWeights Router weights [hidden_dim, num_experts]
 * @param expertWeights All expert weights [num_experts, hidden_dim, expert_dim]
 * @param expertBias All expert biases [num_experts, expert_dim] (optional)
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param numExperts Total number of experts
 * @param topK Number of experts per token
 * @param normalizeProbs Whether to renormalize the selected top-K expert scores.
 *        When true, the raw softmax scores for the selected experts are divided
 *        by their sum, rescaling them to sum to 1.0.  This introduces a gradient
 *        scaling factor of approximately numExperts/topK in the backward pass
 *        (e.g. ~16x for numExperts=32, topK=2), which causes the "~30x too large
 *        gradients" bug.  Pass false to use the raw softmax probabilities, which
 *        is the mathematically correct behaviour for gradient-based training.
 *        Default: false (gradient-safe).
 */
SD_LIB_HIDDEN void mixtureOfExpertsSimple(sd::LaunchContext* context,
                                           NDArray* input,
                                           NDArray* gatingWeights,
                                           NDArray* expertWeights,
                                           NDArray* expertBias,
                                           NDArray* output,
                                           int numExperts,
                                           int topK,
                                           bool normalizeProbs = false);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_MIXTURE_OF_EXPERTS_H
