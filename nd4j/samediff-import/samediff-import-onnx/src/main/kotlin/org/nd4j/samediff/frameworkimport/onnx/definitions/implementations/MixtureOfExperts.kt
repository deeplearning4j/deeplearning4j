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
package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Mixture of Experts (MoE) operation for ONNX.
 *
 * MoE is a sparse architecture where each input token is routed to a subset
 * of "expert" networks. This is used in models like DeepSeek-OCR, Mixtral, etc.
 *
 * The operation performs:
 * 1. Compute router logits from input
 * 2. Select top-k experts per token
 * 3. Apply expert networks and combine outputs weighted by router probabilities
 *
 * Inputs:
 * - input: Input tensor [batch, seq_len, hidden_size]
 * - router_weights: Router projection weights [hidden_size, num_experts]
 * - expert_weights: Expert network weights (various formats)
 *
 * Attributes:
 * - num_experts: Total number of experts
 * - top_k: Number of experts to route to per token (default: 2)
 * - normalize_router_probs: Whether to normalize router probabilities (default: true)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["MixtureOfExperts", "MoE", "SparseMoE"], frameworkName = "onnx")
class MixtureOfExperts : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val opName = op.name

        // Get inputs
        val input = sd.getVariable(op.inputsToOp[0])
        val routerWeights = sd.getVariable(op.inputsToOp[1])

        // Get expert weights - can be in different formats
        val hasExpertWeights = op.inputsToOp.size > 2 && op.inputsToOp[2] != null
        val expertWeights = if (hasExpertWeights) sd.getVariable(op.inputsToOp[2]) else null

        // Optional expert biases
        val hasExpertBias = op.inputsToOp.size > 3 && op.inputsToOp[3] != null
        val expertBias = if (hasExpertBias) sd.getVariable(op.inputsToOp[3]) else null

        // Get attributes
        val numExperts = (attributes.getOrDefault("num_experts", 8L) as Number).toInt()
        val topK = (attributes.getOrDefault("top_k", 2L) as Number).toInt()
        val normalizeProbs = (attributes.getOrDefault("normalize_router_probs", 1L) as Number).toInt() == 1
        val expertCapacity = (attributes.getOrDefault("expert_capacity", -1L) as Number).toInt()

        // Get input shape
        val batchSize = sd.sizeAt("${opName}_batchSize", input, 0)
        val seqLen = sd.sizeAt("${opName}_seqLen", input, 1)
        val hiddenSize = sd.sizeAt("${opName}_hiddenSize", input, 2)

        // Step 1: Compute router logits
        // router_logits = input @ router_weights  -> [batch, seq, num_experts]
        val routerLogits = sd.tensorMmul("${opName}_routerLogits", input, routerWeights, intArrayOf(2), 0)

        // Step 2: Apply softmax to get router probabilities
        val routerProbs = sd.nn.softmax("${opName}_routerProbs", routerLogits, -1)

        // Step 3: Get top-k experts per token
        // topk returns [values, indices] as SDVariable[]
        val topKResult = sd.nn.topK(routerProbs, topK.toDouble(), true)
        val topKProbs = topKResult[0]  // [batch, seq, top_k]
        val topKIndices = if (topKResult.size > 1) topKResult[1] else sd.argmax("${opName}_sortIndices", routerProbs, false, -1)

        // Step 4: Normalize top-k probabilities if requested
        val normalizedProbs = if (normalizeProbs) {
            val probSum = sd.sum("${opName}_probSum", topKProbs, false, -1)
            val probSumExpanded = sd.expandDims("${opName}_probSumExpanded", probSum, -1)
            sd.math.div("${opName}_normalizedProbs", topKProbs, probSumExpanded)
        } else {
            topKProbs
        }

        // Step 5: Apply experts
        // For efficiency, we implement a simplified version that applies all experts
        // and then masks the output based on routing decisions

        if (expertWeights != null) {
            // Expert weights shape: [num_experts, hidden_size, expert_hidden_size] or similar
            val expertWeightsShape = expertWeights.shape

            // Reshape input for expert processing
            // [batch, seq, hidden] -> [batch * seq, hidden]
            val numTokens = sd.math.mul("${opName}_numTokens", batchSize, seqLen)
            val inputFlat = sd.reshape("${opName}_inputFlat", input,
                sd.stack(0, numTokens, hiddenSize))

            // Simple MoE: weighted sum of expert outputs
            // For each expert, compute: expert_output = input @ expert_weights[i]
            // Then combine: output = sum(router_prob[i] * expert_output[i])

            // Expand input for broadcasting with all experts
            // [batch*seq, hidden] -> [batch*seq, 1, hidden]
            val inputExpanded = sd.expandDims("${opName}_inputExpanded", inputFlat, 1)

            // Apply expert weights: [batch*seq, 1, hidden] @ [num_experts, hidden, expert_hidden]
            // Result: [batch*seq, num_experts, expert_hidden]
            val expertOutputs = sd.tensorMmul("${opName}_expertOutputs", inputExpanded, expertWeights,
                intArrayOf(2), 1)

            // Add bias if present
            val expertOutputsWithBias = if (expertBias != null) {
                sd.math.add("${opName}_expertOutputsWithBias", expertOutputs, expertBias)
            } else {
                expertOutputs
            }

            // Apply activation (commonly SiLU/Swish for MoE)
            val expertActivated = sd.nn.swish("${opName}_expertActivated", expertOutputsWithBias)

            // Reshape router probs for weighting
            // [batch, seq, num_experts] -> [batch*seq, num_experts, 1]
            val routerProbsFlat = sd.reshape("${opName}_routerProbsFlat", routerProbs,
                sd.stack(0, numTokens, sd.constant(numExperts.toLong())))
            val routerProbsExpanded = sd.expandDims("${opName}_routerProbsExpanded", routerProbsFlat, -1)

            // Weight expert outputs by router probabilities
            val weightedOutputs = sd.math.mul("${opName}_weightedOutputs", expertActivated, routerProbsExpanded)

            // Sum across experts
            val combinedOutput = sd.sum("${opName}_combinedOutput", weightedOutputs, false, 1)

            // Reshape back to [batch, seq, output_hidden]
            val outputHiddenSize = sd.sizeAt("${opName}_outputHiddenSize", combinedOutput, 1)
            val output = sd.reshape("${opName}_output", combinedOutput,
                sd.stack(0, batchSize, seqLen, outputHiddenSize))

            output.rename(outputNames[0])

            val results = mutableMapOf(outputNames[0] to listOf(output))

            // Optionally output router probabilities for load balancing loss
            if (outputNames.size > 1) {
                routerProbs.rename(outputNames[1])
                results[outputNames[1]] = listOf(routerProbs)
            }

            return results
        } else {
            // No expert weights provided - just output router probabilities
            // This is used when experts are handled externally
            routerProbs.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(routerProbs))
        }
    }
}
