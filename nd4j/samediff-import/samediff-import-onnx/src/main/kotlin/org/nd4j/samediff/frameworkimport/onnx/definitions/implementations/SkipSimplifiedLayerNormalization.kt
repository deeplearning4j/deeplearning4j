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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Microsoft ONNX SkipSimplifiedLayerNormalization operation.
 *
 * This operation combines a skip (residual) connection with SimplifiedLayerNormalization (RMS Norm):
 *
 * 1. Add skip tensor to input: hidden_states = input + skip
 * 2. Apply RMS Norm: output = hidden_states / sqrt(mean(hidden_states^2) + epsilon) * gamma
 *
 * This is commonly used in transformer models like LLaMA, Phi, Mistral where
 * residual connections are fused with layer normalization for efficiency.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization
 *
 * Inputs:
 * - input: Input tensor [batch, seq_len, hidden_size]
 * - skip: Skip tensor to add [batch, seq_len, hidden_size]
 * - gamma: Scale parameter [hidden_size]
 * - bias (optional): Bias parameter [hidden_size]
 *
 * Attributes:
 * - epsilon: Small constant for numerical stability (default: 1e-5)
 *
 * Outputs:
 * - output: Normalized output [batch, seq_len, hidden_size]
 * - (optional) mean: Mean of input (not typically used)
 * - (optional) inv_std_var: Inverse standard deviation (for backward pass)
 * - (optional) input_skip_bias_sum: Sum of input + skip + bias (intermediate result)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["SkipSimplifiedLayerNormalization"], frameworkName = "onnx")
class SkipSimplifiedLayerNormalization : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        // Get inputs
        val input = sd.getVariable(op.inputsToOp[0])
        val skip = sd.getVariable(op.inputsToOp[1])
        val gamma = sd.getVariable(op.inputsToOp[2])

        // Optional bias (input index 3)
        val hasBias = op.inputsToOp.size > 3 &&
                      op.inputsToOp[3].isNotEmpty() &&
                      sd.hasVariable(op.inputsToOp[3])
        val bias = if (hasBias) sd.getVariable(op.inputsToOp[3]) else null

        // Get epsilon attribute
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()

        // Step 1: Add skip connection (residual)
        var hiddenStates = sd.math.add(input, skip)

        // Add bias if present
        if (bias != null) {
            hiddenStates = sd.math.add(hiddenStates, bias)
        }

        // Step 2: RMS Norm: x / sqrt(mean(x^2) + epsilon) * gamma
        // Normalization is typically over the last axis (hidden_size dimension)
        val axis = -1

        val squared = sd.math.pow(hiddenStates, 2.0)
        val meanSquared = sd.math.mean(squared, true, axis.toLong())
        val rms = sd.math.sqrt(sd.math.add(meanSquared, epsilon))

        // Normalize and scale
        val normalized = sd.math.div(hiddenStates, rms)
        val result = sd.math.mul(outputNames[0], normalized, gamma)

        val outputMap = mutableMapOf<String, List<SDVariable>>()
        outputMap[outputNames[0]] = listOf(result)

        // Handle optional outputs (some models use them for backward pass)
        // Output 1: mean (placeholder - SimplifiedLayerNorm doesn't compute mean)
        // Output 2: inv_std_var (inverse of rms for backward)
        // Output 3: input_skip_bias_sum (the sum before normalization)

        if (outputNames.size > 1 && outputNames[1].isNotEmpty()) {
            // Mean is not computed in RMS norm, but some graphs expect this output
            // Use a zero placeholder or skip it
            val placeholder = sd.zerosLike(outputNames[1], meanSquared)
            outputMap[outputNames[1]] = listOf(placeholder)
        }

        if (outputNames.size > 2 && outputNames[2].isNotEmpty()) {
            // Inverse RMS for backward pass
            val invRms = sd.math.reciprocal(outputNames[2], rms)
            outputMap[outputNames[2]] = listOf(invRms)
        }

        if (outputNames.size > 3 && outputNames[3].isNotEmpty()) {
            // input + skip + bias sum (pre-normalization hidden states)
            hiddenStates.rename(outputNames[3])
            outputMap[outputNames[3]] = listOf(hiddenStates)
        }

        return outputMap
    }
}
