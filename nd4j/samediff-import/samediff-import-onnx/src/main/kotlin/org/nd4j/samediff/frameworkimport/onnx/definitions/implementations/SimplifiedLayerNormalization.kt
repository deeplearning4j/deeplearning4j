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
 * Implementation of Microsoft ONNX SimplifiedLayerNormalization operation.
 *
 * SimplifiedLayerNormalization (also known as RMS Norm) normalizes the input
 * using root mean square instead of full mean/variance normalization:
 *
 * output = x / sqrt(mean(x^2) + epsilon) * scale
 *
 * This is used in models like LLaMA, Mistral, etc.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#SimplifiedLayerNormalization
 *
 * Inputs:
 * - X: Input tensor
 * - scale: Scale tensor
 *
 * Attributes:
 * - axis: The axis for normalization (default: -1)
 * - epsilon: Small constant for numerical stability (default: 1e-5)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["SimplifiedLayerNormalization"], frameworkName = "onnx")
class SimplifiedLayerNormalization : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input = sd.getVariable(op.inputsToOp[0])
        val scale = sd.getVariable(op.inputsToOp[1])

        // Get attributes
        val axis = (attributes.getOrDefault("axis", -1) as Number).toInt()
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()

        // RMS Norm: x / sqrt(mean(x^2) + epsilon) * scale
        val squared = sd.math.pow(input, 2.0)
        val meanSquared = sd.math.mean(squared, false, axis.toLong())
        val rms = sd.math.sqrt(sd.math.add(meanSquared, epsilon))

        // Expand dims for broadcasting if necessary
        val rmsExpanded = sd.expandDims(rms, axis)

        // Normalize and scale
        val normalized = sd.math.div(input, rmsExpanded)
        val result = sd.math.mul(normalized, scale)

        result.rename(outputNames[0])

        // Some implementations also output the inverse RMS for backward pass
        if (outputNames.size > 1) {
            val invRms = sd.math.reciprocal(rmsExpanded)
            invRms.rename(outputNames[1])
            return mapOf(
                outputNames[0] to listOf(result),
                outputNames[1] to listOf(invRms)
            )
        }

        return mapOf(outputNames[0] to listOf(result))
    }
}
