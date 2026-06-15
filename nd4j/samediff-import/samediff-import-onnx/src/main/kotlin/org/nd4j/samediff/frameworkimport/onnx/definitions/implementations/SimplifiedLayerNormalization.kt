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
 * @author Adam Gibson
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
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()

        // Use fused rms_norm op — single kernel instead of decomposed pow/mean/sqrt/div/mul
        val result = sd.nn().rmsNorm(outputNames[0], input, scale, epsilon)

        // Output 1: inv_rms — only needed for training backward pass, not inference.
        // Emit a zeros placeholder instead of the expensive decomposed chain
        // (pow→mean→sqrt→reciprocal) which adds 4 ops per norm layer that execute
        // every decode step even though no inference-path op consumes the result.
        if (outputNames.size > 1 && outputNames[1].isNotEmpty()) {
            val placeholder = sd.zerosLike(outputNames[1], input)
            return mapOf(
                outputNames[0] to listOf(result),
                outputNames[1] to listOf(placeholder)
            )
        }

        return mapOf(outputNames[0] to listOf(result))
    }
}
