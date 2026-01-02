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
 * Implementation of ONNX Normalizer operation (ML domain).
 *
 * ONNX Normalizer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Normalizer
 *
 * Normalizes input values using the specified norm type.
 * Supports L1, L2, and MAX normalization.
 *
 * Inputs:
 * - X: Input tensor of shape [N, C]
 *
 * Attributes:
 * - norm: "MAX", "L1", "L2" (default: "MAX")
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Normalizer"], frameworkName = "onnx")
class Normalizer : PreImportHook {

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
        
        // Get input
        val input = sd.getVariable(op.inputsToOp[0])
        
        // Get norm type (default: MAX)
        val normType = attributes.getOrDefault("norm", "MAX") as? String ?: "MAX"
        
        // Normalize along the last axis (features)
        val axis = -1
        val epsilon = 1e-10
        
        val output = when (normType.uppercase()) {
            "MAX" -> {
                // Divide by max absolute value
                val maxVal = sd.max("${opName}_max", sd.math.abs(input), true, axis.toLong())
                val safeMax = sd.math.add("${opName}_safeMax", maxVal, epsilon)
                sd.math.div(outputNames[0], input, safeMax)
            }
            "L1" -> {
                // Divide by L1 norm (sum of absolute values)
                val l1Norm = sd.sum("${opName}_l1", sd.math.abs(input), true, axis.toLong())
                val safeL1 = sd.math.add("${opName}_safeL1", l1Norm, epsilon)
                sd.math.div(outputNames[0], input, safeL1)
            }
            "L2" -> {
                // Divide by L2 norm (sqrt of sum of squares)
                val l2NormSq = sd.sum("${opName}_l2sq", sd.math.square(input), true, axis.toLong())
                val l2Norm = sd.math.sqrt("${opName}_l2", l2NormSq)
                val safeL2 = sd.math.add("${opName}_safeL2", l2Norm, epsilon)
                sd.math.div(outputNames[0], input, safeL2)
            }
            else -> {
                // Default to MAX
                val maxVal = sd.max("${opName}_max", sd.math.abs(input), true, axis.toLong())
                val safeMax = sd.math.add("${opName}_safeMax", maxVal, epsilon)
                sd.math.div(outputNames[0], input, safeMax)
            }
        }
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
