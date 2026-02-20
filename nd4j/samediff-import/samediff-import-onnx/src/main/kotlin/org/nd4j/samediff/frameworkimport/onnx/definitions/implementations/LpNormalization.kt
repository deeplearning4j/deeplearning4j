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
 * Implementation of ONNX LpNormalization operation.
 *
 * ONNX LpNormalization spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization
 *
 * Normalizes the input tensor using Lp normalization:
 * output = input / (sum(abs(input)^p)^(1/p) + epsilon)
 *
 * For p=1: L1 normalization (sum of absolute values)
 * For p=2: L2 normalization (Euclidean norm)
 *
 * Inputs:
 * - input: Input tensor
 *
 * Attributes:
 * - axis: Axis along which to normalize (default: -1)
 * - p: The order of normalization (default: 2)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["LpNormalization"], frameworkName = "onnx")
class LpNormalization : PreImportHook {

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

        // Get attributes
        val axis = (attributes.getOrDefault("axis", -1L) as Number).toInt()
        val p = (attributes.getOrDefault("p", 2L) as Number).toInt()

        // Small epsilon for numerical stability
        val epsilon = 1e-12

        val output = when (p) {
            1 -> {
                // L1 normalization: x / (sum(|x|) + epsilon)
                val absInput = sd.math.abs("${opName}_abs", input)
                val l1Norm = sd.sum("${opName}_l1norm", absInput, true, axis.toLong())
                val l1NormSafe = sd.math.add("${opName}_l1safe", l1Norm, sd.constant(epsilon))
                sd.math.div(outputNames[0], input, l1NormSafe)
            }
            2 -> {
                // L2 normalization: x / (sqrt(sum(x^2)) + epsilon)
                val squared = sd.math.square("${opName}_sq", input)
                val sumSquared = sd.sum("${opName}_sumSq", squared, true, axis.toLong())
                val l2Norm = sd.math.sqrt("${opName}_l2norm", sumSquared)
                val l2NormSafe = sd.math.add("${opName}_l2safe", l2Norm, sd.constant(epsilon))
                sd.math.div(outputNames[0], input, l2NormSafe)
            }
            else -> {
                // General Lp normalization: x / (sum(|x|^p)^(1/p) + epsilon)
                val absInput = sd.math.abs("${opName}_abs", input)
                val powered = sd.math.pow("${opName}_pow", absInput, sd.constant(p.toDouble()))
                val sumPowered = sd.sum("${opName}_sumPow", powered, true, axis.toLong())
                val lpNorm = sd.math.pow("${opName}_lpnorm", sumPowered, sd.constant(1.0 / p))
                val lpNormSafe = sd.math.add("${opName}_lpsafe", lpNorm, sd.constant(epsilon))
                sd.math.div(outputNames[0], input, lpNormSafe)
            }
        }

        return mapOf(outputNames[0] to listOf(output))
    }
}
