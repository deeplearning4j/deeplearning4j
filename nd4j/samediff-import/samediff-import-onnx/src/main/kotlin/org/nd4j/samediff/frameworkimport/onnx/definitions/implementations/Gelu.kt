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
 * Implementation of ONNX Gelu operation.
 *
 * ONNX Gelu spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu
 *
 * Gaussian Error Linear Unit activation function.
 * gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * Or approximate version:
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - approximate: Use approximate computation (default: none)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Gelu"], frameworkName = "onnx")
class Gelu : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])

        // Get approximate attribute
        val approximate = attributes.getOrDefault("approximate", "none") as? String ?: "none"

        // Apply GELU activation
        val output = if (approximate.lowercase() == "tanh") {
            // Approximate GELU using tanh
            // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            val sqrt2OverPi = 0.7978845608028654  // sqrt(2/pi)
            val coef = 0.044715

            val xCubed = sd.math.pow(x, 3.0)
            val inner = sd.math.add(x, sd.math.mul(xCubed, coef))
            val tanhArg = sd.math.mul(inner, sqrt2OverPi)
            val tanhVal = sd.math.tanh(tanhArg)
            val onePlusTanh = sd.math.add(tanhVal, 1.0)
            sd.math.mul("${opName}_gelu", sd.math.mul(x, 0.5), onePlusTanh)
        } else {
            // Exact GELU using erf
            // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            val sqrtTwo = 1.4142135623730951
            val erfArg = sd.math.div(x, sqrtTwo)
            val erfVal = sd.math.erf(erfArg)
            val onePlusErf = sd.math.add(erfVal, 1.0)
            sd.math.mul("${opName}_gelu", sd.math.mul(x, 0.5), onePlusErf)
        }

        val finalOutput = output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
    }
}
