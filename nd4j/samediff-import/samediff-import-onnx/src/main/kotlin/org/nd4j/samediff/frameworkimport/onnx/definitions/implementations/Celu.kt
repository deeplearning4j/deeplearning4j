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
 * Implementation of ONNX Celu (Continuously Differentiable Exponential Linear Units) operation.
 *
 * ONNX Celu spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu
 *
 * Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
 *
 * This is a smooth approximation to ReLU that allows negative values.
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - alpha: Scaling parameter for the negative region (default: 1.0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Celu"], frameworkName = "onnx")
class Celu : PreImportHook {

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

        // Get alpha attribute (default: 1.0)
        val alpha = (attributes.getOrDefault("alpha", 1.0) as Number).toDouble()

        // Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        val zero = sd.constant("${opName}_zero", 0.0)
        val alphaConst = sd.constant("${opName}_alpha", alpha)

        // Positive part: max(0, x)
        val positivePart = sd.math.max("${opName}_pos", x, zero)

        // Negative part: min(0, alpha * (exp(x/alpha) - 1))
        val xOverAlpha = sd.math.div("${opName}_xOverAlpha", x, alphaConst)
        val expPart = sd.math.exp("${opName}_exp", xOverAlpha)
        val expMinus1 = sd.math.sub("${opName}_expMinus1", expPart, sd.constant(1.0))
        val scaled = sd.math.mul("${opName}_scaled", alphaConst, expMinus1)
        val negativePart = sd.math.min("${opName}_neg", scaled, zero)

        // Combine: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        val output = sd.math.add(outputNames[0], positivePart, negativePart)

        return mapOf(outputNames[0] to listOf(output))
    }
}
