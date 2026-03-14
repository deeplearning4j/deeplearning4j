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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX ThresholdedRelu operation.
 *
 * ONNX ThresholdedRelu spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu
 *
 * ThresholdedRelu(x) = x if x > alpha else 0
 *
 * This is a variation of ReLU where the threshold is configurable.
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - alpha: Threshold value (default: 1.0)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["ThresholdedRelu"], frameworkName = "onnx")
class ThresholdedRelu : PreImportHook {

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

        // ThresholdedRelu(x) = x if x > alpha else 0
        // Implemented as: x * (x > alpha)
        val alphaConst = sd.constant("${opName}_alpha", Nd4j.scalar(x.dataType(), alpha))
        val mask = sd.gt("${opName}_mask", x, alphaConst)
        val maskFloat = mask.castTo(x.dataType())
        val output = sd.math.mul(outputNames[0], x, maskFloat)

        return mapOf(outputNames[0] to listOf(output))
    }
}
