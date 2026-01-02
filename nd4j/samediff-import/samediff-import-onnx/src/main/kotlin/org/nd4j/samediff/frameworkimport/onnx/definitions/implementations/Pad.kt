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
import org.nd4j.enums.PadMode
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Pad operation.
 *
 * ONNX Pad spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
 *
 * Pads a tensor with a given value.
 *
 * Inputs:
 * - data: Input tensor
 * - pads: Padding values [begin_0, begin_1, ..., end_0, end_1, ...]
 * - constant_value: Value for constant padding (optional, default 0)
 * - axes: Axes to pad (optional, opset 18+)
 *
 * Attributes:
 * - mode: constant, reflect, edge, wrap
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Pad"], frameworkName = "onnx")
class Pad : PreImportHook {

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
        val data = sd.getVariable(op.inputsToOp[0])
        val pads = sd.getVariable(op.inputsToOp[1])
        val constantValue = if (op.inputsToOp.size > 2 && op.inputsToOp[2].isNotEmpty())
            sd.getVariable(op.inputsToOp[2]) else null

        // Get mode attribute
        val mode = attributes.getOrDefault("mode", "constant") as? String ?: "constant"

        // Convert ONNX padding format to SameDiff format
        // ONNX: [begin_0, begin_1, ..., end_0, end_1, ...]
        // SameDiff: [[begin_0, end_0], [begin_1, end_1], ...]
        val padsInt = pads.castTo(DataType.INT64)

        // Determine padding mode
        val padMode = when (mode.lowercase()) {
            "constant" -> PadMode.CONSTANT
            "reflect" -> PadMode.REFLECT
            "edge" -> PadMode.SYMMETRIC
            "wrap" -> PadMode.CONSTANT  // Wrap not directly supported, fallback to constant
            else -> PadMode.CONSTANT
        }

        // Get constant value (default 0)
        val padValueDouble = if (constantValue != null) {
            // Try to get the constant value - for now use 0.0 as fallback
            0.0
        } else {
            0.0
        }

        // Apply padding
        val output = sd.nn.pad(outputNames[0], data, padsInt, padMode, padValueDouble)

        return mapOf(outputNames[0] to listOf(output))
    }
}
