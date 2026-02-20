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
 * Implementation of ONNX BitShift operation.
 *
 * ONNX BitShift spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift
 *
 * Bitwise shift operator that performs element-wise left or right shift.
 * Left shift: X << Y (equivalent to X * 2^Y)
 * Right shift: X >> Y (equivalent to X / 2^Y)
 *
 * Inputs:
 * - X: First input tensor
 * - Y: Second input tensor (shift amount)
 *
 * Attributes:
 * - direction: "LEFT" or "RIGHT"
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["BitShift"], frameworkName = "onnx")
class Bitshift : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])
        val y = sd.getVariable(op.inputsToOp[1])

        // Get direction attribute
        val direction = (attributes["direction"] as? String)?.uppercase() ?: "LEFT"

        // Implement bit shift using multiplication/division by powers of 2
        // Left shift: X << Y = X * 2^Y
        // Right shift: X >> Y = X / 2^Y (integer division)
        val two = sd.constant("${opName}_two", 2.0)
        val powerOf2 = sd.math.pow("${opName}_pow2", two, y.castTo(x.dataType()))

        val output = if (direction == "LEFT") {
            // Left shift: multiply by 2^Y
            val result = sd.math.mul("${opName}_shift", x.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE), powerOf2)
            result.castTo(x.dataType())
        } else {
            // Right shift: divide by 2^Y (floor division for integers)
            val divided = sd.math.div("${opName}_div", x.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE), powerOf2)
            sd.math.floor("${opName}_shift", divided).castTo(x.dataType())
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }
}
