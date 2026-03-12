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
import org.nd4j.linalg.api.ops.impl.shape.Reshape as Nd4jReshape
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implements ONNX Reshape operation.
 *
 * ONNX Reshape semantics (default, allowzero=0):
 * - A value of 0 in the shape means "copy the corresponding dimension from input"
 * - A value of -1 means "infer this dimension from the total element count"
 *
 * ONNX Reshape semantics (allowzero=1, opset >= 14):
 * - A value of 0 in the shape means the dimension is literally 0 (empty tensor)
 * - A value of -1 still means "infer this dimension"
 *
 * The C++ reshape op handles the default case (0 = copy from input) natively.
 * When allowzero=1, this hook pre-resolves the shape to avoid the C++ copy-from-input behavior
 * by replacing 0s with the input dimensions at the Kotlin level (for the default case) or
 * leaving them as literal 0s (for the allowzero case, creating empty tensors).
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Reshape"],frameworkName = "onnx")
class Reshape : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val inputVariable = sd.getVariable(op.inputsToOp[0])

        // ONNX opset 14+ allowzero attribute: when 1, 0 means literal zero dimension (empty tensor)
        // Default is 0, meaning 0 = copy from input (standard ONNX Reshape behavior)
        val allowZero = (attributes["allowzero"] as? Long ?: 0L) != 0L

        // Older attributes based shape
        val shapeAttr = attributes["shape"]
        if (shapeAttr != null && shapeAttr is List<*> && shapeAttr.isNotEmpty()) {
            @Suppress("UNCHECKED_CAST")
            val newShape = shapeAttr as List<Int>
            val shapeArr = newShape.toIntArray().map { input -> input.toLong() }.toLongArray()
            // For the static shape case, C++ handles 0-copy-from-input natively.
            // allowzero with static shapes containing 0 will produce empty tensors in C++.
            val finalOutput = sd.updateVariableNameAndReference(
                Nd4jReshape(sd, inputVariable, shapeArr).outputVariable(),
                outputNames[0]
            )
            return mapOf(outputNames[0] to listOf(finalOutput))
        } else {
            // Use shape from second input tensor
            // C++ reshape handles ONNX semantics: 0 = copy from input, -1 = infer
            val shapeVarName = op.inputsToOp[1]
            val shapeVar = sd.getVariable(shapeVarName)

            if (allowZero) {
                // When allowzero=1, 0 means literal 0 (empty tensor), not "copy from input".
                // The C++ reshape treats 0 as "copy from input" by default.
                // To handle allowzero=1 correctly, we check if the shape is a constant
                // and if it contains 0s; if so, we pass it as a static shape array
                // which bypasses the C++ copy-from-input logic for the zero dimensions.
                val shapeArr = shapeVar.arr
                if (shapeArr != null) {
                    val shapeLongs = shapeArr.toLongVector()
                    if (shapeLongs.any { it == 0L }) {
                        // Pass shape as static iArgs — C++ will still try copy-from-input for 0s,
                        // but if the input also has 0 at that position, the copy gives 0 (correct).
                        // If the input has non-zero at that position, this is the allowzero=1 case
                        // and we need the literal 0. Replace 0s with a very small marker that
                        // the C++ empty-shape handling will catch.
                        // Since this is an empty tensor (0 element count), use sd.zeros with the shape.
                        val finalOutput = sd.constant(outputNames[0],
                            org.nd4j.linalg.factory.Nd4j.zeros(inputVariable.dataType(), *shapeLongs))
                        return mapOf(outputNames[0] to listOf(finalOutput))
                    }
                }
            }

            val finalOutput = sd.updateVariableNameAndReference(
                Nd4jReshape(sd, inputVariable, shapeVar).outputVariable(),
                outputNames[0]
            )
            return mapOf(outputNames[0] to listOf(finalOutput))
        }
    }
}
