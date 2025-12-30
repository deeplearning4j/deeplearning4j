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
 * Implementation of ONNX Einsum operation mapping for SameDiff.
 *
 * ONNX Einsum spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#einsum
 *
 * Einsum (Einstein summation) provides a powerful way to express tensor operations
 * using Einstein summation notation. The equation string specifies the subscripts
 * for each input tensor and the output tensor.
 *
 * Examples:
 * - Matrix multiplication: "ij,jk->ik"
 * - Transpose: "ij->ji"
 * - Diagonal: "ii->i"
 * - Trace: "ii->"
 * - Batch matmul: "bij,bjk->bik"
 * - Dot product: "i,i->"
 * - Outer product: "i,j->ij"
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Einsum"], frameworkName = "onnx")
class Einsum : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get the equation attribute (required for Einsum)
        val equation = attributes["equation"] as? String
            ?: throw IllegalArgumentException("Einsum operator requires 'equation' attribute")

        // Get all input variables
        val inputs = op.inputsToOp.map { inputName ->
            sd.getVariable(inputName)
        }.toTypedArray()

        // Call SameDiff's einsum operation with the equation and inputs
        val output = sd.linalg().einsum(outputNames[0], equation, *inputs)

        return mapOf(outputNames[0] to listOf(output))
    }
}
