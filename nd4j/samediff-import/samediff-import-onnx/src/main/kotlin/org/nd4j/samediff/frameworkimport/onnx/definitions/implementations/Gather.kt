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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Gather operation mapping for SameDiff.
 * Maps the ONNX Gather operation to SameDiff's gather function.
 *
 * ONNX Gather spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather
 * Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis dimension of data
 * (axis specified as an attribute) indexed by indices, and concatenates them in an output tensor of rank q + (r - 1).
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Gather"], frameworkName = "onnx")
class Gather : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get input variables
        val dataVariable = sd.getVariable(op.inputsToOp[0])
        var indicesVariable = sd.getVariable(op.inputsToOp[1])

        // Get axis attribute (default to 0 if not specified)
        var axis = attributes["axis"]?.let { (it as Long).toInt() } ?: 0

        // ONNX allows float indices but SameDiff requires integer indices
        // Cast to INT64 if indices are not already an integer type
        val indicesType = indicesVariable.dataType()
        if (!indicesType.isIntType) {
            indicesVariable = sd.castTo("${outputNames[0]}_indices_int", indicesVariable, DataType.INT64)
        }

        // Handle scalar-like CONSTANT indices. ONNX models exported from PyTorch often have
        // scalar indices wrapped as [1] or [1,1]. Per ONNX Gather spec, output rank =
        // indices_rank + data_rank - 1, so extra dimensions in indices propagate to output.
        //
        // For CONSTANT indices with exactly one element, squeeze to scalar to match PyTorch behavior.
        // Uses sd.getArrForVarName() for constants (same pattern as ReduceSum.kt) - this is
        // different from accessing .arr property directly.
        val indicesVarType = indicesVariable.variableType
        if (indicesVarType == org.nd4j.autodiff.samediff.VariableType.CONSTANT) {
            val indicesArr = sd.getArrForVarName(indicesVariable.name())
            if (indicesArr != null && indicesArr.length() == 1L) {
                // Single-element constant - squeeze to scalar for PyTorch compatibility
                indicesVariable = sd.squeezeAll("${outputNames[0]}_indices_squeezed", indicesVariable)
            }
        }

        // Handle negative axis per ONNX spec: axis can be negative,
        // meaning it counts from the last dimension. We resolve it using
        // the data variable's known rank when available.
        if (axis < 0) {
            val dataShape = dataVariable.shape
            if (dataShape != null) {
                axis = dataShape.size + axis
            }
        }

        val outputVar = sd.gather(outputNames[0], dataVariable, indicesVariable, axis)

        return mapOf(outputNames[0] to listOf(outputVar))
    }
}
