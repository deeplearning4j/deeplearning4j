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
 * Implementation of ONNX Compress operation.
 *
 * ONNX Compress spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress
 *
 * Selects slices from an input tensor along a given axis where condition evaluates to True.
 * Similar to numpy.compress.
 *
 * Inputs:
 * - input: Input tensor
 * - condition: Boolean tensor indicating which elements to select
 *
 * Attributes:
 * - axis: Axis along which to take slices. If not specified, input is flattened first.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Compress"], frameworkName = "onnx")
class Compress : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])
        val condition = sd.getVariable(op.inputsToOp[1])

        // Get axis attribute (optional)
        val axisAttr = attributes["axis"]

        // Convert condition to indices using where/nonzero
        // condition should be 1D boolean tensor
        val conditionInt = condition.castTo(DataType.INT64)

        // Find indices where condition is true (non-zero)
        // We'll use the nonzero operation or implement with a workaround
        val indices = sd.matchConditionIdx("${opName}_indices", conditionInt,
            org.nd4j.linalg.indexing.conditions.Conditions.notEquals(0.0))

        val output = if (axisAttr != null) {
            val axis = (axisAttr as Number).toInt()
            // Gather along the specified axis
            sd.gather(outputNames[0], input, indices.castTo(DataType.INT64), axis)
        } else {
            // Flatten input first, then gather
            val flattened = sd.reshape("${opName}_flat", input, -1L)
            sd.gather(outputNames[0], flattened, indices.castTo(DataType.INT64), 0)
        }

        return mapOf(outputNames[0] to listOf(output))
    }
}
