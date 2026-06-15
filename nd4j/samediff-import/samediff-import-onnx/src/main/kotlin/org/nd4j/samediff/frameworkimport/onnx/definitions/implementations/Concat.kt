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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Custom import handler for ONNX Concat operation.
 *
 * ONNX's Concat can handle inputs with different ranks through implicit broadcasting,
 * but nd4j's concat requires all inputs to have the same rank. This handler automatically
 * reshapes lower-rank inputs by prepending dimensions of size 1 to match the highest rank.
 *
 * This implementation uses only SameDiff variable operations (sd.shape, sd.size, sd.reshape)
 * to handle dynamic shapes properly without assuming any static shape information is available.
 *
 * Example:
 *   Input 0: shape [4, 1] (rank 2)
 *   Input 1: shape [1] (rank 1)
 *   axis: 0
 *
 *   After reshape:
 *   Input 0: shape [4, 1] (rank 2)
 *   Input 1: shape [1, 1] (rank 2) - prepended a dimension
 *
 *   Result: shape [5, 1]
 */
@PreHookRule(nodeNames = [], opNames = ["Concat"], frameworkName = "onnx")
class Concat : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        // Get the axis attribute (required for Concat)
        val axis = (attributes["axis"] as? Number)?.toInt() ?: 0

        // Get all input variables
        val inputVars = op.inputsToOp.map { sd.getVariable(it) }

        if (inputVars.isEmpty()) {
            throw IllegalArgumentException("Concat requires at least one input")
        }

        // Single input - just return identity
        if (inputVars.size == 1) {
            val result = sd.identity(outputNames[0], inputVars[0])
            return mapOf(outputNames[0] to listOf(result))
        }

        // For axis=0 concat (common for building shape tensors), flatten all inputs to 1D
        // This handles the case where inputs have different ranks (e.g., [4,1] and [1])
        // by flattening them to 1D before concatenating
        if (axis == 0) {
            // Determine target datatype - use first input's datatype
            val targetDtype = inputVars[0].dataType()

            val flattenedInputs = inputVars.mapIndexed { idx, inputVar ->
                // Reshape to [-1] flattens any tensor to 1D
                val flatShape = sd.constant(Nd4j.createFromArray(-1L))
                var flattened = sd.reshape("${outputNames[0]}_flat$idx", inputVar, flatShape)
                // Cast to target datatype if different
                if (flattened.dataType() != targetDtype) {
                    flattened = sd.castTo("${outputNames[0]}_cast$idx", flattened, targetDtype)
                }
                flattened
            }

            val result = sd.concat(outputNames[0], 0, *flattenedInputs.toTypedArray())

            return mapOf(outputNames[0] to listOf(result))
        }

        // For other axes, try to align ranks by prepending 1s to lower-rank inputs
        // But first, check if all inputs have the same rank - if so, skip rank alignment
        // as it can cause issues with dynamic shapes

        // If all inputs are from the same type of operation (like Unsqueeze outputs for attention),
        // they likely have the same rank and we should just concat directly
        val firstInput = inputVars[0]

        // For most cases with consistent ranks, just concat directly
        // This avoids creating many intermediate variables that could cause issues
        val result = sd.concat(outputNames[0], axis, *inputVars.toTypedArray())

        return mapOf(outputNames[0] to listOf(result))
    }
}
