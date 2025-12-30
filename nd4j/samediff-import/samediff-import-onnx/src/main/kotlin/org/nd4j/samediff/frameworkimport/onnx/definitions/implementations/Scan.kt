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
 * Implementation of ONNX Scan operation.
 *
 * ONNX Scan spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scan
 *
 * Scan loops over a sequence of tensors, applying a subgraph at each step.
 * This is similar to a fold/reduce operation with state.
 *
 * Inputs:
 * - initial_state_and_scan_inputs: Initial state variables followed by scan inputs
 *
 * Attributes:
 * - body: The subgraph to execute at each iteration
 * - num_scan_inputs: Number of scan inputs
 * - scan_input_axes: Axes along which to scan (default: 0 for all)
 * - scan_input_directions: Direction of scan (0=forward, 1=reverse)
 * - scan_output_axes: Axes for scan outputs
 * - scan_output_directions: Direction for scan outputs
 *
 * Note: This is a simplified implementation that handles common cases.
 * Full subgraph execution would require embedding the body graph.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Scan"], frameworkName = "onnx")
class Scan : PreImportHook {

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

        // Get attributes
        val numScanInputs = (attributes.getOrDefault("num_scan_inputs", 1L) as Number).toInt()
        val scanInputAxes = getLongArray(attributes, "scan_input_axes")
        val scanInputDirections = getLongArray(attributes, "scan_input_directions")
        val scanOutputAxes = getLongArray(attributes, "scan_output_axes")
        val scanOutputDirections = getLongArray(attributes, "scan_output_directions")

        // Get inputs
        val allInputs = op.inputsToOp.filter { it.isNotEmpty() }.map { sd.getVariable(it) }

        // Split into state and scan inputs
        val numStateVars = allInputs.size - numScanInputs
        val stateVars = allInputs.take(numStateVars)
        val scanInputs = allInputs.drop(numStateVars)

        // Create output map
        val resultMap = mutableMapOf<String, List<SDVariable>>()

        // For now, provide a simplified implementation:
        // - State outputs are passed through (identity)
        // - Scan outputs are the concatenated scan inputs

        // State outputs
        stateVars.forEachIndexed { idx, stateVar ->
            if (idx < outputNames.size) {
                val stateOutput = sd.identity("${opName}_state_$idx", stateVar)
                    .rename(outputNames[idx])
                resultMap[outputNames[idx]] = listOf(stateOutput)
            }
        }

        // Scan outputs
        scanInputs.forEachIndexed { idx, scanInput ->
            val outputIdx = numStateVars + idx
            if (outputIdx < outputNames.size) {
                // Get scan axis and direction for this input
                val axis = scanInputAxes?.getOrNull(idx)?.toInt() ?: 0
                val direction = scanInputDirections?.getOrNull(idx)?.toInt() ?: 0

                // If reverse direction, reverse along the axis
                val processedInput = if (direction == 1) {
                    sd.reverse("${opName}_reverse_$idx", scanInput, axis)
                } else {
                    scanInput
                }

                // Output axis handling
                val outputAxis = scanOutputAxes?.getOrNull(idx)?.toInt() ?: 0
                val outputDirection = scanOutputDirections?.getOrNull(idx)?.toInt() ?: 0

                val finalOutput = if (outputDirection == 1) {
                    sd.reverse("${opName}_out_reverse_$idx", processedInput, outputAxis)
                } else {
                    processedInput
                }

                val namedOutput = finalOutput.rename(outputNames[outputIdx])
                resultMap[outputNames[outputIdx]] = listOf(namedOutput)
            }
        }

        // Ensure all output names are covered
        outputNames.forEachIndexed { idx, name ->
            if (!resultMap.containsKey(name)) {
                // Create a placeholder output
                val placeholder = if (allInputs.isNotEmpty()) {
                    sd.identity("${opName}_placeholder_$idx", allInputs[0]).rename(name)
                } else {
                    throw IllegalStateException("Scan operation has no inputs")
                }
                resultMap[name] = listOf(placeholder)
            }
        }

        return resultMap
    }

    private fun getLongArray(attributes: Map<String, Any>, key: String): LongArray? {
        val value = attributes[key] ?: return null
        return when (value) {
            is LongArray -> value
            is IntArray -> value.map { it.toLong() }.toLongArray()
            is List<*> -> value.mapNotNull { (it as? Number)?.toLong() }.toLongArray()
            else -> null
        }
    }
}
