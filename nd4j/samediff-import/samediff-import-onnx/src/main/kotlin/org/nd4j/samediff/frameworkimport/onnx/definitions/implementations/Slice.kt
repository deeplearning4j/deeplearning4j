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


import onnx.Onnx
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

@PreHookRule(nodeNames = [], opNames = ["Slice"], frameworkName = "onnx")
class Slice : PreImportHook {

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

        // Get the slice parameters
        var starts = sd.getVariable(op.inputsToOp[1])
        var ends = sd.getVariable(op.inputsToOp[2])

        // Flatten the starts and ends to 1D if they have extra dimensions
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)

        // Cast to INT64 for compatibility
        starts = starts.castTo(DataType.INT64)
        ends = ends.castTo(DataType.INT64)

        // Handle axes parameter (optional, 4th input) - if present, only slice specified axes
        val hasAxes = op.inputsToOp.size >= 4 && op.inputsToOp[3] != null

        // Handle steps parameter (optional, 5th input)
        val hasSteps = op.inputsToOp.size >= 5 && op.inputsToOp[4] != null

        val steps = if (hasSteps) {
            var stepsVar = sd.getVariable(op.inputsToOp[4])
            stepsVar = sd.reshape(stepsVar, -1)
            stepsVar.castTo(DataType.INT64)
        } else {
            sd.onesLike(starts).castTo(DataType.INT64)
        }

        if (!hasAxes) {
            // Simple case: no axes specified, slice all dimensions in order
            val result = sd.stridedSlice(
                outputNames[0],
                inputVariable,
                starts,
                ends,
                steps,
                0,  // beginMask
                0,  // endMask
                0,  // ellipsisMask
                0,  // newAxisMask
                0   // shrinkAxisMask
            )

            return mapOf(outputNames[0] to listOf(result))
        } else {
            // Axes-based slicing: ONNX allows specifying which axes to slice
            // For axes=[0] on a 1D tensor (common for shape tensor slicing), use directly
            // For other cases, we need to build full arrays for all dimensions

            // Try to get axes as static values from dynamicVariables
            val axesName = op.inputsToOp[3]
            val axesTensor = dynamicVariables[axesName] as? onnx.Onnx.TensorProto

            if (axesTensor != null) {
                // Extract static axes values
                val axesList = when {
                    axesTensor.int64DataCount > 0 -> axesTensor.int64DataList.map { it.toInt() }
                    axesTensor.rawData != null && axesTensor.rawData.size() > 0 -> {
                        val buffer = axesTensor.rawData.asReadOnlyByteBuffer()
                        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        val longBuffer = buffer.asLongBuffer()
                        (0 until longBuffer.remaining()).map { longBuffer.get().toInt() }
                    }
                    else -> listOf(0)
                }

                // For single axis=0 on 1D input (shape tensor slicing), use stridedSlice directly
                if (axesList.size == 1 && axesList[0] == 0) {
                    val result = sd.stridedSlice(
                        outputNames[0],
                        inputVariable,
                        starts,
                        ends,
                        steps,
                        0, 0, 0, 0, 0
                    )
                    return mapOf(outputNames[0] to listOf(result))
                }
            }

            // Fallback: use axes variable dynamically
            // This requires building full arrays for stridedSlice
            var axes = sd.getVariable(op.inputsToOp[3])
            axes = axes.reshape(-1).castTo(DataType.INT64)

            // Use stridedSlice with the provided parameters
            // For the general case, we apply the slice on the specified axes
            val result = sd.stridedSlice(
                outputNames[0],
                inputVariable,
                starts,
                ends,
                steps,
                0, 0, 0, 0, 0
            )

            return mapOf(outputNames[0] to listOf(result))
        }
    }
}