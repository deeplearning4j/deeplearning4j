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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Unsqueeze operation.
 *
 * ONNX Unsqueeze inserts single-dimensional entries into the shape of an input tensor.
 * Axes are specified as integers indicating where to insert the new dimensions.
 *
 * For opset < 13: axes are specified as an attribute
 * For opset >= 13: axes are specified as an input tensor
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [],opNames = ["Unsqueeze"],frameworkName = "onnx")
class Unsqueeze  : PreImportHook {
    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#unsqueeze

        val input = sd.getVariable(op.inputsToOp[0])

        // Get axes - either from attribute (older opsets) or from input tensor (opset 13+)
        val axes: List<Int> = if (op.inputsToOp.size < 2) {
            // Axes from attribute (opset < 13)
            @Suppress("UNCHECKED_CAST")
            attributes["axes"] as List<Int>
        } else {
            // Axes from input tensor (opset >= 13)
            val axesVarName = op.inputsToOp[1]
            // First try dynamicVariables (ONNX TensorProto)
            getAxesFromTensorProto(dynamicVariables, axesVarName)
                ?: throw IllegalStateException(
                    "Unsqueeze: Could not find axes tensor '$axesVarName'. " +
                    "The axes must be available as an ONNX initializer constant."
                )
        }

        // Sort axes to handle them in order (ONNX allows negative and unordered axes)
        val sortedAxes = axes.sorted()

        // Apply expandDims for each axis in sorted order
        var current = input
        for (i in sortedAxes.indices) {
            val axis = sortedAxes[i]
            // Only the final operation gets the output name
            val opName = if (i == sortedAxes.size - 1) outputNames[0] else null
            current = if (opName != null) {
                sd.expandDims(opName, current, axis)
            } else {
                sd.expandDims(current, axis)
            }
        }

        return mapOf(current.name() to listOf(current))
    }

    /**
     * Extract axes values from ONNX TensorProto in dynamicVariables.
     */
    private fun getAxesFromTensorProto(
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): List<Int>? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.int64DataCount > 0 -> tensorProto.int64DataList.map { it.toInt() }
            tensorProto.int32DataCount > 0 -> tensorProto.int32DataList.toList()
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val result = mutableListOf<Int>()
                if (tensorProto.dataType == Onnx.TensorProto.DataType.INT64_VALUE) {
                    val longBuffer = buffer.asLongBuffer()
                    while (longBuffer.hasRemaining()) {
                        result.add(longBuffer.get().toInt())
                    }
                } else {
                    val intBuffer = buffer.asIntBuffer()
                    while (intBuffer.hasRemaining()) {
                        result.add(intBuffer.get())
                    }
                }
                result
            }
            else -> null
        }
    }
}