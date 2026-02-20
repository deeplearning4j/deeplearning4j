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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX ReduceLogSum operation.
 *
 * ONNX ReduceLogSum spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum
 *
 * Computes the log of the sum of the input tensor's elements along the specified axes.
 * output = log(sum(input))
 *
 * In ONNX opset 18+, axes became an optional input tensor instead of an attribute.
 *
 * Inputs:
 * - data: Input tensor
 * - axes: Optional, axes along which to reduce (opset 18+)
 *
 * Attributes:
 * - axes: Axes along which to reduce (opset < 18)
 * - keepdims: Whether to keep reduced dimensions (default: 1)
 * - noop_with_empty_axes: If true and axes is empty, return input (default: 0)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["ReduceLogSum"], frameworkName = "onnx")
class ReduceLogSum : PreImportHook {

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

        // Get input data
        val data = sd.getVariable(op.inputsToOp[0])

        // Get keepdims attribute (default: true/1)
        val keepDims = (attributes.getOrDefault("keepdims", 1L) as Number).toInt() != 0

        // Get noop_with_empty_axes attribute (default: false/0)
        val noopWithEmptyAxes = (attributes.getOrDefault("noop_with_empty_axes", 0L) as Number).toInt() != 0

        // Get axes - could be from input (opset 18+) or attribute (old opset)
        var axes: LongArray? = null

        // First check if axes is provided as second input (opset 18+)
        if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null) {
            val axesName = op.inputsToOp[1]

            // First try to get axes from an already-imported constant variable in SameDiff
            val axesVar = sd.getVariable(axesName)
            if (axesVar != null && axesVar.isConstant) {
                val axesArr = sd.getArrForVarName(axesName)
                if (axesArr != null) {
                    axes = LongArray(axesArr.length().toInt()) { i -> axesArr.getLong(i.toLong()) }
                }
            }

            // Fall back to dynamicVariables (for truly dynamic inputs)
            if (axes == null) {
                val axesTensor = dynamicVariables[axesName] as? Onnx.TensorProto
                if (axesTensor != null) {
                    axes = extractAxesFromTensor(axesTensor)
                }
            }
        }

        // Fall back to attribute if no input axes
        if (axes == null) {
            val axesAttr = attributes["axes"]
            axes = when (axesAttr) {
                is List<*> -> axesAttr.map { (it as Number).toLong() }.toLongArray()
                is LongArray -> axesAttr
                is IntArray -> axesAttr.map { it.toLong() }.toLongArray()
                else -> null
            }
        }

        // Handle reduction: log(sum(x))
        val output = when {
            axes == null || axes.isEmpty() -> {
                if (noopWithEmptyAxes) {
                    // Return input unchanged
                    sd.identity(outputNames[0], data)
                } else {
                    // Reduce all axes
                    val sumResult = sd.sum("${opName}_sum", data, keepDims)
                    sd.math.log(outputNames[0], sumResult)
                }
            }
            else -> {
                val sumResult = sd.sum("${opName}_sum", data, keepDims, *axes)
                sd.math.log(outputNames[0], sumResult)
            }
        }

        return mapOf(outputNames[0] to listOf(output))
    }

    /**
     * Extract axes from an ONNX TensorProto, handling different data formats.
     */
    private fun extractAxesFromTensor(tensor: Onnx.TensorProto): LongArray? {
        return when {
            // Handle int64 data (most common for axes)
            tensor.int64DataCount > 0 -> tensor.int64DataList.toLongArray()

            // Handle int32 data
            tensor.int32DataCount > 0 -> tensor.int32DataList.map { it.toLong() }.toLongArray()

            // Handle raw data
            tensor.rawData != null && tensor.rawData.size() > 0 -> {
                val buffer = tensor.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)

                // Determine element size from data type
                when (tensor.dataType) {
                    Onnx.TensorProto.DataType.INT64.number -> {
                        val longBuffer = buffer.asLongBuffer()
                        LongArray(longBuffer.remaining()) { longBuffer.get() }
                    }
                    Onnx.TensorProto.DataType.INT32.number -> {
                        val intBuffer = buffer.asIntBuffer()
                        LongArray(intBuffer.remaining()) { intBuffer.get().toLong() }
                    }
                    Onnx.TensorProto.DataType.INT16.number -> {
                        val shortBuffer = buffer.asShortBuffer()
                        LongArray(shortBuffer.remaining()) { shortBuffer.get().toLong() }
                    }
                    Onnx.TensorProto.DataType.INT8.number -> {
                        LongArray(buffer.remaining()) { buffer.get().toLong() }
                    }
                    else -> {
                        // Default to int64 interpretation
                        val longBuffer = buffer.asLongBuffer()
                        if (longBuffer.remaining() > 0) {
                            LongArray(longBuffer.remaining()) { longBuffer.get() }
                        } else {
                            null
                        }
                    }
                }
            }
            else -> null
        }
    }
}
