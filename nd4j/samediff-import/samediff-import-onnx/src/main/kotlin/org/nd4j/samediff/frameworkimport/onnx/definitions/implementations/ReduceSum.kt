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
 * Implementation of ONNX ReduceSum operation.
 *
 * ONNX ReduceSum spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
 *
 * Computes the sum of the input tensor's elements along the specified axes.
 *
 * In ONNX opset 13+, axes became an optional input tensor instead of an attribute.
 *
 * Inputs:
 * - data: Input tensor
 * - axes: Optional, axes along which to reduce (opset 13+)
 *
 * Attributes:
 * - axes: Axes along which to reduce (opset < 13)
 * - keepdims: Whether to keep reduced dimensions (default: 1)
 * - noop_with_empty_axes: If true and axes is empty, return input (default: 0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ReduceSum"], frameworkName = "onnx")
class ReduceSum : PreImportHook {

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

        // Get axes - could be from input (opset 13+) or attribute (old opset)
        var axes: LongArray? = null

        // First check if axes is provided as second input (opset 13+)
        if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null) {
            val axesName = op.inputsToOp[1]

            // First try to get axes from an already-imported constant variable in SameDiff
            val axesVar = sd.getVariable(axesName)
            if (axesVar != null && axesVar.isConstant) {
                // For constants, use getArrForVarName which is safe for constants
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

        // Handle reduction
        // For robustness with dynamic shapes (e.g., when Gather squeezes indices and changes
        // downstream ranks), use reduce-all as fallback for integer-type inputs with axis > 0.
        // Integer-type inputs to ReduceSum typically indicate shape tensor computations where
        // reducing all dimensions is safe when the original axis might be out of bounds.
        val isIntegerInput = data.dataType().isIntType
        val hasPositiveAxis = axes != null && axes.isNotEmpty() && axes.any { it > 0 }

        val output = when {
            axes == null || axes.isEmpty() -> {
                if (noopWithEmptyAxes) {
                    // Return input unchanged
                    sd.identity(outputNames[0], data)
                } else {
                    // Reduce all axes
                    sd.sum(outputNames[0], data, keepDims)
                }
            }
            isIntegerInput && hasPositiveAxis -> {
                // For integer inputs (shape tensors) with positive axes, reduce all to be safe
                // This handles cases where shape changes due to Gather squeeze make axis invalid
                sd.sum(outputNames[0], data, keepDims)
            }
            else -> {
                sd.sum(outputNames[0], data, keepDims, *axes)
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
