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
 * Implementation of ONNX ScatterElements operation using native libnd4j scatter ops.
 *
 * ScatterElements scatters updates into a copy of the input data tensor
 * at specified indices. This is the inverse of GatherElements.
 *
 * Uses native sd.scatterUpdate(), sd.scatterAdd(), sd.scatterNdAdd/Update() ops.
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements
 *
 * Inputs:
 * - data: Input tensor to scatter into
 * - indices: Indices tensor (same rank as data)
 * - updates: Values to scatter (same shape as indices)
 *
 * Attributes:
 * - axis: Axis along which to scatter (default: 0)
 * - reduction: Reduction mode: "none", "add", "mul", "max", "min" (default: "none")
 *
 * Output:
 * - output: Tensor with scattered values
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ScatterElements"], frameworkName = "onnx")
class ScatterElements : PreImportHook {

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

        val data = sd.getVariable(op.inputsToOp[0])
        val indices = sd.getVariable(op.inputsToOp[1])
        val updates = sd.getVariable(op.inputsToOp[2])

        // Get attributes
        val axis = (attributes.getOrDefault("axis", 0L) as Number).toInt()
        val reduction = attributes.getOrDefault("reduction", "none") as? String ?: "none"

        val dataShape = data.shape
        val rank = dataShape?.size ?: 0

        // Normalize axis
        val normalizedAxis = if (axis < 0) rank + axis else axis

        // Handle based on reduction mode
        val output = when (reduction.lowercase()) {
            "none" -> scatterNone(sd, opName, data, indices, updates, normalizedAxis)
            "add" -> scatterAdd(sd, opName, data, indices, updates, normalizedAxis)
            "mul" -> scatterMul(sd, opName, data, indices, updates, normalizedAxis)
            "max" -> scatterMax(sd, opName, data, indices, updates, normalizedAxis)
            "min" -> scatterMin(sd, opName, data, indices, updates, normalizedAxis)
            else -> scatterNone(sd, opName, data, indices, updates, normalizedAxis)
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    /**
     * Scatter with no reduction (replace values) using native ops.
     */
    private fun scatterNone(
        sd: SameDiff,
        opName: String,
        data: SDVariable,
        indices: SDVariable,
        updates: SDVariable,
        axis: Int
    ): SDVariable {
        val dataShape = data.shape
        val rank = dataShape?.size ?: 0

        // For axis=0 case, use native scatterUpdate directly
        if (axis == 0 && rank <= 2) {
            return sd.scatterUpdate("${opName}_scatter", data, indices, updates)
        }

        // For other axes, compute linear indices and use scatterNdUpdate
        val flattened = sd.reshape("${opName}_dataFlat", data, -1)
        val updatesFlat = sd.reshape("${opName}_updatesFlat", updates, -1)

        val linearIndices = computeLinearIndices(sd, opName, indices, dataShape, axis)
        val linearIndicesExpanded = sd.expandDims("${opName}_linIdxExp", linearIndices, 1)

        // Use native scatterNdUpdate
        val result = sd.scatterNdUpdate("${opName}_scatterNdUpdate", flattened, linearIndicesExpanded, updatesFlat)

        return sd.reshape("${opName}_output", result, *dataShape!!)
    }

    /**
     * Scatter with add reduction using native scatterAdd.
     */
    private fun scatterAdd(
        sd: SameDiff,
        opName: String,
        data: SDVariable,
        indices: SDVariable,
        updates: SDVariable,
        axis: Int
    ): SDVariable {
        val dataShape = data.shape ?: return data

        // For axis=0 case, use native scatterAdd directly
        if (axis == 0) {
            return sd.scatterAdd("${opName}_scatterAdd", data, indices, updates)
        }

        // For other axes, compute linear indices and use scatterNdAdd
        val flattened = sd.reshape("${opName}_dataFlat", data, -1)
        val updatesFlat = sd.reshape("${opName}_updatesFlat", updates, -1)
        val linearIndices = computeLinearIndices(sd, opName, indices, dataShape, axis)
        val linearIndicesExpanded = sd.expandDims("${opName}_linIdxExp", linearIndices, 1)

        val scattered = sd.scatterNdAdd("${opName}_scatterNdAdd", flattened, linearIndicesExpanded, updatesFlat)

        return sd.reshape("${opName}_output", scattered, *dataShape)
    }

    /**
     * Scatter with multiply reduction.
     * No native op, so we gather, multiply, and scatter back.
     */
    private fun scatterMul(
        sd: SameDiff,
        opName: String,
        data: SDVariable,
        indices: SDVariable,
        updates: SDVariable,
        axis: Int
    ): SDVariable {
        val dataShape = data.shape ?: return data

        val flattened = sd.reshape("${opName}_dataFlat", data, -1)
        val updatesFlat = sd.reshape("${opName}_updatesFlat", updates, -1)
        val linearIndices = computeLinearIndices(sd, opName, indices, dataShape, axis)

        // Gather existing values, multiply with updates, then scatter back
        val gathered = sd.gather("${opName}_gather", flattened, linearIndices, 0)
        val multiplied = sd.math.mul("${opName}_mul", gathered, updatesFlat)

        val linearIndicesExpanded = sd.expandDims("${opName}_linIdxExp", linearIndices, 1)
        val result = sd.scatterNdUpdate("${opName}_scatterNdUpdate", flattened, linearIndicesExpanded, multiplied)

        return sd.reshape("${opName}_output", result, *dataShape)
    }

    /**
     * Scatter with max reduction.
     * No native op, so we gather, max, and scatter back.
     */
    private fun scatterMax(
        sd: SameDiff,
        opName: String,
        data: SDVariable,
        indices: SDVariable,
        updates: SDVariable,
        axis: Int
    ): SDVariable {
        val dataShape = data.shape ?: return data

        val flattened = sd.reshape("${opName}_dataFlat", data, -1)
        val updatesFlat = sd.reshape("${opName}_updatesFlat", updates, -1)
        val linearIndices = computeLinearIndices(sd, opName, indices, dataShape, axis)

        // Gather existing values, take max with updates, then scatter back
        val gathered = sd.gather("${opName}_gather", flattened, linearIndices, 0)
        val maxed = sd.math.max("${opName}_max", gathered, updatesFlat)

        val linearIndicesExpanded = sd.expandDims("${opName}_linIdxExp", linearIndices, 1)
        val result = sd.scatterNdUpdate("${opName}_scatterNdUpdate", flattened, linearIndicesExpanded, maxed)

        return sd.reshape("${opName}_output", result, *dataShape)
    }

    /**
     * Scatter with min reduction.
     * No native op, so we gather, min, and scatter back.
     */
    private fun scatterMin(
        sd: SameDiff,
        opName: String,
        data: SDVariable,
        indices: SDVariable,
        updates: SDVariable,
        axis: Int
    ): SDVariable {
        val dataShape = data.shape ?: return data

        val flattened = sd.reshape("${opName}_dataFlat", data, -1)
        val updatesFlat = sd.reshape("${opName}_updatesFlat", updates, -1)
        val linearIndices = computeLinearIndices(sd, opName, indices, dataShape, axis)

        // Gather existing values, take min with updates, then scatter back
        val gathered = sd.gather("${opName}_gather", flattened, linearIndices, 0)
        val mined = sd.math.min("${opName}_min", gathered, updatesFlat)

        val linearIndicesExpanded = sd.expandDims("${opName}_linIdxExp", linearIndices, 1)
        val result = sd.scatterNdUpdate("${opName}_scatterNdUpdate", flattened, linearIndicesExpanded, mined)

        return sd.reshape("${opName}_output", result, *dataShape)
    }

    /**
     * Compute linear indices from element-wise indices along axis.
     */
    private fun computeLinearIndices(
        sd: SameDiff,
        opName: String,
        indices: SDVariable,
        dataShape: LongArray?,
        axis: Int
    ): SDVariable {
        if (dataShape == null) return sd.reshape("${opName}_flatIdx", indices, -1)

        val rank = dataShape.size
        val indicesShape = indices.shape ?: return sd.reshape("${opName}_flatIdx", indices, -1)

        // Compute strides for linear index calculation
        val strides = LongArray(rank)
        strides[rank - 1] = 1
        for (i in rank - 2 downTo 0) {
            strides[i] = strides[i + 1] * dataShape[i + 1]
        }

        // Build linear index: sum of coord[dim] * stride[dim] for all dims
        // coord[axis] comes from indices tensor, other coords come from position
        var linearIdx = sd.zerosLike("${opName}_linearInit", indices).castTo(DataType.INT64)

        for (dim in 0 until rank) {
            val stride = strides[dim]
            if (dim == axis) {
                // Use the indices tensor values directly
                val contribution = sd.math.mul("${opName}_contrib$dim",
                    indices.castTo(DataType.INT64), sd.constant(stride))
                linearIdx = sd.math.add("${opName}_linear$dim", linearIdx, contribution)
            } else {
                // Use position index for this dimension
                val size = indicesShape.getOrElse(dim) { 1L }
                val dimRange = sd.range("${opName}_range$dim",
                    sd.constant(0L), sd.constant(size), sd.constant(1L), DataType.INT64)

                // Reshape for broadcasting
                val shape = LongArray(rank) { 1L }
                shape[dim] = size
                val dimRangeReshaped = sd.reshape("${opName}_rangeReshape$dim", dimRange, *shape)

                val contribution = sd.math.mul("${opName}_contrib$dim",
                    dimRangeReshaped, sd.constant(stride))
                linearIdx = sd.math.add("${opName}_linear$dim", linearIdx, contribution)
            }
        }

        return sd.reshape("${opName}_linearFlat", linearIdx, -1)
    }
}
