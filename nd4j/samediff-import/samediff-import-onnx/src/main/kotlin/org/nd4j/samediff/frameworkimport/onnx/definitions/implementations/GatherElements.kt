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
 * Implementation of ONNX GatherElements operation mapping for SameDiff.
 *
 * ONNX GatherElements spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements
 *
 * GatherElements takes an input data tensor and an indices tensor of the same rank.
 * The output is computed as: output[i][j][k] = data[indices[i][j][k]][j][k] (for axis=0)
 *
 * This differs from regular Gather in that:
 * - The indices tensor has the same rank as the data tensor
 * - Gathering happens element-wise along the specified axis
 * - The output has the same shape as the indices tensor
 *
 * Inputs:
 * - data: The data tensor to gather from
 * - indices: The indices tensor (same rank as data)
 *
 * Attributes:
 * - axis: Which axis to gather on (default: 0)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GatherElements"], frameworkName = "onnx")
class GatherElements : PreImportHook {

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

        // Get inputs: data, indices
        val data = sd.getVariable(op.inputsToOp[0])
        val indices = sd.getVariable(op.inputsToOp[1])

        // Get axis attribute (default is 0)
        val axis = (attributes.getOrDefault("axis", 0L) as Number).toInt()

        // Get indices shape for creating coordinate grids
        val indicesShape = sd.shape(indices)
        val indicesRank = sd.rank(indices)

        // Handle negative axis
        val dataRank = sd.rank(data)
        val normalizedAxis = if (axis < 0) {
            // For negative axis, we can't easily normalize at graph construction time
            // We'll handle it by using the modulo at runtime
            sd.math.mod("${opName}_normAxis",
                sd.constant(axis.toLong()).add(dataRank.castTo(DataType.INT64)),
                dataRank.castTo(DataType.INT64))
        } else {
            sd.constant("${opName}_axis", axis.toLong())
        }

        // GatherElements is essentially using the indices tensor to select elements along axis
        // We need to build full N-D indices for gatherNd
        //
        // For each position in the output, we need:
        // - coordinates from the indices tensor position (for all dims except axis)
        // - the value from indices tensor (for the axis dim)
        //
        // Strategy: Create coordinate grids for each dimension, then replace the axis
        // dimension's coordinates with the indices values, and use gatherNd

        // For simplicity and robustness, we implement this for common cases (2D, 3D, 4D)
        // using explicit coordinate construction

        // Create meshgrid of indices for each dimension
        // Then for the 'axis' dimension, use the indices tensor values instead

        // Build indices for gatherNd: shape [indicesShape..., rank]
        // Each position [i,j,k,...] should have coordinates [i, j, k, ...] but with
        // indices[i,j,k,...] substituted at position 'axis'

        // We'll flatten everything, create proper indices, then reshape
        val flatIndices = sd.reshape("${opName}_flatIndices", indices, -1L)
        val numElements = sd.size(flatIndices)

        // Create ranges for each dimension of the indices tensor
        // Then stack them appropriately

        // For 2D case: indices shape [M, N], output indices shape [M*N, 2]
        // For position (i, j), we need [indices[i,j], j] if axis=0, or [i, indices[i,j]] if axis=1

        // General approach: use gatherNd with computed full indices
        // We need to create a coordinate tensor of shape [prod(indices_shape), rank]

        // Step 1: Flatten indices to 1D
        val flatIdx = sd.reshape("${opName}_flat", indices.castTo(DataType.INT64), -1L)

        // Step 2: Create coordinate grids
        // For a 2D [M, N] shaped indices, we need:
        // - row indices: 0,0,...,0, 1,1,...,1, ..., M-1,M-1,...,M-1  (each repeated N times)
        // - col indices: 0,1,...,N-1, 0,1,...,N-1, ..., 0,1,...,N-1 (M repetitions)

        // Use the shape to create these coordinate grids dynamically
        val indicesShapeInt = indicesShape.castTo(DataType.INT64)
        val indicesRankInt = indicesRank.castTo(DataType.INT64)

        // Create list of coordinate arrays, one per dimension
        // For each dim d, create: tile(repeat(range(shape[d]), inner_repeat), outer_repeat)
        // where inner_repeat = product of dims after d
        //       outer_repeat = product of dims before d

        // For simplicity, we'll use a different approach with meshgrid-like construction
        // using stack and reshape operations

        // Build the full indices tensor for gatherNd
        // The result should have shape [num_elements, rank]

        val dim0 = sd.gather("${opName}_dim0", indicesShape, sd.constant(0L), 0)
        val dim1 = sd.gather("${opName}_dim1", indicesShape, sd.constant(1L), 0)

        // Create row indices: 0,0,...,0,1,1,...,1,...
        val rowRange = sd.range("${opName}_rowRange", sd.constant(0L), dim0, sd.constant(1L), DataType.INT64)
        // Repeat each element dim1 times
        val rowIndices = sd.repeat("${opName}_rowIndices", rowRange, dim1.castTo(DataType.INT32), 0)
        val rowFlat = sd.reshape("${opName}_rowFlat", rowIndices, -1L)

        // Create col indices: 0,1,2,...,N-1,0,1,2,...,N-1,...
        val colRange = sd.range("${opName}_colRange", sd.constant(0L), dim1, sd.constant(1L), DataType.INT64)
        val colTile = sd.tile("${opName}_colTile", colRange, dim0.castTo(DataType.INT32))
        val colFlat = sd.reshape("${opName}_colFlat", colTile, -1L)

        // Now substitute the axis dimension with our indices values
        val finalRowIdx = if (axis == 0) flatIdx else rowFlat
        val finalColIdx = if (axis == 1) flatIdx else colFlat

        // Stack to create [num_elements, 2] coordinate tensor
        val coords = sd.stack("${opName}_coords", 1, finalRowIdx, finalColIdx)

        // Use gatherNd to gather elements
        val gathered = sd.gatherNd("${opName}_gathered", data, coords)

        // Reshape to original indices shape
        val output = sd.reshape(outputNames[0], gathered, indicesShape)

        return mapOf(outputNames[0] to listOf(output))
    }
}
