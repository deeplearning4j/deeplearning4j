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
 * Custom import hook for ONNX GatherND operation.
 *
 * ONNX GatherND spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
 *
 * GatherND gathers slices from data tensor using indices.
 * The indices tensor has shape [i_0, ..., i_{r-1}, Q] where Q is the number of dimensions to index.
 * Output has shape [i_0, ..., i_{r-1}, j_{Q+1}, ..., j_{m-1}] where m is data rank.
 *
 * For the common case where Q=1 (indices are 1D coordinates), this is equivalent to
 * gathering along axis 0 using the flattened indices.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GatherND"], frameworkName = "onnx")
class GatherND : PreImportHook {

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

        // Get batch_dims attribute (default is 0)
        val batchDims = (attributes.getOrDefault("batch_dims", 0L) as Number).toInt()

        // If shapes are not statically known, fall back to native gatherNd
        // which handles dynamic shapes at execution time.
        val indicesShape = indices.shape
        val dataShape = data.shape

        if (indicesShape == null || dataShape == null) {
            val result = sd.gatherNd(outputNames[0], data, indices)
            return mapOf(outputNames[0] to listOf(result))
        }

        val Q = indicesShape.last().toInt()

        val result = if (Q == 1 && batchDims == 0) {
            // Simple case: 1D indices, no batch dims
            // indices shape: [N, 1], squeeze to [N], gather along axis 0
            val indicesSqueezed = sd.squeeze("${opName}_idx", indices, 1)
            sd.gather(outputNames[0], data, indicesSqueezed.castTo(DataType.INT32), 0)
        } else if (batchDims > 0) {
            // Batch mode: first batch_dims dimensions are batch dimensions
            sd.gatherNd(outputNames[0], data, indices)
        } else {
            // Q > 1: Multi-dimensional indices
            val strides = computeStrides(sd, data, Q)
            val linearIndices = computeLinearIndices(sd, opName, indices, strides)
            val gathered = sd.gather("${opName}_gathered", data, linearIndices.castTo(DataType.INT32), 0)
            gathered.rename(outputNames[0])
        }

        return mapOf(outputNames[0] to listOf(result))
    }

    private fun computeStrides(sd: SameDiff, data: SDVariable, Q: Int): List<SDVariable> {
        val dataShape = data.shape ?: throw IllegalStateException("GatherND requires static input shape")
        val strides = mutableListOf<SDVariable>()
        var stride = 1
        for (i in 0 until Q) {
            strides.add(sd.constant("${data.name()}_stride$i", stride.toLong()))
            stride *= dataShape[i].toInt()
        }
        return strides
    }

    private fun computeLinearIndices(sd: SameDiff, opName: String, indices: SDVariable, strides: List<SDVariable>): SDVariable {
        val Q = strides.size
        var result: SDVariable? = null
        for (i in 0 until Q) {
            val idxI = sd.gather("${opName}_idx$i", indices, sd.constant(i.toLong()), -1)
            val weighted = idxI.mul(strides[i])
            result = if (result == null) weighted else result.add(weighted)
        }
        return result ?: throw IllegalStateException("Q must be > 0")
    }
}
