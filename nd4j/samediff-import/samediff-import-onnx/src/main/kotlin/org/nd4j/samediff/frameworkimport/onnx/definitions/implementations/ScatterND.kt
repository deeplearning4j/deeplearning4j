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
 * Implementation of ONNX ScatterND operation mapping for SameDiff.
 *
 * ONNX ScatterND spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND
 *
 * ScatterND takes three inputs: data, indices, updates, and copies data to an output tensor
 * then applies updates at the specified indices.
 *
 * Inputs:
 * - data: The original data tensor to be updated
 * - indices: Index tensor of shape [num_updates, index_depth]
 * - updates: Values to scatter at the specified indices
 *
 * Attributes:
 * - reduction: Type of reduction to apply (none, add, mul). Default is "none" (replace).
 *
 * Output:
 * - output: A copy of data with updates scattered at the indices
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ScatterND"], frameworkName = "onnx")
class ScatterND : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get inputs: data, indices, updates
        val data = sd.getVariable(op.inputsToOp[0])
        val indices = sd.getVariable(op.inputsToOp[1])
        val updates = sd.getVariable(op.inputsToOp[2])

        // Get reduction attribute (default is "none" which means replace)
        val reduction = attributes.getOrDefault("reduction", "none") as? String ?: "none"

        // Perform the scatter operation based on reduction type
        val output = when (reduction.lowercase()) {
            "none" -> {
                // Replace mode: use scatter_nd_update
                sd.scatterNdUpdate(outputNames[0], data, indices, updates)
            }
            "add" -> {
                // Add mode: use scatter_nd_add
                sd.scatterNdAdd(outputNames[0], data, indices, updates)
            }
            "mul" -> {
                // Multiply mode: use scatter_nd_mul or implement manually
                // If scatter_nd_mul doesn't exist, we implement it as:
                // output = data * scatter_nd(ones_like(data), indices, updates)
                // where scatter_nd places updates at indices on top of ones
                // Actually for mul, we need: at each index, output[i] = data[i] * updates[i]
                // This is more complex - gather, multiply, then scatter back
                val gathered = sd.gatherNd("${op.name}_gathered", data, indices)
                val multiplied = sd.math.mul("${op.name}_multiplied", gathered, updates)
                sd.scatterNdUpdate(outputNames[0], data, indices, multiplied)
            }
            "max" -> {
                // Max mode: keep the maximum of data and updates at each index
                val gathered = sd.gatherNd("${op.name}_gathered", data, indices)
                val maxed = sd.math.max("${op.name}_maxed", gathered, updates)
                sd.scatterNdUpdate(outputNames[0], data, indices, maxed)
            }
            "min" -> {
                // Min mode: keep the minimum of data and updates at each index
                val gathered = sd.gatherNd("${op.name}_gathered", data, indices)
                val minned = sd.math.min("${op.name}_minned", gathered, updates)
                sd.scatterNdUpdate(outputNames[0], data, indices, minned)
            }
            else -> {
                throw IllegalArgumentException("Unsupported reduction mode for ScatterND: $reduction. " +
                        "Supported modes: none, add, mul, max, min")
            }
        }

        return mapOf(outputNames[0] to listOf(output))
    }
}
