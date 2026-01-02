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
 * Implementation of ONNX ReduceLogSum operation.
 *
 * ONNX ReduceLogSum spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum
 *
 * Computes the log of the sum of the input tensor's elements along the specified axes.
 * output = log(sum(input))
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
 * @author Eclipse Deeplearning4j Development Team
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

        // Get input
        val data = sd.getVariable(op.inputsToOp[0])

        // Get keepdims attribute (default: true/1)
        val keepDims = (attributes.getOrDefault("keepdims", 1L) as Number).toInt() != 0

        // Get axes - could be from attribute (old opset) or input (new opset)
        val axesAttr = attributes["axes"]
        val axes: IntArray = when {
            op.inputsToOp.size > 1 -> {
                // Axes from input tensor (opset 18+)
                // We'll handle this dynamically - for now assume it's available as constant
                // This is a simplification; full implementation would need dynamic handling
                intArrayOf()
            }
            axesAttr != null -> {
                // Axes from attribute
                when (axesAttr) {
                    is List<*> -> axesAttr.map { (it as Number).toInt() }.toIntArray()
                    is LongArray -> axesAttr.map { it.toInt() }.toIntArray()
                    else -> intArrayOf()
                }
            }
            else -> intArrayOf() // Reduce all axes
        }

        // Compute sum along axes
        val sumResult = if (axes.isEmpty()) {
            // Reduce all dimensions
            sd.sum("${opName}_sum", data, keepDims)
        } else {
            val axesLong = axes.map { it.toLong() }.toLongArray()
            sd.sum("${opName}_sum", data, keepDims, *axesLong)
        }

        // Compute log of sum
        val output = sd.math.log(outputNames[0], sumResult)

        return mapOf(outputNames[0] to listOf(output))
    }
}
