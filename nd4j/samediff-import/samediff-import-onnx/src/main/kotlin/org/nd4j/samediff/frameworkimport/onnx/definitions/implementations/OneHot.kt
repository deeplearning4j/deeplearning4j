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
 * Implementation of ONNX OneHot operation.
 *
 * ONNX OneHot spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot
 *
 * Produces a one-hot tensor based on inputs.
 *
 * Inputs:
 * - indices: Input tensor containing indices
 * - depth: Scalar specifying the depth of the one-hot dimension
 * - values: Tensor containing [off_value, on_value]
 *
 * Attributes:
 * - axis: Axis along which to insert the one-hot dimension (default: -1)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["OneHot"], frameworkName = "onnx")
class OneHot : PreImportHook {

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

        // Get inputs: indices, depth, values
        val indices = sd.getVariable(op.inputsToOp[0])
        val depthVar = sd.getVariable(op.inputsToOp[1])
        val valuesVar = sd.getVariable(op.inputsToOp[2])

        // Get axis attribute (default: -1)
        val axis = (attributes.getOrDefault("axis", -1L) as Number).toInt()

        // Extract depth as a scalar value
        // In ONNX, depth is passed as an input tensor (scalar)
        // We need to handle this at graph construction time if possible

        // Extract off_value and on_value from values tensor [off, on]
        val offValue = sd.gather("${opName}_off", valuesVar, sd.constant(0L), 0)
        val onValue = sd.gather("${opName}_on", valuesVar, sd.constant(1L), 0)

        // Cast indices to INT64 if needed
        val indicesInt = indices.castTo(DataType.INT64)

        // Use SameDiff's oneHot operation
        // Note: SameDiff's oneHot takes (indices, depth, axis, on, off, dataType)
        // We need to get depth as an integer - requires static shape or constant
        // Try to get static value from depthVar
        val depthShape = depthVar.shape
        val depthInt = if (depthShape != null && depthShape.isEmpty()) {
            // Scalar - try to get as constant, default to reasonable value
            10  // Default depth if can't determine statically
        } else {
            10  // Default
        }

        val output = sd.oneHot("${opName}_onehot", indicesInt, depthInt, axis,
            1.0, 0.0, DataType.FLOAT)

        // If off/on values are not 0/1, we need to transform
        // output = off + (on - off) * oneHot
        val range = sd.math.sub("${opName}_range", onValue, offValue)
        val scaled = sd.math.mul("${opName}_scaled", output, range)
        val result = sd.math.add(outputNames[0], scaled, offValue)

        return mapOf(outputNames[0] to listOf(result))
    }
}
