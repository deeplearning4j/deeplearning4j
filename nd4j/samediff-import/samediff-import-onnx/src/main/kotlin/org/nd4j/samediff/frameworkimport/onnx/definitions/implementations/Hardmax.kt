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
 * Implementation of ONNX Hardmax operation.
 *
 * ONNX Hardmax spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax
 *
 * Hardmax computes the hardmax values for the given input:
 * The output is a one-hot tensor where the maximum value along the specified axis
 * is set to 1 and all other values are set to 0.
 *
 * Inputs:
 * - input: Input tensor
 *
 * Attributes:
 * - axis: Axis along which to compute hardmax (default: -1)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Hardmax"], frameworkName = "onnx")
class Hardmax : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])

        // Get axis attribute (default: -1, meaning last axis)
        val axis = (attributes.getOrDefault("axis", -1L) as Number).toInt()

        // Normalize negative axis
        val inputShapeArr = input.shape ?: throw IllegalStateException("Hardmax requires static input shape")
        val normalizedAxis = if (axis < 0) inputShapeArr.size + axis else axis
        val depth = inputShapeArr[normalizedAxis].toInt()

        // Compute argmax along the axis - keepDims=false gives us indices
        val argmaxResult = sd.argmax("${opName}_argmax", input, false, normalizedAxis.toLong())

        // Create one-hot encoding - argmax result is now rank-1, oneHot will add dimension back
        val oneHot = sd.oneHot("${opName}_onehot", argmaxResult.castTo(DataType.INT32),
            depth, normalizedAxis, 1.0, 0.0, input.dataType())

        val output = oneHot.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(output))
    }
}
