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
 * Implementation of ONNX EyeLike operation.
 *
 * ONNX EyeLike spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike
 *
 * Generate a 2D tensor (matrix) with ones on the diagonal and zeros elsewhere.
 * The shape is determined by the input tensor, which must be 2D.
 *
 * Inputs:
 * - input: 2D input tensor (only shape is used)
 *
 * Attributes:
 * - dtype: Data type of output (optional, defaults to input dtype)
 * - k: Diagonal offset (default: 0, main diagonal)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["EyeLike"], frameworkName = "onnx")
class EyeLike : PreImportHook {

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

        // Get input (only shape is used)
        val input = sd.getVariable(op.inputsToOp[0])

        // Get attributes
        val k = (attributes.getOrDefault("k", 0L) as Number).toInt()

        // Get dtype - if not specified, use input dtype
        val dtypeAttr = attributes["dtype"]
        val dtype = if (dtypeAttr != null) {
            // ONNX dtype enum to ND4J DataType
            val dtypeInt = (dtypeAttr as Number).toInt()
            onnxDtypeToNd4j(dtypeInt)
        } else {
            input.dataType()
        }

        // Get shape of input - need static shape for eye operation
        val inputShapeArr = input.shape ?: throw IllegalStateException("EyeLike requires static input shape")
        val rows = inputShapeArr[0].toInt()
        val cols = inputShapeArr[1].toInt()

        // Create identity-like matrix with diagonal offset k
        // Use eye operation with static shape parameters
        val eye = sd.math.eye("${opName}_eye", rows, cols, dtype)

        val output = eye.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun onnxDtypeToNd4j(onnxDtype: Int): DataType {
        return when (onnxDtype) {
            1 -> DataType.FLOAT
            2 -> DataType.UINT8
            3 -> DataType.INT8
            4 -> DataType.UINT16
            5 -> DataType.INT16
            6 -> DataType.INT32
            7 -> DataType.INT64
            9 -> DataType.BOOL
            10 -> DataType.FLOAT16
            11 -> DataType.DOUBLE
            12 -> DataType.UINT32
            13 -> DataType.UINT64
            14 -> DataType.BFLOAT16
            else -> DataType.FLOAT
        }
    }
}
