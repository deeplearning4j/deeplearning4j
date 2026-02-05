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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.custom.Triu
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Trilu operation using native libnd4j ops.
 *
 * Trilu returns the upper or lower triangular part of a 2D matrix,
 * or batch of matrices. If the attribute "upper" is set to true,
 * the upper triangular matrix is returned. If it is set to false,
 * the lower triangular matrix is returned.
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu
 *
 * Implementation:
 * - For upper=true: Uses native sd.linalg().triu(input, k) op
 * - For upper=false: Uses mask-based approach (no native tril op available)
 *
 * Inputs:
 * - input: Tensor of shape [*, M, N], where * is zero or more batch dimensions
 * - k: Optional diagonal offset (default 0)
 *
 * Attributes:
 * - upper: Boolean, if true return upper triangle, else lower (default: 1)
 *
 * Output:
 * - output: Tensor with same shape as input, with zeros in the excluded region
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Trilu"], frameworkName = "onnx")
class Trilu : PreImportHook {

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

        val input = sd.getVariable(op.inputsToOp[0])

        // Get diagonal offset k (default 0)
        val kValue = if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null && op.inputsToOp[1].isNotEmpty()) {
            getStaticIntValue(sd, dynamicVariables, op.inputsToOp[1]) ?: 0
        } else {
            0
        }

        // Get upper attribute (default true = 1)
        val upper = (attributes.getOrDefault("upper", 1L) as Number).toInt() != 0

        val output = if (upper) {
            // Use native triu op for upper triangular
            // sd.linalg().triu(input, diag) returns upper triangle with k diagonal offset
            sd.linalg().triu("${opName}_output", input, kValue)
        } else {
            // For lower triangle, use mask-based approach since there's no native tril op
            // Lower triangle: keep where col - row <= k (i.e., row >= col - k)
            computeLowerTriangle(sd, opName, input, kValue)
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    /**
     * Compute lower triangular matrix using mask-based approach.
     * Keeps elements where col - row <= k (equivalently, row >= col - k).
     */
    private fun computeLowerTriangle(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        k: Int
    ): SDVariable {
        // Get last two dimensions (M, N) using sizeAt with negative indices
        // These operations work dynamically without needing to know the rank at graph construction time
        val m = sd.sizeAt("${opName}_m", input, -2)
        val n = sd.sizeAt("${opName}_n", input, -1)

        // Create row and column indices
        val rows = sd.range("${opName}_rows", sd.constant(0L), m, sd.constant(1L), DataType.INT64)
        val cols = sd.range("${opName}_cols", sd.constant(0L), n, sd.constant(1L), DataType.INT64)

        // Reshape for broadcasting: rows [M, 1], cols [1, N]
        val rowsExpanded = sd.expandDims("${opName}_rowsExp", rows, 1)
        val colsExpanded = sd.expandDims("${opName}_colsExp", cols, 0)

        // Lower triangle mask: col - row <= k
        val diff = sd.math.sub("${opName}_diff", colsExpanded, rowsExpanded)
        val kConst = sd.constant("${opName}_k", k.toLong())
        val mask = sd.lte("${opName}_mask", diff, kConst)

        // Cast mask to input dtype and apply
        val maskCast = sd.castTo("${opName}_maskCast", mask, input.dataType())
        return sd.math.mul("${opName}_result", input, maskCast)
    }

    /**
     * Get static int value from dynamic variables if available.
     */
    private fun getStaticIntValue(
        sd: SameDiff,
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): Int? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.int64DataCount > 0 -> tensorProto.int64DataList[0].toInt()
            tensorProto.int32DataCount > 0 -> tensorProto.int32DataList[0]
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                if (tensorProto.dataType == Onnx.TensorProto.DataType.INT64_VALUE) {
                    buffer.asLongBuffer().get().toInt()
                } else {
                    buffer.asIntBuffer().get()
                }
            }
            else -> null
        }
    }
}
