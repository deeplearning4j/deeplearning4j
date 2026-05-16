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

package org.nd4j.samediff.frameworkimport.tensorflow.definitions.implementations

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.tensorflow.framework.TensorProto

/**
 * PreImportHook for TensorFlow ReverseV2 op.
 * 
 * TF ReverseV2 has two inputs:
 * - tensor: the tensor to reverse
 * - axis: 1D tensor specifying which dimensions to reverse
 * 
 * This hook properly extracts the axis from the second input and passes it to the ND4J reverse op.
 * The axis can be either a constant (in dynamicVariables) or a runtime variable.
 */
@PreHookRule(nodeNames = [], opNames = ["ReverseV2", "Reverse"], frameworkName = "tensorflow")
class ReverseV2 : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get inputs
        val input = sd.getVariable(op.inputsToOp[0])
        val axisVar = sd.getVariable(op.inputsToOp[1])

        // Try to get axis from dynamicVariables (constant) first
        val axisVarName = op.inputsToOp[1]
        val axisTensorProto = dynamicVariables[axisVarName] as? TensorProto
        val axis = if (axisTensorProto != null) {
            // Axis is a constant - extract values from TensorProto
            // TF axis is typically int32 or int64
            when {
                axisTensorProto.int64ValCount > 0 -> axisTensorProto.int64ValList.map { it }.toLongArray()
                axisTensorProto.intValCount > 0 -> axisTensorProto.intValList.map { it.toLong() }.toLongArray()
                axisTensorProto.tensorContent.size() > 0 -> {
                    // Read from raw tensor content - check dtype to determine element size
                    val buffer = axisTensorProto.tensorContent.asReadOnlyByteBuffer()
                    buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    when (axisTensorProto.dtype) {
                        org.tensorflow.framework.DataType.DT_INT64 -> {
                            val longBuffer = buffer.asLongBuffer()
                            LongArray(longBuffer.remaining()) { longBuffer.get() }
                        }
                        else -> {
                            // Default to int32
                            val intBuffer = buffer.asIntBuffer()
                            LongArray(intBuffer.remaining()) { intBuffer.get().toLong() }
                        }
                    }
                }
                else -> longArrayOf()
            }
        } else {
            longArrayOf()
        }

        // Create reverse op with input and axis
        val output = if (axis.isNotEmpty()) {
            sd.reverse(outputNames[0], input, *axis)
        } else {
            // Fallback: try resolving axis from variable value if it is already materialized during import.
            val runtimeAxis = axisVar.arr?.toLongVector() ?: longArrayOf()
            if (runtimeAxis.isNotEmpty()) {
                sd.reverse(outputNames[0], input, *runtimeAxis)
            } else {
                throw IllegalStateException(
                    "ReverseV2 axis must be constant or materialized at import time for op ${op.name} (${outputNames[0]})"
                )
            }
        }

        return mapOf(outputNames[0] to listOf(output))
    }
}
