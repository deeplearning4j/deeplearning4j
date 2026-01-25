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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

@PreHookRule(nodeNames = [],opNames = ["Unsqueeze"],frameworkName = "onnx")
class Unsqueeze  : PreImportHook {
    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#unsqueeze
        val axes = if(op.inputsToOp.size < 2) {
            @Suppress("UNCHECKED_CAST")
            attributes["axes"] as List<Int>
        } else {
            // Get axes from dynamicVariables (ONNX TensorProto)
            val axesVarName = op.inputsToOp[1]
            getAxesFromTensorProto(dynamicVariables, axesVarName) ?: listOf(0)
        }
        var ret: SDVariable? = null

        val input = sd.getVariable(op.inputsToOp[0])

        if(axes.size != 1) {
            for(i in axes.indices) {
                if(i < axes.size - 1)
                    ret = sd.expandDims(outputNames[0],input,axes[i])
                else {
                    ret = sd.expandDims(outputNames[0],input,axes[i])
                }
            }
        } else {
            ret = sd.expandDims(outputNames[0],input,axes[0])
        }

        return mapOf(ret!!.name() to listOf(ret!!))
    }

    /**
     * Extract axes values from ONNX TensorProto in dynamicVariables.
     */
    private fun getAxesFromTensorProto(
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): List<Int>? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.int64DataCount > 0 -> tensorProto.int64DataList.map { it.toInt() }
            tensorProto.int32DataCount > 0 -> tensorProto.int32DataList.toList()
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val result = mutableListOf<Int>()
                if (tensorProto.dataType == Onnx.TensorProto.DataType.INT64_VALUE) {
                    val longBuffer = buffer.asLongBuffer()
                    while (longBuffer.hasRemaining()) {
                        result.add(longBuffer.get().toInt())
                    }
                } else {
                    val intBuffer = buffer.asIntBuffer()
                    while (intBuffer.hasRemaining()) {
                        result.add(intBuffer.get())
                    }
                }
                result
            }
            else -> null
        }
    }
}