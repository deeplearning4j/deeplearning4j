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

/**
 * A port of cast.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/cast.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["CumSum"],frameworkName = "onnx")
class CumSum : PreImportHook  {

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
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#cumsum

        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val exclusive = attributes.getOrDefault("exclusive",false) as Boolean
        val reverse = attributes.getOrDefault("reverse",false) as Boolean

        // Get axis value from dynamicVariables (ONNX TensorProto)
        val axisVarName = op.inputsToOp[1]
        val axisValue = getLongFromTensorProto(dynamicVariables, axisVarName) ?: 0L

        val outputVar = sd.cumsum(outputNames[0],inputVariable,exclusive,reverse,axisValue)
        return mapOf(outputVar.name() to listOf(outputVar))
    }

    /**
     * Extract long value from ONNX TensorProto in dynamicVariables.
     */
    private fun getLongFromTensorProto(
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): Long? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.int64DataCount > 0 -> tensorProto.int64DataList[0]
            tensorProto.int32DataCount > 0 -> tensorProto.int32DataList[0].toLong()
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                when (tensorProto.dataType) {
                    Onnx.TensorProto.DataType.INT64_VALUE -> buffer.asLongBuffer().get()
                    Onnx.TensorProto.DataType.INT32_VALUE -> buffer.asIntBuffer().get().toLong()
                    else -> null
                }
            }
            else -> null
        }
    }
}