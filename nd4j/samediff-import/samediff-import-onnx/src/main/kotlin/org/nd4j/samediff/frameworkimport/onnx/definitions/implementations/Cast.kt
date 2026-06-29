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
 * Custom handler for ONNX Cast operation.
 *
 * This handler preserves ONNX Cast as SameDiff's dtype Cast op. Cast accepts
 * NDARRAY inputs, including BOOL tensors produced by comparison ops.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Cast"], frameworkName = "onnx")
class Cast : PreImportHook {

    // ONNX tensor type enum values
    companion object {
        const val ONNX_FLOAT = 1
        const val ONNX_UINT8 = 2
        const val ONNX_INT8 = 3
        const val ONNX_UINT16 = 4
        const val ONNX_INT16 = 5
        const val ONNX_INT32 = 6
        const val ONNX_INT64 = 7
        const val ONNX_STRING = 8
        const val ONNX_BOOL = 9
        const val ONNX_FLOAT16 = 10
        const val ONNX_DOUBLE = 11
        const val ONNX_UINT32 = 12
        const val ONNX_UINT64 = 13
        const val ONNX_BFLOAT16 = 16
    }

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input = sd.getVariable(op.inputsToOp[0])
        // Get target type from 'to' attribute
        val toOnnxType = (attributes["to"] as? Number)?.toInt() ?: ONNX_FLOAT
        val targetDtype = onnxTypeToNd4j(toOnnxType)

        val result = sd.castTo(outputNames[0], input, targetDtype)

        return mapOf(outputNames[0] to listOf(result))
    }

    private fun onnxTypeToNd4j(onnxType: Int): DataType {
        return when (onnxType) {
            ONNX_FLOAT -> DataType.FLOAT
            ONNX_UINT8 -> DataType.UINT8
            ONNX_INT8 -> DataType.INT8
            ONNX_UINT16 -> DataType.UINT16
            ONNX_INT16 -> DataType.INT16
            ONNX_INT32 -> DataType.INT32
            ONNX_INT64 -> DataType.INT64
            ONNX_BOOL -> DataType.BOOL
            ONNX_FLOAT16 -> DataType.FLOAT16
            ONNX_DOUBLE -> DataType.DOUBLE
            ONNX_UINT32 -> DataType.UINT32
            ONNX_UINT64 -> DataType.UINT64
            ONNX_BFLOAT16 -> DataType.BFLOAT16
            else -> DataType.FLOAT
        }
    }
}
