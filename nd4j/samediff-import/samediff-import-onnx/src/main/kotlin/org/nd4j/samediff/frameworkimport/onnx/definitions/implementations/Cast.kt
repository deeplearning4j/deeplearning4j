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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Custom handler for ONNX Cast operation.
 *
 * This handler works around a CUDA backend limitation where casting from BOOL
 * to INT64/UINT64 fails. When the source is BOOL and target is an integer type,
 * we cast through INT8 first to avoid the unsupported kernel.
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
        val inputDtype = input.dataType()

        // Get target type from 'to' attribute
        val toOnnxType = (attributes["to"] as? Number)?.toInt() ?: ONNX_FLOAT
        val targetDtype = onnxTypeToNd4j(toOnnxType)

        // Check if this is a BOOL cast that needs workaround
        // CUDA backend doesn't support direct BOOL -> any other type using Cast/assign kernel
        // Note: inputDtype may be null during import for dynamically shaped inputs
        val needsWorkaround = (inputDtype == DataType.BOOL || inputDtype == null) && targetDtype != DataType.BOOL

        val result = if (needsWorkaround && inputDtype == DataType.BOOL) {
            // For BOOL inputs, use math operations to avoid unsupported Cast kernel
            // Multiply by 1 to convert: bool * 1 = 0 or 1 as numeric
            // First create a constant of the target type with value 1
            val multiplier = sd.constant(Nd4j.scalar(targetDtype, 1.0))
            // Broadcasting mul with BOOL should work and produce target type
            val converted = sd.math.mul("${outputNames[0]}_bool_convert", input, multiplier)
            // If we still need to cast (shouldn't be needed but just in case)
            if (converted.dataType() != targetDtype) {
                sd.castTo(outputNames[0], converted, targetDtype)
            } else {
                converted.rename(outputNames[0])
                converted
            }
        } else {
            // Direct cast for non-BOOL inputs
            sd.castTo(outputNames[0], input, targetDtype)
        }

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
