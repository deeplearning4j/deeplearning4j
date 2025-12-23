package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

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
 * Implementation of ONNX Where operation with type casting support.
 *
 * The Where operation has two modes:
 * 1. Single input (condition): Returns coordinates of true elements
 * 2. Three inputs (condition, x, y): Returns x where condition is true, y where false
 *
 * This implementation automatically casts x and y inputs to a common type before selection
 * based on data type width, promoting to the wider type.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Where"], frameworkName = "onnx")
class Where : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val result = when (op.inputsToOp.size) {
            1 -> {
                // Single input mode - return coordinates of true elements
                val condition = sd.getVariable(op.inputsToOp[0])
                sd.where(condition)
            }
            3 -> {
                // Three input mode - conditional selection with type casting
                val condition = sd.getVariable(op.inputsToOp[0])
                val x = sd.getVariable(op.inputsToOp[1])
                val y = sd.getVariable(op.inputsToOp[2])

                val conditionCasted = condition.castTo(DataType.BOOL)
                // Cast x and y to common type if needed
                val castedX = x.castTo(DataType.INT64)

                val castedY = y.castTo(DataType.INT64)

                // Perform conditional selection
                sd.where(castedX, castedY,conditionCasted)
            }
            else -> {
                throw IllegalArgumentException("Where operation requires 1 or 3 inputs, got ${op.inputsToOp.size}")
            }
        }

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }

    private fun determineCommonDataType(type1: DataType, type2: DataType): DataType {
        if (type1 == type2) return type1

        return if (type1.width() >= type2.width()) type1 else type2
    }
}