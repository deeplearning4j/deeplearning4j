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
 * Implementation of ONNX Equal operation with type casting support.
 *
 * The Equal operation compares two input tensors element-wise for equality.
 * This implementation automatically casts inputs to a common type before comparison
 * based on data type width, promoting to the wider type.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Equal"], frameworkName = "onnx")
class Equal : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input1 = sd.getVariable(op.inputsToOp[0])
        val input2 = sd.getVariable(op.inputsToOp[1])

        // Determine common data type for casting
        val commonDataType = determineCommonDataType(input1.dataType(), input2.dataType())

        // Cast inputs to common type if needed
        val castedInput1 = if (input1.dataType() != commonDataType) {
            input1.castTo(commonDataType)
        } else {
            input1
        }

        val castedInput2 = if (input2.dataType() != commonDataType) {
            input2.castTo(commonDataType)
        } else {
            input2
        }

        // Perform equal comparison
        val result = castedInput1.eq(castedInput2).castTo(DataType.BOOL).rename(outputNames[0])


        return mapOf(outputNames[0] to listOf(result))
    }

    private fun determineCommonDataType(type1: DataType, type2: DataType): DataType {
        if (type1 == type2) return type1

        return if (type1.width() >= type2.width()) type1 else type2
    }
}