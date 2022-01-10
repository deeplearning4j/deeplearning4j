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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
  Implements reshape
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Reshape"],frameworkName = "onnx")
class Reshape : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        //older attributes based shape
        if(attributes.isNotEmpty()) {
            val newShape = attributes["shape"] as List<Int>
            val shapeArr = newShape.toIntArray().map { input -> input.toLong() }.toLongArray()
            val finalOutput = sd.reshape(outputNames[0],inputVariable,*shapeArr)
            return mapOf(outputNames[0] to listOf(finalOutput))
        } else {
            val shapeVar = sd.getVariable(op.inputsToOp[1])
            val finalOutput = sd.reshape(outputNames[0],inputVariable,shapeVar)
            return mapOf(outputNames[0] to listOf(finalOutput))
        }

    }
}