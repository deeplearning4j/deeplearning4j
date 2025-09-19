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
 * Implementation of ONNX Gather operation mapping for SameDiff.
 * Maps the ONNX Gather operation to SameDiff's gather function.
 * 
 * ONNX Gather spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather
 * Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis dimension of data
 * (axis specified as an attribute) indexed by indices, and concatenates them in an output tensor of rank q + (r - 1).
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Gather"], frameworkName = "onnx")
class Gather : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get input variables
        val dataVariable = sd.getVariable(op.inputsToOp[0])
        val indicesVariable = sd.getVariable(op.inputsToOp[1])
        
        // Get axis attribute (default to 0 if not specified)
        val axis = attributes["axis"]?.let { (it as Long).toInt() } ?: 0
        
        // Call SameDiff's gather method
        val outputVarName = outputNames[0]
        val outputVar = sd.gather(outputVarName, dataVariable, indicesVariable, axis)
        
        return mapOf(outputVar.name() to listOf(outputVar))
    }
}
