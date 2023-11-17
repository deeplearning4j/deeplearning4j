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
import org.nd4j.autodiff.samediff.SameDiffNoArgSingleLambda
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of if.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/if.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["If"],frameworkName = "onnx")
class If : PreImportHook  {

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
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#non

        val registryCast = mappingRegistry as OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>
        val importGraphCast = importGraph as ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>
        val wrappedThenBranch = attributes["then_branch"] as OnnxIRGraph
        val wrappedElseBranch = attributes["else_branch"] as OnnxIRGraph
        val thenBranchSubGraph = importGraphCast.importGraph(
            wrappedThenBranch,
            null,
            null, mutableMapOf(),
            registryCast,
            false
        )

        sd.putSubFunction("${op.name}_then_branch",thenBranchSubGraph)
        val elseBranchSubGraph = importGraphCast.importGraph(
            wrappedElseBranch,
            null,
            null, mutableMapOf(),
            registryCast,
            false
        )
        sd.putSubFunction("${op.name}_else_branch",elseBranchSubGraph)

        val outputVarName = outputNames[0]

        val outputVar = sd.ifCond(outputVarName,outputVarName,SameDiffNoArgSingleLambda {
            sd.getVariable(op.inputsToOp[0])
        }, SameDiffNoArgSingleLambda {
            val definedFunction = sd.getFunction("${op.name}_then_branch")
            definedFunction.invokeGraphOn(sd)
        }, SameDiffNoArgSingleLambda {
            val definedFunction = sd.getFunction("${op.name}_else_branch")
            definedFunction.invokeGraphOn(sd)

        })

        return mapOf(outputVar.name() to listOf(outputVar))
    }


}