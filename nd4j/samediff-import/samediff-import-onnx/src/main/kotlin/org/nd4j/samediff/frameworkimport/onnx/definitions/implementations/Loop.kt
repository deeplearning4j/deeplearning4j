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
import org.nd4j.autodiff.samediff.ControlFlow
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.SameDiffLambda
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.definitions.OnnxInputTensors
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum


/**
 * A port of loop.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/loop.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Loop"],frameworkName = "onnx")
class Loop : PreImportHook  {

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

        val funcName = "${op.name}"
        val importedBody = attributes["body"] as OnnxIRGraph
        val body = importAndValidateGraph(
            importedBody = importedBody,
            dynamicVariables = dynamicVariables,sd = sd,
            funcName = funcName,mappingRegistry = mappingRegistry,
            importGraph = importGraph
        )


        val tensors = OnnxInputTensors(importedBody = importedBody,op = op,sd = sd)
        val inputTensors = tensors.toInputTensors()
        val outputs = ArrayList<String>()
        outputs.addAll(inputTensors.map { input -> input.name() })

        //  combine for loop and while loop together
        val forCondBody = ControlFlow.condBody()

        /**
         * inputs here:
         * condition
         * trip_count
         * curr_iteration
         * seq_empty
         */
        val loopBody  = loopBody(
            importedBody = importedBody,
            funcName = funcName,
            parent =  sd, funcBody = body)

        val outputNames = inputTensors.map { input -> "${funcName}_${input.name()}_output" }.toMutableList()
        for(i in 0 until op.outputsOfOp.size) {
            outputNames[outputNames.size - i - 1] = op.outputsOfOp[i]
        }

        val loopRet = sd.whileLoop(
            outputNames.toTypedArray(),
            funcName,
            inputTensors.toTypedArray(),
            forCondBody,loopBody)





        val ret2 = loopRet.associate { input -> input.name() to listOf(input) }.toMutableMap()
        return ret2

    }



    fun loopBody(importedBody: OnnxIRGraph,parent: SameDiff,funcBody: SameDiff,funcName: String): SameDiffLambda {
        return ControlFlow.loopBody(parent,
            funcBody,
            funcName,
            importedBody.inputList.toTypedArray(),
            importedBody.outputList.toTypedArray())
    }


    fun importAndValidateGraph(importedBody: OnnxIRGraph,
                               dynamicVariables: Map<String, GeneratedMessageV3>,
                               sd: SameDiff,
                               funcName: String,
                               mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
                               importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
    ): SameDiff {
        val registryCast = mappingRegistry as OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>
        val importGraphCast = importGraph as ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>

        val body = importGraphCast.importGraph(
            importedBody,
            null,
            null, dynamicVariables as MutableMap<String,Onnx.TensorProto>,
            registryCast)
        body.isEagerMode = false
        sd.putSubFunction(funcName,body)
        sd.isEagerMode = false

        //only validate if present
        //all graphs must have iteration number, condition (2 + N inputs) and extra deps
        val iterCountVar = importedBody.inputList[0]
        val iterCountVarImported = body.getVariable(iterCountVar)
        if(!iterCountVarImported.dataType().isNumerical) {
            throw IllegalArgumentException("Attribute trip count on graph is invalid data type. Must be numerical.")
        }


        val condInVar = importedBody.inputList[1]
        val condInVarImported = body.getVariable(condInVar)
        if(condInVarImported.dataType() != DataType.BOOL) {
            throw IllegalArgumentException("Attribute cond on graph is invalid data type. Must be boolean.")
        }

        return body
    }

}