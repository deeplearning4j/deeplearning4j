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
import org.nd4j.autodiff.samediff.SameDiffLambda
import org.nd4j.autodiff.samediff.SameDiffNoArgSingleLambda
import org.nd4j.autodiff.samediff.SameDiffSingleLambda
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException

/**
 * A port of if.py from onnx tensorflow for samediff:
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
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#non

        val registryCast = mappingRegistry as OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>
        val importGraphCast = importGraph as ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>
        val importedBody = attributes["body"] as OnnxIRGraph
        val body = importGraphCast.importGraph(
            importedBody,
            null,
            null, mutableMapOf(),
            registryCast)
        body.isEagerMode = false
        sd.putSubFunction("${op.name}_loop_body",body)
        sd.isEagerMode = false
        val inputTensors = ArrayList<SDVariable>()
        val cond: SDVariable? = if(op.inputsToOp.size > 1 && op.inputsToOp[1] != "") sd.getVariable(op.inputsToOp[1]) else null
        val condBody: SameDiffSingleLambda? = if(cond != null) {
            SameDiffSingleLambda { sameDiff, inputs ->
                inputs[0].castTo(DataType.BOOL)
            }
        } else {
            null
        }
        for(i in 2 until op.inputsToOp.size) {
            inputTensors.add(sd.getVariable(op.inputsToOp[i]))
        }


        val terminationIterations: SDVariable? = if(op.inputsToOp.size > 0 && op.inputsToOp[0] != "") sd.getVariable(op.inputsToOp[0]) else null
        //  for loop:  if M is not None and cond_init is None
        if(terminationIterations != null && cond == null) {
            val condBody =  SameDiffSingleLambda { sameDiff, inputs ->
                inputs[0].lt(inputs[1]).castTo(DataType.BOOL)
            }

            //ensure first variable is loop termination variable with body
            //being a variable update + the intended body
            val loopVars = ArrayList<SDVariable>()
            loopVars.add(terminationIterations)
            loopVars.addAll(inputTensors)

            val ret = sd.whileLoop(loopVars.toTypedArray(),
                condBody
            ) { sameDiff, inputs ->
               arrayOf(inputs[0].add(1.0))
            }


            if(ret.size != outputNames.size)
                throw IllegalArgumentException("Unable to set name variable s${outputNames}, output variable names was size ${ret.size}, specified names was size ${outputNames.size}")

            ret.forEachIndexed { index, sdVariable ->
                sdVariable.rename(outputNames[index])
            }

            return ret.associate{ input -> input.name() to listOf(input) }

        } else if(terminationIterations == null && cond != null) {
            // # while and do-while loop
            val ret = sd.whileLoop(inputTensors.toTypedArray(),
                condBody!!) { sameDiff, inputs ->
                arrayOf(sameDiff.invokeGraphOn(sd))
            }


            if(ret.size != outputNames.size)
                throw IllegalArgumentException("Unable to set name variable s${outputNames}, output variable names was size ${ret.size}, specified names was size ${outputNames.size}")

            ret.forEachIndexed { index, sdVariable ->
                sdVariable.rename(outputNames[index])
            }

            return ret.associate{ input -> input.name() to listOf(input) }
        } else if(cond != null && terminationIterations != null) {
            // # combine for loop and while loop together
            val forCondBody: SameDiffSingleLambda? = if(cond != null) {
                SameDiffSingleLambda { sameDiff, inputs ->
                    sd.bitwise().and(inputs[0].castTo(DataType.INT64),cond.castTo(DataType.INT64)).castTo(DataType.BOOL)
                }
            } else {
                null
            }
            val ret = sd.whileLoop(inputTensors.toTypedArray(),
                forCondBody!!
            ) { sameDiff, inputs ->
                arrayOf(inputs[0].add(1.0))
            }

            if(ret.size != outputNames.size)
                throw IllegalArgumentException("Unable to set name variable s${outputNames}, output variable names was size ${ret.size}, specified names was size ${outputNames.size}")

            ret.forEachIndexed { index, sdVariable ->
                sdVariable.rename(outputNames[index])
            }


            return ret.associate{ input -> input.name() to listOf(input) }

        } else {
            //both are null
            throw IllegalArgumentException("Unable to support infinite loops")
        }
    }
}