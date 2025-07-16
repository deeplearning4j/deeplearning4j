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
import org.nd4j.linalg.factory.Nd4j
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
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop

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

        // ONNX Loop inputs:
        // 0: max_trip_count (may be empty scalar for infinite)
        // 1: termination_condition (boolean)
        // 2+: loop variables

        // We need to transform these into the format expected by ControlFlow.condBody()
        // which expects: [current_iteration, max_iterations, condition, ...extra_args]

        val maxTripCount = inputTensors[0]
        val terminationCondition = inputTensors[1]
        val loopVariables = if (inputTensors.size > 2) inputTensors.drop(2) else emptyList()

        // Create initial iteration counter
        val initialIteration = sd.constant("${funcName}_iter_start", 0)

        // Create the loop variables array in the expected format
        val loopVars = mutableListOf<SDVariable>()
        loopVars.add(initialIteration)      // current iteration
        loopVars.add(maxTripCount)         // max iterations
        loopVars.add(terminationCondition) // condition
        loopVars.addAll(loopVariables)     // user variables

        // Use the standard ControlFlow.condBody() - this will create curr_cond properly
        val forCondBody = ControlFlow.condBody()

        // Create the loop body that handles ONNX semantics
        val loopBody = loopBody(
            importedBody = importedBody,
            funcName = funcName,
            parent = sd,
            funcBody = body
        )

        // Create output names
        val outputNamesList = mutableListOf<String>()
        // First add internal loop variable names
        outputNamesList.add("${funcName}_final_iter")
        outputNamesList.add("${funcName}_final_max_iter")
        outputNamesList.add("${funcName}_final_cond")
        // Then add user loop variable output names
        loopVariables.forEach { lv ->
            outputNamesList.add("${funcName}_${lv.name()}_output")
        }

        // Override with actual expected output names if provided
        for(i in 0 until op.outputsOfOp.size) {
            val outputIndex = 3 + i // Skip the first 3 internal outputs
            if (outputIndex < outputNamesList.size) {
                outputNamesList[outputIndex] = op.outputsOfOp[i]
            }
        }

        val loopRet = sd.whileLoop(
            outputNamesList.toTypedArray(),
            funcName,
            loopVars.toTypedArray(),
            forCondBody,
            loopBody
        )

        // Return only the actual user outputs (skip internal loop variables)
        val userOutputs = if (loopRet.size > 3) loopRet.drop(3) else emptyList()
        val ret2 = userOutputs.associate { output -> output.name() to listOf(output) }.toMutableMap()
        return ret2
    }

    fun loopBody(importedBody: OnnxIRGraph, parent: SameDiff, funcBody: SameDiff, funcName: String): SameDiffLambda {
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
            registryCast,
            false
        )
        body.isEagerMode = false
        sd.putSubFunction(funcName,body)
        sd.isEagerMode = false

        // Validate the imported graph inputs/outputs
        if (importedBody.inputList.size < 2) {
            throw IllegalArgumentException("Loop body must have at least 2 inputs (iteration_num, condition)")
        }

        if (importedBody.outputList.isEmpty()) {
            throw IllegalArgumentException("Loop body must have at least 1 output (condition)")
        }

        // Validate iteration number input (first input should be numerical)
        val iterCountVar = importedBody.inputList[0]
        val iterCountVarImported = body.getVariable(iterCountVar)
        if(iterCountVarImported != null && !iterCountVarImported.dataType().isNumerical) {
            throw IllegalArgumentException("Loop body first input (iteration number) must be numerical, got: ${iterCountVarImported.dataType()}")
        }

        // Validate condition input (second input should be boolean)
        val condInVar = importedBody.inputList[1]
        val condInVarImported = body.getVariable(condInVar)
        if(condInVarImported != null && condInVarImported.dataType() != DataType.BOOL) {
            throw IllegalArgumentException("Loop body second input (condition) must be boolean, got: ${condInVarImported.dataType()}")
        }

        return body
    }
}