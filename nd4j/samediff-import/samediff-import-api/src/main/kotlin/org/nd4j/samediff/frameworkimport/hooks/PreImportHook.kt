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
package org.nd4j.samediff.frameworkimport.hooks

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * The hook fore preprocessing
 * model import contexts.
 * Can be used to implement custom import flows
 * if an [MappingProcess] can't be defined for the op.
 *
 * @author Adam Gibson
 */
interface PreImportHook {

    fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>,
        isFinalOutput: Boolean,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>

    ): HookResult {
        val ret =  HookResult(outputVariables = handleOutputs(
            outputNames,
            sd,
            op,
            attributes,
            mappingRegistry,
            importGraph,dynamicVariables
        ),
            proceedWithInit = false)
        //override old op
        ret.outputVariables.entries.forEach { entry ->
            //relative to each op's creator rename the relevant op to match the output variable name
            //this will compensate for each variable
            val op = entry.value[0].creator
            val renameOp = sd.ops.remove(op.ownName)
            val oldName = op.ownName
            val inputVar = sd.variables[entry.value[0].name()]
            op.ownName = entry.key
            renameOp!!.name = entry.key
            sd.ops[entry.key] = renameOp
            val opVar = sd.variables[entry.value[0].name()]
            sd.variables.forEach { name,variable ->
                if(variable.inputsForOp != null)
                    while(variable.inputsForOp.contains(oldName)) {
                        variable.inputsForOp[variable.inputsForOp.indexOf(oldName)] = entry.key
                    }
            }

            opVar!!.outputOfOp = entry.key
            /**
             * Change op output to make sure op name is accounted for. The op name
             * is not propagated for output variables.
             */

        }

        return ret
    }

    fun handleOutputs(
        outputNames: List<String>,
        sd: SameDiff,
        op: SameDiffOp,
        attributes: Map<String, Any>,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String,List<SDVariable>> {
        outputNames.forEach { outputVarName ->
            if(outputVarName != null && sd.hasVariable(outputVarName)) {
                sd.variables.remove(outputVarName)
                sd.ops.remove(outputVarName)
            }
        }

        op.outputsOfOp = outputNames

        return doImport(sd, attributes, outputNames, op, mappingRegistry, importGraph, dynamicVariables)
    }

    fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String,List<SDVariable>>


}