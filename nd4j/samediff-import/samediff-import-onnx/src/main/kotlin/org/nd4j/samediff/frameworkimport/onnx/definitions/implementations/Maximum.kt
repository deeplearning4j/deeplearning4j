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
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule

/**
 * A port of maximum.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/maximum.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Max"],frameworkName = "onnx")
class Maximum : PreImportHook  {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>,
        isFinalOutput: Boolean
    ): HookResult {

        val outputVarName: String? = if(isFinalOutput) {
            outputNames[0]
        } else null

        //remove pre existing output variable
        if(outputVarName != null && sd.hasVariable(outputVarName)) {
            sd.variables.remove(outputVarName)
            sd.ops.remove(outputVarName)
        }

        var onGoingOutput: SDVariable? = null
        op.inputsToOp.forEachIndexed { index,input ->
            val currVariable = sd.getVariable(input)
            if(onGoingOutput == null) {
                onGoingOutput = currVariable
            } else {
                if(index < op.inputsToOp.size - 1)
                    onGoingOutput = sd.max(onGoingOutput,currVariable)
                else {
                    onGoingOutput = sd.max(outputVarName,onGoingOutput,currVariable)
                }
            }
        }



        return HookResult(outputVariables = mapOf(onGoingOutput!!.name() to listOf(onGoingOutput!!)),
            proceedWithInit = false)


    }



}