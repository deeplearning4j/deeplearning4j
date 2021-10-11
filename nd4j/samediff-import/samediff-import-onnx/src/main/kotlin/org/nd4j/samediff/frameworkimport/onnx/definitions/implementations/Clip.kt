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

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxDataType

/**
 * A port of expand.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/clip.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Clip"],frameworkName = "onnx")
class Clip : PreImportHook  {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>,
        isFinalOutput: Boolean
    ): HookResult {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#clip

        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val dtype2 = convertToOnnxDataType(inputVariable.dataType())

        /**
         * Need to figure out how to get min and max value for a given data type.
         * Initial suggestion: convert to known data type and get min/max values
         * Alternative: implement data type min max for each data type
         */
        val outputVarName: String? = if(isFinalOutput) {
            outputNames[0]
        } else null

        if(outputVarName != null && sd.hasVariable(outputVarName)) {
            sd.variables.remove(outputVarName)
            sd.ops.remove(outputVarName)
        }

        val min = if(op.inputsToOp.size > 1) {
            sd.getVariable(op.inputsToOp[1])
        } else {
            sd.minMax(inputVariable.dataType().toInt(),0)
        }


        val max = if(op.inputsToOp.size > 2) {
            sd.getVariable(op.inputsToOp[2])
        } else {
            sd.minMax(inputVariable.dataType().toInt(),1)
        }

        val clipped = sd.clipByValue(inputVariable,min,max)
        val outputVar = sd.castTo(outputVarName,clipped,inputVariable.dataType())


        return HookResult(outputVariables = mapOf(outputVar.name() to listOf(outputVar)),
            proceedWithInit = false)


    }



}