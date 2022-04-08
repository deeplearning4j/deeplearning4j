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
package org.nd4j.samediff.frameworkimport.onnx.definitions

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.custom.Invoke.InvokeParams
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph


class OnnxInputTensors(op: SameDiffOp,sd: SameDiff,importedBody: OnnxIRGraph) {
    val op: SameDiffOp = op
    val sd: SameDiff = sd
    val importedBody = importedBody

    fun toInputTensors(): List<SDVariable> {
        val inputTensors = ArrayList<SDVariable>()
        val currIteration = sd.constant(0).castTo(DataType.INT64)
        //loop has 2 to N dependencies: the termination iterations and the custom condition
        //note when not specified we just loop the maximum number of iterations and let the user specify the termination condition
        val terminationIterations: SDVariable? = if(op.inputsToOp.size > 0 && op.inputsToOp[0] != "") sd.getVariable(op.inputsToOp[0]) else sd.constant(Long.MAX_VALUE)
        val cond: SDVariable? = if(op.inputsToOp.size > 1 && op.inputsToOp[1] != "") sd.getVariable(op.inputsToOp[1]) else sd.constant(true)
        inputTensors.add(currIteration)
        if(terminationIterations != null)
            inputTensors.add(terminationIterations)

        if(cond != null)
            inputTensors.add(cond)

        for(i in 2 until importedBody.inputList.size) {
            inputTensors.add(sd.getVariable(op.inputsToOp[i]))
        }

        return inputTensors

    }
}
