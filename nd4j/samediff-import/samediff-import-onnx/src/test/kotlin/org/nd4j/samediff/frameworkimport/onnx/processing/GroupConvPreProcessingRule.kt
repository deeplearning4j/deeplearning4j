/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.samediff.frameworkimport.onnx.processing

import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.enums.DataFormat
import org.nd4j.enums.WeightsFormat
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import java.lang.IllegalArgumentException

@PreHookRule(nodeNames = ["conv2_1"],opNames = [],"onnx")
class GroupConvPreProcessingRule: PreImportHook {
    override fun preProcess(op: SameDiffOp, sd: SameDiff, attributes: Map<String, Any>): HookResult {
        if(op.op.opName() != "conv2d") {
            throw IllegalArgumentException("Illegal op being processed of type ${op.op.opName()} with node name ${op.op.ownName}")
        }

        val numSizeSplits = attributes.getOrDefault("group",1) as Long
        if(numSizeSplits.toInt() == 1) {
            //no need to split, just perform 1 convolution op
            return HookResult()
        }

        val conv2d = op.op as DynamicCustomOp
        val config = Conv2DConfig.builder()
            .sH(conv2d.getIArgument(2))
            .sW(conv2d.getIArgument(3))
            .kH(conv2d.getIArgument(0))
            .kW(conv2d.getIArgument(1))
            .pH(conv2d.getIArgument(4))
            .pW(conv2d.getIArgument(5))
            .dH(conv2d.getIArgument(6))
            .dW(conv2d.getIArgument(7))
            .isSameMode(conv2d.getIArgument(8) > 0)
            .weightsFormat(WeightsFormat.values()[conv2d.getIArgument(10).toInt()])
            .dataFormat(DataFormat.values()[conv2d.getIArgument(9).toInt()].name)
            .build()

        val listOfFunctions = ArrayList<DifferentialFunction>()
        val weights = sd.getVariable(op.inputsToOp[1])
        //for onnx, this is the number of ops
        val split = sd.split(op.name + "_split",weights,numSizeSplits.toInt(),1)
        val resultMap = HashMap<String,List<SDVariable>>()
        /**
         * NOTE: Need to look in to how to wire up inputs and outputs properly.
         * May need HookResult to return an indicator of variables and ops to remove.
         */
        val outputVars = ArrayList<SDVariable>()
        split.forEachIndexed { index,input ->
            val varName = "${op.name}_split/$index"
            if(sd.hasVariable(varName))
                sd.variables.remove(varName)
            val outputVariable = sd.cnn().conv2d(varName,input,weights,config)
            resultMap[outputVariable.name()] = listOf(outputVariable)
            outputVars.add(outputVariable)
        }

        sd.ops.remove(op.name)
        sd.variables.remove(op.name)

        /**
         * TODO: Fix output names and potentially look for other inputs
         * in graph where we need to redirect the input/output names
         */
        val toTypedArray = outputVars.toTypedArray()
        val concat = sd.concat(op.name,0,*toTypedArray)
        resultMap[op.name] = listOf(concat)

        return HookResult(outputVariables = resultMap,listOfFunctions)
    }
}