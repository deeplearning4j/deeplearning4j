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

import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.common.util.ArrayUtil
import org.nd4j.enums.ImageResizeMethod
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import java.lang.IllegalArgumentException

/**
 * A port of minimum.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/minimum.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Min"],frameworkName = "onnx")
class Minimum : PreImportHook  {
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
                    onGoingOutput = sd.min(onGoingOutput,currVariable)
                else {
                    onGoingOutput = sd.min(outputVarName,onGoingOutput,currVariable)
                }
            }
        }



        return HookResult(outputVariables = mapOf(onGoingOutput!!.name() to listOf(onGoingOutput!!)),
            proceedWithInit = false)

    }

    fun invokeResize(
        type: String,
        sd: SameDiff,
        input: SDVariable,
        size: SDVariable,
        alignCorners: Boolean,
        halfPixelCenters: Boolean
    ): SDVariable? {
        return when (type) {
            "linear" -> {
                val height = size.arr.getInt(0)
                val width = size.arr.getInt(1)
                sd.image().resizeBiLinear(input,height,width, alignCorners, halfPixelCenters)
            }
            "cubic" -> {
                sd.image().resizeBiCubic(input,size,alignCorners,halfPixelCenters)
            }
            else -> {
                sd.image().imageResize(input,size,true,true,ImageResizeMethod.ResizeNearest)
            }
        }
    }

    fun outputSize(
        sd: SameDiff,
        op: SameDiffOp,
        input: SDVariable,
        scales: SDVariable,
        sizes: SDVariable,
        inputVariableShape: SDVariable
    ): SDVariable?  {
        var ret: SDVariable? = null
        ret = if(op.inputsToOp.size == 3) {
            val heightWidthScale = scales.get(SDIndex.interval(2,-1))
            val subGet = inputVariableShape.get(SDIndex.interval(2,-1))
            val heightWidthShape = sd.castTo(subGet,heightWidthScale.dataType())
            val scaled = sd.castTo(sd.math.mul(heightWidthScale,heightWidthShape),DataType.INT32)
            scaled
        } else {
            sizes.setShape(*inputVariableShape.shape)
            sd.castTo(sizes.get(SDIndex.interval(2, -1)),DataType.INT32)
        }
        return ret.castTo(DataType.INT32)
    }

    fun alignCornersFor(coordTransformationMode: String): Boolean {
        //note this includes the coordTransformationMode == "asymmetric"
        return coordTransformationMode == "align_corners"
    }

    fun sizes(sd: SameDiff,op: SameDiffOp): SDVariable {
        if(op.inputsToOp.size == 4)
            return sd.getVariable(op.inputsToOp[3])
        else
            return sd.constant(Nd4j.empty())
    }

}