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

/**
 * A port of resize.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/resize.py#L195
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Resize"],frameworkName = "onnx")
class Resize : PreImportHook  {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>
    ): HookResult {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#resize
        val inputVariable = sd.getVariable(op.inputsToOp[0])
        val inputShape = sd.shape(inputVariable)
        val roi = sd.getVariable(op.inputsToOp[1])
        val scales = sd.getVariable(op.inputsToOp[2])
        val sizes = sizes(sd,op)
        /**
         *
         * If coordinate_transformation_mode is "half_pixel",
        x_original = (x_resized + 0.5) / scale - 0.5,

        if coordinate_transformation_mode is "pytorch_half_pixel",
        x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,

        if coordinate_transformation_mode is "align_corners",
        x_original = x_resized * (length_original - 1) / (length_resized - 1),

        if coordinate_transformation_mode is "asymmetric",
        x_original = x_resized / scale,

        if coordinate_transformation_mode is "tf_crop_and_resize",
        x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).
         */
        val coordTransformationMode = attributes.getOrDefault("coordinate_transformation_mode","half_pixel") as String
        val extrapolationValue = attributes.getOrDefault("extrapolation_value",0.0) as Double
        /**
         * Three interpolation modes: nearest (default), linear and cubic. The "linear" mode includes linear
         * interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor).
         * The "cubic" mode includes cubic interpolation for 1D tensor
         * and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).
         */
        val mode = attributes.getOrDefault("mode","nearest") as String

        val outputSize = outputSize(sd,op,inputVariable,scales,sizes)
        outputSize!!.setShape(2)

        //switch to NWHC (tensorflow format) and then back to NCHW (onnx format)
        val transpose = sd.permute(inputVariable,0,2,3,1)
        var result: SDVariable? = null
        when (coordTransformationMode) {
            "tf_crop_and_resize" -> {
                val indices = mutableListOf<Int>()
                val rank = inputVariable.arr.rank()
                for(i in 2 until rank) {
                    indices.add(i - 2,i)
                    indices.add(i,i + rank)
                }

                val boxes = sd.expandDims(sd.gather(roi,indices.toIntArray(),0),0)
                val boxIndices = sd.range(0.0,inputVariable.shape[0] as Double,1.0, DataType.INT64)
                result =  sd.image().cropAndResize(inputVariable,boxes,boxIndices,outputSize,extrapolationValue)
            }
            "align_corners" -> {
                result =  invokeResize(mode,sd,inputVariable,outputSize,true,false)
            }
            "asymmetirc" -> {
                result = invokeResize(mode,sd,inputVariable,outputSize,false,false)
            }
            else -> {
                result = sd.image().imageResize(inputVariable,outputSize,false,false,ImageResizeMethod.ResizeNearest)
            }
        }

        val finalOutput = sd.permute(result,0,3,1,2)

        return HookResult(outputVariables = mapOf(finalOutput.name() to listOf(finalOutput)),
            proceedWithInit = false)


    }

    fun invokeResize(type: String,sd: SameDiff,input: SDVariable,size: SDVariable,alignCorners: Boolean,halfPixelCenters: Boolean): SDVariable? {
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
                sd.image().imageResize(input,size,false,false,ImageResizeMethod.ResizeNearest)
            }
        }
    }

    fun outputSize(sd: SameDiff,op: SameDiffOp,input: SDVariable,scales: SDVariable,sizes: SDVariable): SDVariable?  {
        var ret: SDVariable? = null
        ret = if(op.inputsToOp.size == 3) {
            val heightWidthScale = sd.constant(scales.arr.get(NDArrayIndex.interval(2,scales.arr.length())))
            val heightWidthShape = sd.constant(Nd4j.create(input.shape.asList().subList(2,input.shape.size)))
            val scaled = sd.math.mul(heightWidthScale,heightWidthShape)
            scaled
        } else {
            sizes.setShape(*input.shape)
            sizes.get(SDIndex.interval(2, ArrayUtil.prod(*sizes.shape)))
        }
        return ret
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