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
import org.nd4j.enums.Mode
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.ImportUtils
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of roi_align.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/roi_align.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["RoiAlign"],frameworkName = "onnx")
class RoiAlign : PreImportHook  {


    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        var features = sd.getVariable(op.inputsToOp[0])
        val boxes = sd.getVariable(op.inputsToOp[1])
        val indx = sd.getVariable(op.inputsToOp[2])
        val outputHeight = attributes["output_height"] as Long
        var outputWidth = attributes["output_width"] as Long
        var samplingRatio = attributes["sampling_ratio"] as Long
        var spatialScale = attributes["spatial_scale"] as Float
        var adaptiveRatio = false
        if(samplingRatio <= 0) {
            samplingRatio = (outputHeight + outputWidth) / 2
            adaptiveRatio = true
        }

        val dataFormat = ImportUtils.getDataFormat(features.arr.rank())
        val needsTrans = dataFormat.first.startsWith("NC")
        if(needsTrans) {
            val computeFormat = "N${dataFormat.first.substring(2)}C"
            val getPerm = ImportUtils.getPermFromFormats(dataFormat.first,computeFormat)
            features = sd.permute(features,*getPerm)
        }

        val newBoxes = boxes.mul(spatialScale.toDouble())
        val cropped = cropAndResize(sd,features,newBoxes,indx,
            intArrayOf(outputHeight.toInt(),outputWidth.toInt()),
            samplingRatio,adaptiveRatio)

        val pooled = sd.cnn().avgPooling2d(cropped,
            Pooling2DConfig.builder()
                .kH(samplingRatio)
                .kW(samplingRatio)
                .sH(samplingRatio)
                .sW(samplingRatio)
                .pH(1).pW(1).isNHWC(true)
                .isSameMode(true).build())
        val outputVar = sd.permute(outputNames[0],pooled,0,3,1,2)
        return mapOf(outputNames[0] to listOf(outputVar))
    }


    private fun cropAndResize(sd: SameDiff, image: SDVariable, boxes: SDVariable, boxesInd: SDVariable, cropSize: IntArray,
                              samplingRatio: Long, adaptiveRatio: Boolean = false, padBorder: Boolean = false): SDVariable {

        var boxes2 = if(padBorder) {
            boxes.add(1.0)
        } else {
            boxes
        }
        var image2 = if(padBorder) {
            sd.image().pad(image,sd.constant(Nd4j.create(
                floatArrayOf(0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f)
            )).reshape(4,2), Mode.SYMMETRIC,0.0)
        } else {
            image
        }

        val imageShape = sd.shape(image2).get(SDIndex.interval(1,3))
        val boxes3 = transformFpCoorTf(sd,boxes2,imageShape,cropSize,samplingRatio, adaptiveRatio)
        val sdCrop = sd.constant(Nd4j.create(Nd4j.createBuffer(longArrayOf(cropSize[0] * samplingRatio,cropSize[1] * samplingRatio))))
        val ret = sd.image().cropAndResize(image2,boxes3,boxesInd,sdCrop)
        return ret

    }

    fun transformFpCoorTf(sd: SameDiff,boxes: SDVariable,imageShape: SDVariable,cropSize: IntArray,samplingRatio: Long,adaptiveRatio: Boolean): SDVariable {
        val splitInput = sd.split(boxes,4,1)
        val x0 = splitInput[0]
        val y0 = splitInput[1]
        val x1 = splitInput[2]
        val y1 = splitInput[3]
        if(!adaptiveRatio) {
            val cropShape = arrayOf(cropSize[0] * samplingRatio,cropSize[1] * samplingRatio)
            val spacingWidth = x1.sub(x0).div(floatConstVar(sd,cropShape[1]))
            val spacingHeight = y1.sub(y0).div(floatConstVar(sd,cropShape[0]))
            val nx0 = x0.add(spacingWidth.div(2.0)).div(imageShape.get(SDIndex.point(1)).sub(1.0))
            val ny0 = y0.add(spacingHeight.div(2.0)).div(imageShape.get(SDIndex.point(0)).sub(1.0))
            val nW = spacingWidth.mul(floatConstVar(sd,cropShape[1] - 1).div((imageShape.get(SDIndex.point(1)).sub(1.0))))
            val nH = spacingWidth.mul(floatConstVar(sd,cropShape[0] - 1).div((imageShape.get(SDIndex.point(0)).sub(1.0))))
            return sd.concat(1,ny0,nx0,ny0.add(nH),nx0.add(nW))
        } else {
            val roiWidth = x1.sub(x0)
            val roiHeight = y1.sub(y0)
            val nx0 = x0.div(imageShape.get(SDIndex.point(1)).sub(1.0))
            val ny0 = y0.div(imageShape.get(SDIndex.point(1)).sub(1.0))
            val nW = roiWidth.sub(1.0).div(imageShape.get(SDIndex.point(1)).sub(1.0))
            val nH = roiHeight.sub(1.0).div(imageShape.get(SDIndex.point(1)).sub(1.0))
            return sd.concat(1,ny0,nx0,ny0.add(nH),nx0.add(nW))
        }
    }

    fun floatConstVar(sd: SameDiff,input: Long): SDVariable {
        return sd.constant(Nd4j.create(floatArrayOf(input.toFloat())))
    }

}