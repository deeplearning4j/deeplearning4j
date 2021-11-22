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

import org.apache.commons.lang3.StringUtils
import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.common.util.ArrayUtil
import org.nd4j.enums.Mode
import org.nd4j.enums.WeightsFormat
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.ImportUtils
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of cast.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/cast.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Conv"],frameworkName = "onnx")
class Conv : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        val inWeights = sd.getVariable(op.inputsToOp[1])
        val weightsRank = inWeights.shape.size

        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val rank = weightsRank
        val xShape = inputVariable.shape
        val spatialSize = rank - 2
        val storageComputeFormat = ImportUtils.getDataFormat(rank)
        val computeIndex = storageComputeFormat.second.indexOf('C')
        val spatialFormat = StringUtils.join(storageComputeFormat.second.filter { input -> input == 'C' || input == 'W' })

        val perm = (2 to weightsRank - 1).toList() + listOf(1,0)
        val kernelShape = if(attributes.containsKey("kernel_shape")) {
            val kernelShapeList = attributes["kernel_shape"] as List<Int>
            kernelShapeList.map { input -> input }.toIntArray()
        } else {
            val weightsShape = inWeights.shape
            weightsShape.map { input -> input.toInt() }.toIntArray()
        }

        var weights = sd.permute(inWeights,*perm.toIntArray())
        var inWeightsShape = ArrayUtil.permute(ArrayUtil.copy(inWeights.shape),perm.toIntArray())
        val dilations = if(attributes.containsKey("dilations")) {
            val dilationsList = attributes["dilations"] as List<Int>
            val dilationsArr = dilationsList
            dilationsList.map { input -> input.toLong() }
        } else {
            List<Long>(spatialSize) { _ -> 1}
        }

        val spatialSizeConst = sd.constant(spatialSize)

        val strides = if(attributes.containsKey("strides")) {
            val stridesList = attributes["strides"] as List<Int>
            val stridesArr = stridesList
            stridesArr.map { input -> input.toLong() }

        } else {
            List<Long>(spatialSize) { _ -> 1}
        }

        val pads = if(attributes.containsKey("pads")) {
            val padsList = attributes["pads"] as List<Int>
            padsList.map { input -> input.toLong() }
        } else {
            val newPadsList = mutableListOf<Long>(0,0)
            for(i in 0 until spatialSize) {
                newPadsList.add(0)
            }
            newPadsList
        }

        val defaultPads2 = defaultPads(spatialSize)
        var padMode = attributes["auto_pad"] as String?
        if(!attributes.containsKey("auto_pad") || attributes["auto_pad"] == "NOTSET") {
            if(pads != defaultPads2) {
                inputVariable = paddingOp(sd,inputVariable,pads)
                //note our padding is not quite the same is onnx
                //our valid is equivalent to NOTSET and paddings should not be modified
                padMode = "NOTSET"
            }
        } else if(padMode == "SAME_UPPER") {
            padMode = "SAME"
        } else if(padMode == "VALID") {
            padMode = "VALID"
        } else if(padMode == "SAME_LOWER") {
            throw IllegalArgumentException("Unable to convert model running SAME_LOWER")
        }


        var groups = attributes.getOrDefault("group",1) as Long
        var depthWise = (rank == 4 && weightsRank == 4 && groups.toInt() != 1)
        if(depthWise && xShape != null && xShape[1].toInt() != -1) {
            depthWise = depthWise && groups == xShape[1]
        }
        /*  if depthwise and x.get_shape().as_list()[1] != None:
      depthwise = bool(group == x.get_shape().as_list()[1])
        * */
        var xs = mutableListOf<SDVariable>()
        var weightGroupsList = mutableListOf<SDVariable>()
        if(depthWise) {
            val depthWiseFilterShape = mutableListOf<Int>()
            for(i in 0 until 2) depthWiseFilterShape.add(inWeightsShape[i].toInt())
            depthWiseFilterShape.add(-1)
            depthWiseFilterShape.add(Math.floorDiv(weights.shape[3].toInt(),groups.toInt()))
            weights = weights.reshape(*depthWiseFilterShape.toIntArray())
            inputVariable = sd.permute(inputVariable,*ImportUtils.getPermFromFormats(storageComputeFormat.first,storageComputeFormat.second))
            xs.add(inputVariable)
            weightGroupsList.add(weights)

        } else {
            val weightGroups = sd.split(weights,groups.toInt(),-1)
            inputVariable = sd.permute(inputVariable,*ImportUtils.getPermFromFormats(storageComputeFormat.first,storageComputeFormat.second))
            if(groups.toInt() == 1)
                xs.add(inputVariable)
            else {
                xs.addAll(sd.split(inputVariable,groups.toInt(),-1))
            }
            weightGroupsList.addAll(weightGroups)
        }

        val convolvedList = mutableListOf<SDVariable>()
        var stridesList = mutableListOf<Long>()
        if(depthWise) {
            if(storageComputeFormat.second == "NHWC") {
                stridesList.add(1)
                stridesList.addAll(strides)
                stridesList.add(1)
            } else {
                stridesList.add(1)
                stridesList.add(1)
                stridesList.addAll(strides)
            }

            val convConfig = Conv2DConfig.builder()
                .kH(kernelShape[0].toLong())
                .kW(kernelShape[1].toLong())
                .sH(strides[0])
                .sW(strides[1])
                .dH(dilations[0])
                .dW(dilations[1])
                .dataFormat("NWHC")
                .weightsFormat(WeightsFormat.YXIO)
                .paddingMode(padModeForName(padMode!!))
                .build()

            for(i in 0 until xs.size) {
                var depthWiseConv2d = sd.cnn().depthWiseConv2d(xs[i.toInt()], weightGroupsList[i.toInt()], convConfig)
                convolvedList.add(depthWiseConv2d)
            }
        } else {
            for(i in 0 until groups) {
                if(rank == 3) {
                    //notset => valid
                    //valid => valid + pads zeroed
                    var totalPad = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[0]
                    }
                    val oneDConfig = Conv1DConfig.builder()
                        .k(kernelShape[0].toLong())
                        .dataFormat("NWC")
                        .d(dilations[0])
                        .p(totalPad)
                        .s(strides[0])
                        .paddingMode(PaddingMode.valueOf(padMode!!))
                        .build()
                    var convolved = sd.cnn().conv1d(xs[i.toInt()],weightGroupsList[i.toInt()], oneDConfig)
                    if(pads[0] > 0) {
                        convolved = convolved.get(*indicesForPads("NWC",pads).toTypedArray())
                    }
                    convolvedList.add(convolved)

                } else if(rank == 4) {
                    //notset => valid
                    //valid => valid + pads zeroed
                    var totalPadHeight = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[1]
                    }
                    var totalPadWidth = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[2]
                    }

                    val convConfig = Conv2DConfig.builder()
                        .kH(kernelShape[0].toLong())
                        .kW(kernelShape[1].toLong())
                        .sH(strides[0])
                        .sW(strides[1])
                        .pH(totalPadHeight)
                        .pW(totalPadWidth)
                        .dH(dilations[0])
                        .dW(dilations[1])
                        .dataFormat("NHWC")
                        .weightsFormat(WeightsFormat.YXIO)
                        .paddingMode(padModeForName(padMode!!))
                        .build()
                    var conv2d = sd.cnn().conv2d(xs[i.toInt()], weightGroupsList[i.toInt()], convConfig)
                    convolvedList.add(conv2d)

                } else if(rank == 5) {
                    var totalPadHeight = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[1]
                    }
                    var totalPadWidth = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[2]
                    }

                    var totalPadDepth = if(padMode == "NOTSET") {
                        0
                    } else {
                        pads[2]
                    }

                    val threeDConfig = Conv3DConfig.builder()
                        .kD(kernelShape[0].toLong())
                        .kH(kernelShape[1].toLong())
                        .kW(kernelShape[2].toLong())
                        .dD(dilations[0])
                        .dH(dilations[1])
                        .dW(dilations[2])
                        .pD(totalPadDepth).pH(totalPadHeight)
                        .pW(totalPadWidth)
                        .biasUsed(false)
                        .dataFormat("NWHDC")
                        .paddingMode(padModeForName(padMode!!))
                        .build()
                    var conv3d = sd.cnn().conv3d(xs[i.toInt()],weightGroupsList[i.toInt()], threeDConfig)
                    convolvedList.add(conv3d)

                }
            }
        }


        //remove old op: as there is no 1 to 1 relationship
        //when saving the model it will throw a nullpointer exception
        //with a not fully configured op. The convolution operation
        //in this model replaces the 1 to 1 intended call
        //otherwise used.
        sd.ops.remove(op.name)

        //grouped convolutions need to handle bias differently
        if(op.inputsToOp.size > 2) {
            val bias = sd.getVariable(op.inputsToOp[2])
            var output = sd.concat(-1,*convolvedList.toTypedArray())
            output = output.add(bias)
            output = sd.permute(outputNames[0],output,*ImportUtils.getPermFromFormats(storageComputeFormat.second,storageComputeFormat.first))
            return mapOf(output.name() to listOf(output))
        } else {
            var output = sd.concat(-1,*convolvedList.toTypedArray())
            val newPermute = ImportUtils.getPermFromFormats(storageComputeFormat.second,storageComputeFormat.first)
            output = sd.permute(outputNames[0],output,*newPermute)
            return mapOf(output.name() to listOf(output))
        }
    }



    fun padModeForName(name: String): PaddingMode {
        return when(name) {
            "VALID" -> PaddingMode.VALID
            "SAME" -> PaddingMode.SAME
            "NOTSET" -> PaddingMode.VALID
            else -> PaddingMode.CAUSAL
        }
    }

    fun indicesForPads(dataFormat: String,pads: List<Long>): List<SDIndex> {
        val ret = ArrayList<SDIndex>()
        val rank = dataFormat.length
        when(pads.size) {
            //1D cnn
            3 -> {
                val widthIdx = dataFormat.indexOf("W")
                for(i in 0 until rank) {
                    if(i == widthIdx) {
                        ret.add(SDIndex.interval(pads[i], - pads[i] - 1))
                    } else {
                        ret.add(SDIndex.all())

                    }
                }
            }
            //2d CNN
            4 -> {
                val widthIdx = dataFormat.indexOf("W")
                val heightIdx = dataFormat.indexOf("H")
                for(i in 0 until rank) {
                    if(i == widthIdx) {
                        ret.add(SDIndex.interval(pads[i], - pads[i] - 1))

                    } else if(i == heightIdx) {
                        ret.add(SDIndex.interval(pads[i],- pads[i] - 1))

                    } else {
                        ret.add(SDIndex.all())

                    }
                }
            }
            //3D CNN
            5 -> {
                val widthIdx = dataFormat.indexOf("W")
                val heightIdx = dataFormat.indexOf("H")
                val depthIdx = dataFormat.indexOf("D")
                for(i in 0 until rank) {
                    if(i == widthIdx) {
                        ret.add(SDIndex.interval(pads[i], - pads[i] - 1))

                    } else if(i == heightIdx) {
                        ret.add(SDIndex.interval(pads[i], - pads[i] - 1))

                    } else if(i == depthIdx) {
                        ret.add(SDIndex.interval(pads[i], - pads[i] - 1))

                    } else {
                        ret.add(SDIndex.all())
                    }
                }

            }
        }

        return ret
    }

    fun defaultPads(spatialSize : Int): List<Int> {
        val newPadsList = mutableListOf(0,0)
        for(i in 0 until spatialSize) {
            newPadsList.add(0)
        }
        return newPadsList
    }

    fun paddingOp(sd: SameDiff,x: SDVariable,pads: List<Long>): SDVariable {
        val numDim = pads.size / 2
        val newPads = Nd4j.create(Nd4j.createBuffer(pads.toLongArray())).transpose().reshape('c',2,numDim)
        val newPads2 = Nd4j.concat(0,Nd4j.create(Nd4j.createBuffer(longArrayOf(0,0,0,0))).reshape(4),newPads.ravel().reshape(newPads.length()))
        val inputPadding = sd.constant(newPads2.reshape('c',numDim + 2,2).castTo(DataType.INT32))
        return sd.image().pad(x,inputPadding,Mode.CONSTANT,0.0)
    }

}