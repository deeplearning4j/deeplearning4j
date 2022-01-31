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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.indexing.masking.Masking
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of  RPN-FasterRCNN for pytorch interop. See:
 * https://github.com/chaudhary-rohit/RPN-Faster-R-CNN/blob/2e63ee184241e2df3f8ecf7ca0cf7f27bed47d6e/RPN.py
 *
 * This handles implementing pytorch's GenerateProposals op used within the detectron framework
 * and exposed as a proprietary onnx op.
 * 
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["GenerateProposals"],frameworkName = "onnx")
class GenerateProposals : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#cumsum
        var scores = sd.getVariable(op.inputsToOp[0])
        var bboxDeltas = sd.getVariable(op.inputsToOp[1])
        var imInfo = sd.getVariable(op.inputsToOp[2])
        var anchors = sd.getVariable(op.inputsToOp[3])


        val spatialScale = attributes["spatial_scale"] as Float
        val prenmSTopN = attributes["pre_nms_topN"] as Int
        val postNmsTopN = attributes["post_nms_topN"] as Int
        val nmsThreshold = attributes["nms_thresh"] as Float
        val minSize = attributes["min_size"] as Float
        val angleboundOn = attributes.getOrDefault("angle_bound_on",true)
        val angleBoundLow = attributes.getOrDefault("angle_bound_lo",-90) as Int
        val angleBoundHigh = attributes.getOrDefault("angle_bound_hi",90) as Int
        val clipAngleThreshold = attributes.getOrDefault("clip_angle_thresh",1.0f) as Float


        //first step: compute filters for sorting scores
        val filteredAnchors = filter(sd,anchors, imInfo)
        var allScores = scores.get(SDIndex.all(),SDIndex.point(1))
        allScores = sd.reshape(allScores,-1)

        anchors = Masking.applyMask(sd,anchors,filteredAnchors,0)
        val bboxPredict = Masking.applyMask(sd,bboxDeltas,filteredAnchors,0)
        scores = Masking.applyMask(sd,allScores,filteredAnchors,0)
        var proposals = decode(sd,anchors,bboxPredict)
        val minProbFilter = sd.gte(scores,0.0)
        val (x1,y1,x2,y2) = sd.unstack(proposals,1,4)
        val width = x2.sub(x1).add(1.0)
        val height = y2.sub(y1).add(1.0)
        val area = width.mul(height)
        val areaFilter = sd.gte(area,0.0)
         val netFilter = sd.bitwise().and(minProbFilter,areaFilter)
        val unsortedProposals = Masking.applyMask(sd,proposals,netFilter,0)
        val unsortedScores = Masking.applyMask(sd,scores,netFilter,0)
        val (topKScores,indices) = sd.nn().topK(unsortedScores,prenmSTopN as Double,true)
        var topKProposals = clipBoxes(sd,sd.gather(unsortedProposals,indices,0),imInfo)
        val orderedProposals = changeOrder(sd,topKProposals)
        val selectedIndices = sd.image().nonMaxSuppression(orderedProposals,sd.reshape(topKScores,-1),
            postNmsTopN,
            nmsThreshold.toDouble(),
            0.5)
        val nmsProposalOrder = sd.gather(orderedProposals,selectedIndices,0)
        proposals = changeOrder(sd,nmsProposalOrder)
        scores = sd.gather(topKScores,selectedIndices,0)
        //output 0: rois
        //output1: roi_probs
        return mapOf(outputNames[0] to listOf(proposals),outputNames[1] to listOf(scores))
    }


    fun changeOrder(sd: SameDiff,bboxes: SDVariable): SDVariable {
        val (firstMin,secondMin,firstMax,secondMax) = sd.unstack(bboxes,1,4)
        return sd.stack(1,secondMin,firstMin,secondMax,firstMax)
    }

    fun clipBoxes(sd: SameDiff,bboxes: SDVariable,imShape: SDVariable): SDVariable {
        val castedBboxes = bboxes.castTo(DataType.FLOAT)
        val castedImShape = imShape.castTo(DataType.FLOAT)
        var (x1,y1,x2,y2) = sd.split(castedBboxes,4,1)
        val width = castedImShape.get(SDIndex.point(1))
        val height = castedImShape.get(SDIndex.point(0))
        x1 = sd.math().max(sd.min(x1,width.sub(1.0)),sd.constant(0.0))
        x2 = sd.math().max(sd.min(x2,width.sub(1.0)),sd.constant(0.0))

        y1 = sd.math().max(sd.min(y1,height.sub(1.0)),sd.constant(0.0))
        y2 = sd.math().max(sd.min(y2,height.sub(1.0)),sd.constant(0.0))
        return sd.concat(1,x1,y1,x2,y2)

    }

    fun decode(sd: SameDiff,anchors: SDVariable,bboxPred: SDVariable): SDVariable {
        val (roiWidth,roiHeight,roiUrx,roiUry) = centerCorner(sd,anchors)
        val (dx,dy,dw,dh) = sd.split(bboxPred,4,1)
        val predUrX = dx.mul(roiWidth).add(roiUrx)
        val predUrY = dy.mul(roiHeight).add(roiUry)
        val predW = sd.math().exp(dw).mul(roiWidth)
        val predH = sd.math().exp(dh).mul(roiHeight)
        val bboxX1 = predUrX.sub(0.5).mul(predW)
        val bboxY1 = predUrY.sub(0.5).mul(predH)
        val bboxX2 = predUrX.add(0.5).mul(predW).sub(1.0)
        val bboxY2 = predUrY.add(0.5).mul(predH).sub(1.0)
        val bboxes = sd.concat(1,bboxX1,bboxX2,bboxY1,bboxY2)
        return bboxes
    }


    fun centerCorner(sd: SameDiff,bboxes: SDVariable): Array<SDVariable> {
        val bboxesCast = bboxes.castTo(DataType.FLOAT)
        val (x1,y1,x2,y2) = sd.split(bboxesCast,4,1)
        val width = x2.sub(x1).add(1.0)
        val height = y2.sub(y1).add(1.0)
        val urx = x1.add(.5).mul(width)
        val ury = y1.add(.5).mul(height)
        return arrayOf(width,height,urx,ury)
    }

    fun filter(sd: SameDiff, anchors: SDVariable, imInfo: SDVariable): SDVariable {
        val (xMin, yMin, xMax, yMax) = sd.unstack(anchors, 1, 4)
        return sd.bitwise().and(
            sd.bitwise().and(
                sd.gte(xMin, 0.0),
                sd.gte(yMin, 0.0)
            ),
            sd.bitwise().and(
                sd.lte(xMax, imInfo.get(SDIndex.point(1))),
                sd.lte(yMax, imInfo.get(SDIndex.point(1)))
            )
        )
    }


}