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
import org.nd4j.samediff.frameworkimport.ImportGraph
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

        anchors = sd.bitwise().


        //output 0: rois
        //output1: roi_probs
        return mapOf("" to listOf(filteredAnchors))
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