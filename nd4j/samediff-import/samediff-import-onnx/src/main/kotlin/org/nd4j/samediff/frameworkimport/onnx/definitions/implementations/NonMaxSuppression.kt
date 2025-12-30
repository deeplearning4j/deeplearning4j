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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX NonMaxSuppression operation.
 *
 * ONNX NonMaxSuppression spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
 *
 * Performs non-maximum suppression on bounding boxes.
 *
 * Inputs:
 * - boxes: Box coordinates [num_batches, spatial_dimension, 4]
 * - scores: Scores [num_batches, num_classes, spatial_dimension]
 * - max_output_boxes_per_class: Maximum boxes per class (optional)
 * - iou_threshold: IOU threshold for suppression (optional)
 * - score_threshold: Score threshold for filtering (optional)
 *
 * Attributes:
 * - center_point_box: Box format (0=corner, 1=center)
 *
 * Output:
 * - selected_indices: [num_selected_indices, 3] with [batch_index, class_index, box_index]
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["NonMaxSuppression"], frameworkName = "onnx")
class NonMaxSuppression : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val opName = op.name

        // Get inputs
        val boxes = sd.getVariable(op.inputsToOp[0])   // [batch, spatial, 4]
        val scores = sd.getVariable(op.inputsToOp[1])  // [batch, classes, spatial]

        val maxOutputBoxes = if (op.inputsToOp.size > 2 && op.inputsToOp[2].isNotEmpty())
            sd.getVariable(op.inputsToOp[2]) else null
        val iouThreshold = if (op.inputsToOp.size > 3 && op.inputsToOp[3].isNotEmpty())
            sd.getVariable(op.inputsToOp[3]) else null
        val scoreThreshold = if (op.inputsToOp.size > 4 && op.inputsToOp[4].isNotEmpty())
            sd.getVariable(op.inputsToOp[4]) else null

        // Get center_point_box attribute (0=corner format, 1=center format)
        val centerPointBox = (attributes.getOrDefault("center_point_box", 0L) as Number).toInt()

        // Default values
        val maxBoxes = maxOutputBoxes?.castTo(DataType.INT32) ?: sd.constant(0)
        val iouThresh = iouThreshold?.castTo(DataType.FLOAT) ?: sd.constant(0.0f)
        val scoreThresh = scoreThreshold?.castTo(DataType.FLOAT) ?: sd.constant(Float.NEGATIVE_INFINITY)

        // Use SameDiff's non_max_suppression_v3
        // Note: SameDiff NMS expects [batch, num_boxes, 4] for boxes
        // and [batch, num_boxes] for scores (single class at a time)

        // For multi-class NMS, we need to process each class separately
        // This is a simplified implementation for single-class or first-class NMS
        val scoresFirstClass = sd.stridedSlice("${opName}_scores_slice", scores,
            intArrayOf(0, 0, 0), intArrayOf(Int.MAX_VALUE, 1, Int.MAX_VALUE), intArrayOf(1, 1, 1))
        val scoresSqueeezed = sd.squeeze("${opName}_scores_squeeze", scoresFirstClass, 1)

        // Apply NMS
        val selectedIndices = sd.image.nonMaxSuppression(
            "${opName}_nms",
            boxes,
            scoresSqueeezed,
            maxBoxes,
            iouThresh,
            scoreThresh
        )

        // Reshape output to [num_selected, 3] format
        val numSelected = sd.shape("${opName}_num_selected", selectedIndices)[0]
        val batchIndices = sd.zerosLike("${opName}_batch_idx", selectedIndices)
        val classIndices = sd.zerosLike("${opName}_class_idx", selectedIndices)

        val output = sd.stack("${opName}_stack", 1,
            batchIndices.castTo(DataType.INT64),
            classIndices.castTo(DataType.INT64),
            selectedIndices.castTo(DataType.INT64)
        ).rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(output))
    }
}
