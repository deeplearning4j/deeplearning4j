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
 * @author Adam Gibson
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
        var boxes = sd.getVariable(op.inputsToOp[0])   // ONNX: [batch, num_boxes, 4]
        var scores = sd.getVariable(op.inputsToOp[1])  // ONNX: [batch, num_classes, num_boxes]

        // Native op expects 2D boxes [num_boxes, 4] and 1D scores [num_boxes]
        // Squeeze batch dimension (assumes batch=1 for now)
        boxes = sd.squeeze("${opName}_boxesSqueeze", boxes, 0)  // [num_boxes, 4]
        
        // Squeeze batch dimension first, then class dimension
        scores = sd.squeeze("${opName}_scoresSqueeze1", scores, 0)  // [num_classes, num_boxes]
        scores = sd.squeeze("${opName}_scoresSqueeze2", scores, 0)  // [num_boxes]

        val maxOutputBoxes = if (op.inputsToOp.size > 2 && op.inputsToOp[2].isNotEmpty())
            sd.getVariable(op.inputsToOp[2]) else sd.constant(0L)
        val iouThreshold = if (op.inputsToOp.size > 3 && op.inputsToOp[3].isNotEmpty())
            sd.getVariable(op.inputsToOp[3]) else sd.constant(0.5)
        val scoreThreshold = if (op.inputsToOp.size > 4 && op.inputsToOp[4].isNotEmpty())
            sd.getVariable(op.inputsToOp[4]) else sd.constant(0.0)

        // Call native NMS op
        val result = org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression(
            sd,
            boxes,
            scores,
            maxOutputBoxes,
            iouThreshold,
            scoreThreshold
        ).outputVariable().rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(result))
    }
}
