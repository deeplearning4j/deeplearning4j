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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX MaxRoiPool operation.
 *
 * ONNX MaxRoiPool spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool
 *
 * ROI max pooling over regions of interest from feature maps.
 *
 * Inputs:
 * - X: Feature map of shape [N, C, H, W]
 * - rois: Regions of interest [num_rois, 5] where each row is [batch_index, x1, y1, x2, y2]
 *
 * Attributes:
 * - pooled_shape: Output spatial size [pooled_height, pooled_width]
 * - spatial_scale: Scale to map ROI coordinates to feature map (default: 1.0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["MaxRoiPool"], frameworkName = "onnx")
class MaxRoiPool : PreImportHook {

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
        val featureMap = sd.getVariable(op.inputsToOp[0])  // [N, C, H, W]
        val rois = sd.getVariable(op.inputsToOp[1])  // [num_rois, 5]
        
        // Get attributes
        val pooledShape = getIntList(attributes, "pooled_shape", listOf(1, 1))
        val pooledH = pooledShape.getOrElse(0) { 1 }
        val pooledW = pooledShape.getOrElse(1) { 1 }
        val spatialScale = (attributes.getOrDefault("spatial_scale", 1.0) as Number).toDouble()
        
        // Use SameDiff's ROI pooling if available
        // Note: This maps to the RoiPooling2D op with max pooling mode
        val output = sd.cnn.maxRoiPooling2d(
            outputNames[0],
            featureMap,
            rois,
            pooledH,
            pooledW,
            spatialScale
        )
        
        return mapOf(outputNames[0] to listOf(output))
    }
    
    private fun getIntList(attributes: Map<String, Any>, key: String, default: List<Int>): List<Int> {
        val value = attributes[key] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toInt() }
            is LongArray -> value.map { it.toInt() }
            is IntArray -> value.toList()
            else -> default
        }
    }
}
