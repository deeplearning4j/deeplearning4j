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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig
import org.nd4j.linalg.factory.Nd4j
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
 * This operation pools regions from a feature map using max pooling.
 *
 * Algorithm:
 * 1. Scale ROI coordinates by spatial_scale
 * 2. For each ROI, extract the corresponding region from the feature map
 * 3. Divide the region into (pooled_h x pooled_w) bins
 * 4. Apply max pooling over each bin
 *
 * Inputs:
 * - X: Feature map of shape [N, C, H, W]
 * - rois: Regions of interest [num_rois, 5] where each row is [batch_index, x1, y1, x2, y2]
 *
 * Attributes:
 * - pooled_shape: Output spatial size [pooled_height, pooled_width]
 * - spatial_scale: Scale to map ROI coordinates to feature map (default: 1.0)
 *
 * Output:
 * - Y: Pooled features of shape [num_rois, C, pooled_h, pooled_w]
 *
 * @author Adam Gibson
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

        // Get feature map shape
        val featureShape = featureMap.shape ?: throw IllegalStateException("MaxRoiPool requires static feature map shape")
        val numChannels = featureShape[1]
        val featureH = featureShape[2]
        val featureW = featureShape[3]

        // Convert feature map from NCHW to NHWC for cropAndResize
        val featureNhwc = sd.permute("${opName}_nhwc", featureMap, 0, 2, 3, 1)

        // Extract batch indices and box coordinates from ROIs
        // ROI format: [batch_index, x1, y1, x2, y2]
        val batchIndices = sd.squeeze(
            sd.stridedSlice("${opName}_batch_idx", rois, longArrayOf(0, 0), longArrayOf(Long.MAX_VALUE, 1), 1L, 1L),
            -1
        ).castTo(DataType.INT32)

        // Extract normalized box coordinates [y1, x1, y2, x2] (cropAndResize expects this order)
        // and normalize to [0, 1] range based on feature map size
        val x1 = sd.squeeze(sd.stridedSlice("${opName}_x1", rois, longArrayOf(0, 1), longArrayOf(Long.MAX_VALUE, 2), 1L, 1L), -1)
        val y1 = sd.squeeze(sd.stridedSlice("${opName}_y1", rois, longArrayOf(0, 2), longArrayOf(Long.MAX_VALUE, 3), 1L, 1L), -1)
        val x2 = sd.squeeze(sd.stridedSlice("${opName}_x2", rois, longArrayOf(0, 3), longArrayOf(Long.MAX_VALUE, 4), 1L, 1L), -1)
        val y2 = sd.squeeze(sd.stridedSlice("${opName}_y2", rois, longArrayOf(0, 4), longArrayOf(Long.MAX_VALUE, 5), 1L, 1L), -1)

        // Apply spatial scale and normalize to [0, 1]
        val scaledX1 = x1.mul(spatialScale).div(featureW.toDouble())
        val scaledY1 = y1.mul(spatialScale).div(featureH.toDouble())
        val scaledX2 = x2.mul(spatialScale).div(featureW.toDouble())
        val scaledY2 = y2.mul(spatialScale).div(featureH.toDouble())

        // Stack boxes in [y1, x1, y2, x2] format as expected by cropAndResize
        val normalizedBoxes = sd.stack("${opName}_boxes", 1, scaledY1, scaledX1, scaledY2, scaledX2)

        // Use a sampling ratio for better approximation of max pooling
        // We crop at a higher resolution and then max pool down
        val samplingRatio = 2L
        val cropH = pooledH * samplingRatio
        val cropW = pooledW * samplingRatio
        val cropSize = sd.constant("${opName}_cropSize", Nd4j.createFromArray(cropH.toInt(), cropW.toInt()))

        // Crop and resize each ROI from the feature map
        val croppedRegions = sd.image().cropAndResize(
            "${opName}_crop",
            featureNhwc,
            normalizedBoxes,
            batchIndices,
            cropSize
        )

        // Apply max pooling to get final pooled output
        // croppedRegions shape: [num_rois, crop_h, crop_w, C]
        val poolConfig = Pooling2DConfig.builder()
            .kH(samplingRatio)
            .kW(samplingRatio)
            .sH(samplingRatio)
            .sW(samplingRatio)
            .pH(0)
            .pW(0)
            .isNHWC(true)
            .paddingMode(PaddingMode.VALID)
            .build()

        val maxPooled = sd.cnn().maxPooling2d("${opName}_maxpool", croppedRegions, poolConfig)

        // Convert back from NHWC to NCHW format: [num_rois, pooled_h, pooled_w, C] -> [num_rois, C, pooled_h, pooled_w]
        val output = sd.permute(outputNames[0], maxPooled, 0, 3, 1, 2)

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
