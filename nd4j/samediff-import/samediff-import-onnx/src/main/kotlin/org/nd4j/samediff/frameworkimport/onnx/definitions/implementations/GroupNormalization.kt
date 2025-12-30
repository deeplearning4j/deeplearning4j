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
 * Implementation of ONNX GroupNormalization operation.
 *
 * ONNX GroupNormalization spec: https://onnx.ai/onnx/operators/onnx__GroupNormalization.html
 *
 * Group Normalization normalizes the input tensor by first separating the channels into groups
 * and computing the mean and variance within each group. The normalization is then applied
 * as: y = scale * (x - mean) / sqrt(variance + epsilon) + bias
 *
 * When num_groups equals the number of channels, this is equivalent to Instance Normalization.
 * When num_groups equals 1, this is equivalent to Layer Normalization over all channels.
 *
 * Inputs:
 * - X: Input tensor of shape [N, C, ...] where N is batch, C is channels
 * - scale: Scale tensor of shape [C]
 * - bias: Bias tensor of shape [C]
 *
 * Attributes:
 * - epsilon: Small constant for numerical stability (default: 1e-5)
 * - num_groups: Number of groups to divide channels into
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["GroupNormalization"], frameworkName = "onnx")
class GroupNormalization : PreImportHook {

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

        // Get inputs: X, scale, bias
        val input = sd.getVariable(op.inputsToOp[0])
        val scale = sd.getVariable(op.inputsToOp[1])
        val bias = sd.getVariable(op.inputsToOp[2])

        // Get attributes
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()
        val numGroups = (attributes["num_groups"] as? Number)?.toInt()
            ?: throw IllegalArgumentException("GroupNormalization requires 'num_groups' attribute")

        // Get input shape dynamically
        val inputShape = sd.shape(input)

        // For a 4D input [N, C, H, W], we reshape to [N, G, C/G, H, W]
        // then normalize over [C/G, H, W] and reshape back.
        //
        // The general algorithm for any spatial dimensions:
        // 1. Reshape [N, C, *spatial] -> [N*G, C/G, *spatial]
        // 2. Apply instance norm (normalize over all but batch dim)
        // 3. Reshape back [N*G, C/G, *spatial] -> [N, C, *spatial]
        // 4. Apply scale and bias

        // Get N and C
        val n = sd.gather("${opName}_n", inputShape, sd.constant(0L), 0)
        val c = sd.gather("${opName}_c", inputShape, sd.constant(1L), 0)

        // Calculate channels per group
        val numGroupsVal = numGroups.toLong()
        val channelsPerGroup = sd.math.div("${opName}_cpg", c, sd.constant(numGroupsVal))

        // Create new batch dimension: N * num_groups
        val newBatch = sd.math.mul("${opName}_newBatch", n, sd.constant(numGroupsVal))

        // Build new shape: [N*G, C/G, spatial_dims...]
        // We need to extract spatial dimensions (indices 2 onwards)
        val inputRank = sd.rank(input)
        val spatialStart = sd.constant(2L)
        val spatialDimIndices = sd.range("${opName}_spatialIndices", spatialStart,
            inputRank.castTo(org.nd4j.linalg.api.buffer.DataType.INT64),
            sd.constant(1L),
            org.nd4j.linalg.api.buffer.DataType.INT64)

        // Gather spatial dimensions from input shape
        val spatialShapeExpanded = sd.expandDims("${opName}_spatialIndicesExp", spatialDimIndices, 1)
        val spatialShape = sd.gatherNd("${opName}_spatialShape", inputShape, spatialShapeExpanded)

        // Build reshaped dimensions: [N*G, C/G, spatial...]
        val reshapedPrefix = sd.stack("${opName}_reshapePrefix", 0, newBatch, channelsPerGroup)
        val reshapedShape = sd.concat("${opName}_reshapedShape", 0, reshapedPrefix, spatialShape)

        // Reshape input
        val reshaped = sd.reshape("${opName}_reshaped", input, reshapedShape)

        // Compute mean and variance over all dimensions except the first (batch*groups)
        // For reshaped tensor of shape [N*G, C/G, H, W], we normalize over dims 1, 2, 3
        val reshapedRank = sd.rank(reshaped)

        // Compute mean over dims 1 onwards, keeping dims for broadcasting
        val mean = sd.mean("${opName}_mean", reshaped, true, 1, 2, 3)
        val variance = sd.variance("${opName}_var", reshaped, false, true, 1, 2, 3)

        // Normalize: (x - mean) / sqrt(variance + epsilon)
        val centered = sd.math.sub("${opName}_centered", reshaped, mean)
        val stddev = sd.math.sqrt("${opName}_std", variance.add(epsilon))
        val normalized = sd.math.div("${opName}_normalized", centered, stddev)

        // Reshape back to original shape
        val ungrouped = sd.reshape("${opName}_ungrouped", normalized, inputShape)

        // Apply scale and bias
        // Scale and bias are [C], need to reshape to [1, C, 1, 1, ...] for broadcasting
        // Build broadcast shape based on input rank
        val onesVec = sd.onesLike("${opName}_onesVec", inputShape)
        val cIndex = sd.constant("${opName}_cIdx", longArrayOf(1))
        val cValue = sd.expandDims("${opName}_cVal", c, 0)
        val broadcastShape = sd.scatterNdUpdate("${opName}_bcastShape", onesVec,
            sd.expandDims(cIndex, 1), cValue)

        val scaleBroadcast = sd.reshape("${opName}_scaleBcast", scale, broadcastShape)
        val biasBroadcast = sd.reshape("${opName}_biasBcast", bias, broadcastShape)

        // y = scale * normalized + bias
        val scaled = sd.math.mul("${opName}_scaled", ungrouped, scaleBroadcast)
        val output = sd.math.add(outputNames[0], scaled, biasBroadcast)

        return mapOf(outputNames[0] to listOf(output))
    }
}
