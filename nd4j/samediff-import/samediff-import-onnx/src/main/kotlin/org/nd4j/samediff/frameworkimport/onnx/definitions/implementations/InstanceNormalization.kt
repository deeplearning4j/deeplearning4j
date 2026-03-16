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
 * Implementation of ONNX InstanceNormalization operation.
 *
 * ONNX InstanceNormalization spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
 *
 * Computes instance normalization as described in the paper:
 * https://arxiv.org/abs/1607.08022
 *
 * y = scale * (x - mean) / sqrt(variance + epsilon) + B
 *
 * where mean and variance are computed per instance per channel.
 * For input of shape [N, C, H, W], normalization is done over H and W dimensions.
 *
 * Inputs:
 * - input: Input tensor of shape [N, C, D1, D2, ..., Dn]
 * - scale: Scale tensor of shape [C]
 * - B: Bias tensor of shape [C]
 *
 * Attributes:
 * - epsilon: Small constant for numerical stability (default: 1e-5)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["InstanceNormalization"], frameworkName = "onnx")
class InstanceNormalization : PreImportHook {

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

        // Get inputs: input, scale, B
        val input = sd.getVariable(op.inputsToOp[0])
        val scale = sd.getVariable(op.inputsToOp[1])
        val bias = sd.getVariable(op.inputsToOp[2])

        // Get epsilon attribute (default: 1e-5)
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()

        // For 4D input [N, C, H, W], normalize over spatial dimensions (H, W)
        // Compute mean over spatial dimensions (keep dims for broadcasting)
        val mean = sd.mean("${opName}_mean", input, true, 2, 3)

        // Compute variance over spatial dimensions (unbiased=false, keepDims=true)
        val variance = sd.variance("${opName}_var", input, false, true, 2, 3)

        // Normalize: (x - mean) / sqrt(variance + epsilon)
        val centered = sd.math.sub("${opName}_centered", input, mean)
        val stddev = sd.math.sqrt("${opName}_std", variance.add(epsilon))
        val normalized = sd.math.div("${opName}_normalized", centered, stddev)

        // Apply scale and bias - reshape from [C] to [1, C, 1, 1] for broadcasting
        val inputShape = input.shape ?: throw IllegalStateException("InstanceNormalization requires static input shape")
        val c = inputShape[1].toInt()
        val shapeArr = longArrayOf(1, c.toLong(), 1, 1)

        val scaleReshaped = sd.reshape("${opName}_scaleReshaped", scale, *shapeArr)
        val biasReshaped = sd.reshape("${opName}_biasReshaped", bias, *shapeArr)

        // y = scale * normalized + bias
        val scaled = sd.math.mul("${opName}_scaled", normalized, scaleReshaped)
        val output = sd.math.add(outputNames[0], scaled, biasReshaped)

        return mapOf(outputNames[0] to listOf(output))
    }
}
