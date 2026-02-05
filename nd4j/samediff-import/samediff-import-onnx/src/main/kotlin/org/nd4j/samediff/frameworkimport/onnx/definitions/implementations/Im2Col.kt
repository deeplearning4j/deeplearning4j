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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Im2Col operation using native libnd4j im2col op.
 *
 * Im2Col extracts sliding local blocks from a batched input tensor.
 * This is the inverse of Col2Im and is commonly used in:
 * - Implementing convolutions as matrix multiplications
 * - Vision transformers for patch extraction
 * - Deformable convolutions
 *
 * Uses native sd.cnn().im2Col() which maps to libnd4j's im2col op.
 *
 * Inputs:
 * - input: Input tensor [N, C, H, W]
 *
 * Attributes:
 * - kernel_shape: Sizes of the sliding blocks [kH, kW]
 * - dilations: Dilation factors (default: all 1)
 * - pads: Padding (default: all 0)
 * - strides: Stride factors (default: all 1)
 *
 * Output:
 * - output: Tensor [N, C, kH, kW, outH, outW] from native op,
 *           reshaped to [N, C * kH * kW, L] for ONNX compatibility
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Im2Col", "im2col", "Unfold"], frameworkName = "onnx")
class Im2Col : PreImportHook {

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

        val input = sd.getVariable(op.inputsToOp[0])

        // Get attributes
        val kernelShape = getListAttribute(attributes, "kernel_shape", listOf(3L, 3L))
        val dilations = getListAttribute(attributes, "dilations", listOf(1L, 1L))
        val pads = getListAttribute(attributes, "pads", listOf(0L, 0L, 0L, 0L))
        val strides = getListAttribute(attributes, "strides", listOf(1L, 1L))

        val kH = kernelShape[0].toInt()
        val kW = kernelShape[1].toInt()
        val sH = strides.getOrElse(0) { 1L }.toInt()
        val sW = strides.getOrElse(1) { 1L }.toInt()
        val dH = dilations.getOrElse(0) { 1L }.toInt()
        val dW = dilations.getOrElse(1) { 1L }.toInt()
        val pH = pads.getOrElse(0) { 0L }.toInt()
        val pW = pads.getOrElse(1) { 0L }.toInt()

        // Build Conv2DConfig for native im2col op
        val config = Conv2DConfig.builder()
            .kH(kH.toLong())
            .kW(kW.toLong())
            .sH(sH.toLong())
            .sW(sW.toLong())
            .pH(pH.toLong())
            .pW(pW.toLong())
            .dH(dH.toLong())
            .dW(dW.toLong())
            .paddingMode(PaddingMode.VALID)
            .build()

        // Use native im2col op via sd.cnn().im2Col()
        // Output shape: [N, C, kH, kW, outH, outW]
        val im2colResult = sd.cnn().im2Col("${opName}_im2col", input, config)

        // Get input shape for reshaping
        val inputShape = input.shape
        val batchSize = inputShape?.get(0) ?: -1L
        val numChannels = inputShape?.get(1) ?: -1L
        val inputH = inputShape?.get(2) ?: -1L
        val inputW = inputShape?.get(3) ?: -1L

        // Calculate output spatial dimensions
        val outH = (inputH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
        val outW = (inputW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

        // ONNX expects output shape [N, C * kH * kW, L] where L = outH * outW
        // Native im2col gives [N, C, kH, kW, outH, outW]
        // Need to reshape
        val blockChannels = numChannels * kH * kW
        val numBlocks = outH * outW

        // Reshape to [N, C*kH*kW, outH*outW]
        val output = sd.reshape("${opName}_output", im2colResult,
            batchSize, blockChannels, numBlocks)

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun getListAttribute(attributes: Map<String, Any>, name: String, default: List<Long>): List<Long> {
        val value = attributes[name] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toLong() }
            is LongArray -> value.toList()
            is IntArray -> value.map { it.toLong() }
            else -> default
        }
    }
}
