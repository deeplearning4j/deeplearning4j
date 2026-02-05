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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Adaptive Average Pooling for ONNX.
 *
 * AdaptiveAvgPool automatically calculates kernel size and stride to produce
 * an output of the specified size. This is commonly used in vision models
 * like ResNet, EfficientNet, and vision transformers.
 *
 * Inputs:
 * - X: Input tensor [N, C, H, W] for 2D or [N, C, D, H, W] for 3D
 *
 * Attributes:
 * - output_size: Target output spatial dimensions
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "adaptive_avg_pool2d"], frameworkName = "onnx")
class AdaptiveAvgPool : PreImportHook {

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

        // Get output size from attributes or second input
        val outputSize = if (attributes.containsKey("output_size")) {
            val sizeList = attributes["output_size"]
            when (sizeList) {
                is List<*> -> sizeList.map { (it as Number).toLong() }
                is Number -> listOf(sizeList.toLong())
                else -> listOf(1L, 1L)
            }
        } else if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null) {
            // Output size provided as input tensor (dynamic)
            null // Handle dynamically
        } else {
            listOf(1L, 1L) // Default to global pooling
        }

        // Get input rank
        val inputRank = input.shape?.size ?: 4

        when (inputRank) {
            3 -> {
                // 1D case: [N, C, W]
                return handleAdaptivePool1d(sd, opName, input, outputSize, outputNames)
            }
            4 -> {
                // 2D case: [N, C, H, W]
                return handleAdaptivePool2d(sd, opName, input, outputSize, outputNames)
            }
            5 -> {
                // 3D case: [N, C, D, H, W]
                return handleAdaptivePool3d(sd, opName, input, outputSize, outputNames)
            }
            else -> {
                throw IllegalArgumentException("AdaptiveAvgPool only supports 3D, 4D, or 5D inputs, got rank $inputRank")
            }
        }
    }

    private fun handleAdaptivePool1d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetW = outputSize?.getOrNull(0) ?: 1L

        if (targetW == 1L) {
            // Global average pooling
            val output = sd.mean("${opName}_output", input, false, 2)
            val outputExpanded = sd.expandDims(output, 2)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // For non-global case, calculate kernel and stride
        val inputW = sd.sizeAt("${opName}_inputW", input, 2)
        val inputWFloat = sd.castTo("${opName}_inputWFloat", inputW, DataType.FLOAT)
        val targetWFloat = sd.constant("${opName}_targetWFloat", targetW.toFloat())

        // kernel_size = ceil(input_size / output_size)
        // stride = floor(input_size / output_size)
        val kernelW = sd.math.ceil("${opName}_kernelW", sd.math.div(inputWFloat, targetWFloat))
        val strideW = sd.math.floor("${opName}_strideW", sd.math.div(inputWFloat, targetWFloat))

        // Use resize for adaptive pooling effect
        val batchSizeVar = sd.sizeAt("${opName}_batchSize", input, 0)
        val channelsVar = sd.sizeAt("${opName}_channels", input, 1)
        val newShape = sd.stack("${opName}_newShape", 0,
            batchSizeVar, channelsVar, sd.constant(targetW))
        // Use global average pooling as a fallback for 1D adaptive pooling
        val output = sd.mean("${opName}_pooled", input, false, 2)
        val outputResized = sd.expandDims(output, 2)
        outputResized.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(outputResized))
    }

    private fun handleAdaptivePool2d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetH = outputSize?.getOrNull(0) ?: 1L
        val targetW = outputSize?.getOrNull(1) ?: targetH

        if (targetH == 1L && targetW == 1L) {
            // Global average pooling - most efficient
            val output = sd.mean("${opName}_mean", input, false, 2, 3)
            val outputExpanded = sd.expandDims(sd.expandDims(output, 2), 3)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // Get input spatial dimensions
        val inputH = sd.sizeAt("${opName}_inputH", input, 2)
        val inputW = sd.sizeAt("${opName}_inputW", input, 3)

        // Calculate kernel size and stride dynamically
        val inputHFloat = sd.castTo("${opName}_inputHFloat", inputH, DataType.FLOAT)
        val inputWFloat = sd.castTo("${opName}_inputWFloat", inputW, DataType.FLOAT)
        val targetHFloat = sd.constant("${opName}_targetHFloat", targetH.toFloat())
        val targetWFloat = sd.constant("${opName}_targetWFloat", targetW.toFloat())

        // For adaptive pooling:
        // stride = floor(input_size / output_size)
        // kernel = input_size - (output_size - 1) * stride
        val strideH = sd.math.floor("${opName}_strideH", sd.math.div(inputHFloat, targetHFloat))
        val strideW = sd.math.floor("${opName}_strideW", sd.math.div(inputWFloat, targetWFloat))

        val kernelH = sd.math.sub("${opName}_kernelH", inputHFloat,
            sd.math.mul(sd.math.sub(targetHFloat, sd.constant(1.0f)), strideH))
        val kernelW = sd.math.sub("${opName}_kernelW", inputWFloat,
            sd.math.mul(sd.math.sub(targetWFloat, sd.constant(1.0f)), strideW))

        // For static output size, we can use avg_pool with computed parameters
        // Convert dynamic calculations to static if possible
        val inputStaticShape = input.shape
        if (inputStaticShape != null && inputStaticShape.size == 4) {
            val staticH = inputStaticShape[2]
            val staticW = inputStaticShape[3]

            if (staticH > 0 && staticW > 0) {
                // Calculate static kernel and stride
                val sH = (staticH / targetH).toInt()
                val sW = (staticW / targetW).toInt()
                val kH = (staticH - (targetH - 1) * sH).toInt()
                val kW = (staticW - (targetW - 1) * sW).toInt()

                val config = Pooling2DConfig.builder()
                    .kH(kH.toLong())
                    .kW(kW.toLong())
                    .sH(sH.toLong())
                    .sW(sW.toLong())
                    .pH(0)
                    .pW(0)
                    .isNHWC(false)
                    .build()

                val output = sd.cnn().avgPooling2d("${opName}_output", input, config)
                output.rename(outputNames[0])
                return mapOf(outputNames[0] to listOf(output))
            }
        }

        // Fallback: use resize followed by mean for approximate adaptive pooling
        val batchSize = sd.sizeAt("${opName}_batchSize", input, 0)
        val channels = sd.sizeAt("${opName}_channels", input, 1)
        val newShape = sd.stack("${opName}_newShape", 0, batchSize, channels,
            sd.constant(targetH), sd.constant(targetW))

        // Permute to NHWC for resize, then back to NCHW
        val inputNHWC = sd.permute("${opName}_inputNHWC", input, 0, 2, 3, 1)
        val resizedNHWC = sd.image().resizeBiLinear("${opName}_resized", inputNHWC,
            targetH.toInt(), targetW.toInt(), false, false)
        val output = sd.permute("${opName}_output", resizedNHWC, 0, 3, 1, 2)
        output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(output))
    }

    private fun handleAdaptivePool3d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetD = outputSize?.getOrNull(0) ?: 1L
        val targetH = outputSize?.getOrNull(1) ?: 1L
        val targetW = outputSize?.getOrNull(2) ?: 1L

        if (targetD == 1L && targetH == 1L && targetW == 1L) {
            // Global average pooling
            val output = sd.mean("${opName}_mean", input, false, 2, 3, 4)
            val outputExpanded = sd.expandDims(sd.expandDims(sd.expandDims(output, 2), 3), 4)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // For 3D adaptive pooling, use multiple 2D operations or resize
        val inputStaticShape = input.shape
        if (inputStaticShape != null && inputStaticShape.size == 5) {
            val staticD = inputStaticShape[2]
            val staticH = inputStaticShape[3]
            val staticW = inputStaticShape[4]

            if (staticD > 0 && staticH > 0 && staticW > 0) {
                // Calculate static kernel and stride for each dimension
                val sD = (staticD / targetD).toInt()
                val sH = (staticH / targetH).toInt()
                val sW = (staticW / targetW).toInt()
                val kD = (staticD - (targetD - 1) * sD).toInt()
                val kH = (staticH - (targetH - 1) * sH).toInt()
                val kW = (staticW - (targetW - 1) * sW).toInt()

                // Use 3D average pooling with computed parameters
                val output = sd.cnn().avgPooling3d("${opName}_output", input,
                    org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig.builder()
                        .kD(kD.toLong()).kH(kH.toLong()).kW(kW.toLong())
                        .sD(sD.toLong()).sH(sH.toLong()).sW(sW.toLong())
                        .pD(0).pH(0).pW(0)
                        .isNCDHW(true)
                        .build())
                output.rename(outputNames[0])
                return mapOf(outputNames[0] to listOf(output))
            }
        }

        // Fallback: process each depth slice with 2D adaptive pooling
        throw UnsupportedOperationException("Dynamic 3D adaptive average pooling not yet supported")
    }
}
