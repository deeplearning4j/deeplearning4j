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
 * Implementation of Adaptive Max Pooling for ONNX.
 *
 * AdaptiveMaxPool automatically calculates kernel size and stride to produce
 * an output of the specified size. This is commonly used in vision models
 * like ResNet, EfficientNet, and OCR backbones.
 *
 * Inputs:
 * - X: Input tensor [N, C, H, W] for 2D or [N, C, D, H, W] for 3D
 *
 * Attributes:
 * - output_size: Target output spatial dimensions
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "adaptive_max_pool2d"], frameworkName = "onnx")
class AdaptiveMaxPool : PreImportHook {

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
            null // Handle dynamically
        } else {
            listOf(1L, 1L) // Default to global pooling
        }

        // Get input shape
        val inputShape = sd.shape(input).rename("${opName}_inputShape")
        val inputRank = input.shape?.size ?: 4

        when (inputRank) {
            3 -> {
                // 1D case: [N, C, W]
                return handleAdaptiveMaxPool1d(sd, opName, input, inputShape, outputSize, outputNames)
            }
            4 -> {
                // 2D case: [N, C, H, W]
                return handleAdaptiveMaxPool2d(sd, opName, input, inputShape, outputSize, outputNames)
            }
            5 -> {
                // 3D case: [N, C, D, H, W]
                return handleAdaptiveMaxPool3d(sd, opName, input, inputShape, outputSize, outputNames)
            }
            else -> {
                throw IllegalArgumentException("AdaptiveMaxPool only supports 3D, 4D, or 5D inputs, got rank $inputRank")
            }
        }
    }

    private fun handleAdaptiveMaxPool1d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        inputShape: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetW = outputSize?.getOrNull(0) ?: 1L

        if (targetW == 1L) {
            // Global max pooling
            val output = sd.max("${opName}_output", input, false, 2)
            val outputExpanded = sd.expandDims(output, 2)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // For non-global case, we need to compute kernel and stride
        val inputStaticShape = input.shape
        if (inputStaticShape != null && inputStaticShape.size == 3 && inputStaticShape[2] > 0) {
            val staticW = inputStaticShape[2]
            val sW = (staticW / targetW).toInt()
            val kW = (staticW - (targetW - 1) * sW).toInt()

            // Use 1D max pooling with calculated parameters
            val inputExpanded = sd.expandDims("${opName}_expanded", input, 2) // [N, C, 1, W]

            val config = Pooling2DConfig.builder()
                .kH(1)
                .kW(kW.toLong())
                .sH(1)
                .sW(sW.toLong())
                .pH(0)
                .pW(0)
                .isNHWC(false)
                .build()

            val pooled = sd.cnn().maxPooling2d("${opName}_pooled", inputExpanded, config)
            val output = sd.squeeze("${opName}_output", pooled, 2)
            output.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        }

        throw UnsupportedOperationException("Dynamic 1D adaptive max pooling not yet supported")
    }

    private fun handleAdaptiveMaxPool2d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        inputShape: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetH = outputSize?.getOrNull(0) ?: 1L
        val targetW = outputSize?.getOrNull(1) ?: targetH

        if (targetH == 1L && targetW == 1L) {
            // Global max pooling - most efficient
            val output = sd.max("${opName}_max", input, false, 2, 3)
            val outputExpanded = sd.expandDims(sd.expandDims(output, 2), 3)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // For static shapes, calculate kernel and stride
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

                val output = sd.cnn().maxPooling2d("${opName}_output", input, config)
                output.rename(outputNames[0])

                val results = mutableMapOf(outputNames[0] to listOf(output))

                // Handle indices output if requested
                if (outputNames.size > 1) {
                    // For max pooling with indices, we'd need a different approach
                    // For now, return dummy indices
                    val indices = sd.zero("${opName}_indices", DataType.INT64,
                        output.shape!![0], output.shape!![1], targetH, targetW)
                    indices.rename(outputNames[1])
                    results[outputNames[1]] = listOf(indices)
                }

                return results
            }
        }

        throw UnsupportedOperationException("Dynamic 2D adaptive max pooling not yet supported")
    }

    private fun handleAdaptiveMaxPool3d(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        inputShape: SDVariable,
        outputSize: List<Long>?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val targetD = outputSize?.getOrNull(0) ?: 1L
        val targetH = outputSize?.getOrNull(1) ?: 1L
        val targetW = outputSize?.getOrNull(2) ?: 1L

        if (targetD == 1L && targetH == 1L && targetW == 1L) {
            // Global max pooling
            val output = sd.max("${opName}_max", input, false, 2, 3, 4)
            val outputExpanded = sd.expandDims(sd.expandDims(sd.expandDims(output, 2), 3), 4)
            outputExpanded.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(outputExpanded))
        }

        // For 3D adaptive max pooling with static shapes
        val inputStaticShape = input.shape
        if (inputStaticShape != null && inputStaticShape.size == 5) {
            val staticD = inputStaticShape[2]
            val staticH = inputStaticShape[3]
            val staticW = inputStaticShape[4]

            if (staticD > 0 && staticH > 0 && staticW > 0) {
                val sD = (staticD / targetD).toInt()
                val sH = (staticH / targetH).toInt()
                val sW = (staticW / targetW).toInt()
                val kD = (staticD - (targetD - 1) * sD).toInt()
                val kH = (staticH - (targetH - 1) * sH).toInt()
                val kW = (staticW - (targetW - 1) * sW).toInt()

                val output = sd.cnn().maxPooling3d("${opName}_output", input,
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

        throw UnsupportedOperationException("Dynamic 3D adaptive max pooling not yet supported")
    }
}
