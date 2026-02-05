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

import onnx.Onnx
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
 * Implementation of ONNX Col2Im operation using native libnd4j col2im op.
 *
 * Col2Im is the inverse of Im2Col. It rearranges column blocks back into
 * a multidimensional image format. This is commonly used in:
 * - Deformable convolutions
 * - Image reconstruction from patches
 * - Vision transformers (ViT) for patch unfolding
 *
 * Uses native sd.cnn().col2Im() which maps to libnd4j's col2im op.
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Col2Im
 *
 * Inputs:
 * - input: Input tensor [N, C * block_size, L] where L is the number of blocks
 * - image_shape: Shape of output image spatial dimensions [H, W]
 * - block_shape: Size of blocks [kH, kW]
 *
 * Attributes:
 * - dilations: Dilation factors for each spatial dimension (default: all 1)
 * - pads: Padding for each spatial dimension [start, end, ...] (default: all 0)
 * - strides: Stride factors for each spatial dimension (default: all 1)
 *
 * Output:
 * - output: Image tensor [N, C, H, W]
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Col2Im"], frameworkName = "onnx")
class Col2Im : PreImportHook {

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
        val dilations = getListAttribute(attributes, "dilations", listOf(1L, 1L))
        val pads = getListAttribute(attributes, "pads", listOf(0L, 0L, 0L, 0L))
        val strides = getListAttribute(attributes, "strides", listOf(1L, 1L))

        // Get static shapes for image_shape and block_shape
        val imageShapeStatic = getStaticShape(dynamicVariables, op.inputsToOp[1])
        val blockShapeStatic = getStaticShape(dynamicVariables, op.inputsToOp[2])

        // Input shape: [N, C * kH * kW, L]
        val inputShape = input.shape
        val batchSize = inputShape?.get(0) ?: -1L
        val numChannelsTimesBlock = inputShape?.get(1) ?: -1L
        val numBlocks = inputShape?.get(2) ?: -1L

        if (imageShapeStatic != null && blockShapeStatic != null && blockShapeStatic.size == 2) {
            // 2D case with static shapes - use native col2im operation
            val outH = imageShapeStatic[0]
            val outW = imageShapeStatic[1]
            val kH = blockShapeStatic[0].toInt()
            val kW = blockShapeStatic[1].toInt()

            val sH = strides.getOrElse(0) { 1L }.toInt()
            val sW = strides.getOrElse(1) { 1L }.toInt()
            val dH = dilations.getOrElse(0) { 1L }.toInt()
            val dW = dilations.getOrElse(1) { 1L }.toInt()
            val pH = pads.getOrElse(0) { 0L }.toInt()
            val pW = pads.getOrElse(1) { 0L }.toInt()

            val numChannels = numChannelsTimesBlock / (kH * kW)

            // Calculate output spatial dimensions for col2im
            val colH = (outH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
            val colW = (outW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

            // ONNX input is [N, C * kH * kW, L] where L = colH * colW
            // Native col2im expects [N, C, kH, kW, colH, colW]
            // Reshape input to match native format
            val reshaped = sd.reshape("${opName}_reshaped", input,
                batchSize, numChannels, kH.toLong(), kW.toLong(), colH, colW)

            // Build Conv2DConfig for native col2im op
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

            // Use native col2im op via sd.cnn().col2Im()
            // Output shape: [N, C, H, W]
            val output = sd.cnn().col2Im("${opName}_output", reshaped, config)

            output.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        } else {
            // Dynamic case or 3D - fallback to reshape-based approach
            // This handles cases where shapes are not known at graph construction time
            return col2imDynamic(sd, opName, input, op, dynamicVariables, strides, dilations, pads, outputNames)
        }
    }

    /**
     * Fallback implementation for dynamic shapes.
     */
    private fun col2imDynamic(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        op: SameDiffOp,
        dynamicVariables: Map<String, GeneratedMessageV3>,
        strides: List<Long>,
        dilations: List<Long>,
        pads: List<Long>,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val imageShape = sd.getVariable(op.inputsToOp[1])
        val blockShape = sd.getVariable(op.inputsToOp[2])

        val kHVar = sd.sizeAt("${opName}_kH", blockShape, 0)
        val kWVar = sd.sizeAt("${opName}_kW", blockShape, 1)
        val outHVar = sd.sizeAt("${opName}_outH", imageShape, 0)
        val outWVar = sd.sizeAt("${opName}_outW", imageShape, 1)

        val sH = sd.constant("${opName}_sH", strides.getOrElse(0) { 1L })
        val sW = sd.constant("${opName}_sW", strides.getOrElse(1) { 1L })

        // Calculate output blocks
        val numBlocksH = sd.math.floorDiv(outHVar, sH)
        val numBlocksW = sd.math.floorDiv(outWVar, sW)

        // Reshape and reconstruct
        val numChannels = sd.math.floorDiv(
            sd.sizeAt("${opName}_inputDim1", input, 1),
            sd.math.mul(kHVar, kWVar)
        )

        // Reshape: [N, C*kH*kW, L] -> [N, C, kH, kW, nH, nW]
        val inputDim0 = sd.sizeAt("${opName}_inputDim0", input, 0)
        val reshapeShape1 = sd.stack("${opName}_reshapeShape1", 0,
            inputDim0, numChannels, kHVar, kWVar, numBlocksH, numBlocksW)
        val reshaped = sd.reshape("${opName}_reshaped", input, reshapeShape1)

        // Transpose: [N, C, kH, kW, nH, nW] -> [N, C, nH, kH, nW, kW]
        val transposed = sd.permute("${opName}_transposed", reshaped, 0, 1, 4, 2, 5, 3)

        // Reshape to merge spatial dims: [N, C, nH*kH, nW*kW]
        val reshapeShape2 = sd.stack("${opName}_reshapeShape2", 0,
            inputDim0, numChannels, sd.math.mul(numBlocksH, kHVar), sd.math.mul(numBlocksW, kWVar))
        val output = sd.reshape("${opName}_output", transposed, reshapeShape2)

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

    private fun getStaticShape(dynamicVariables: Map<String, GeneratedMessageV3>, varName: String): LongArray? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.int64DataCount > 0 -> tensorProto.int64DataList.map { it }.toLongArray()
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val longBuffer = buffer.asLongBuffer()
                LongArray(longBuffer.remaining()) { longBuffer.get() }
            }
            else -> null
        }
    }
}
