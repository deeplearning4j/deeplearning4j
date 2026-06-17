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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.enums.ImageResizeMethod
import org.nd4j.linalg.factory.Nd4j

/**
 * Implementation of ONNX Upsample operation (deprecated but still used).
 *
 * Upsample resizes the input tensor. It's deprecated in favor of Resize
 * but still appears in older models.
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample
 *
 * Inputs:
 * - X: Input tensor [N, C, H, W] or [N, C, D, H, W]
 * - scales: Scale factors for each dimension
 *
 * Attributes:
 * - mode: Interpolation mode: "nearest" or "linear" (default: "nearest")
 *
 * Output:
 * - Y: Upsampled tensor
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Upsample"], frameworkName = "onnx")
class Upsample : PreImportHook {

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

        // Get scales (second input in opset >= 9, attribute in older versions)
        val scales: FloatArray = if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null && op.inputsToOp[1].isNotEmpty()) {
            getScalesFromInput(sd, dynamicVariables, op.inputsToOp[1])
        } else {
            getScalesFromAttribute(attributes)
        }

        // Get mode attribute
        val mode = (attributes.getOrDefault("mode", "nearest") as? String) ?: "nearest"

        val inputShape = input.shape
        val rank = inputShape?.size ?: 4

        if (rank == 4) {
            // 2D case: [N, C, H, W]
            return upsample2D(sd, opName, input, scales, mode, outputNames)
        } else if (rank == 5) {
            // 3D case: [N, C, D, H, W]
            return upsample3D(sd, opName, input, scales, mode, outputNames)
        } else {
            // Fallback - just return input
            val output = sd.identity("${opName}_output", input)
            output.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        }
    }

    private fun upsample2D(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        scales: FloatArray,
        mode: String,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val inputShape = input.shape ?: return mapOf(outputNames[0] to listOf(input))

        val batchSize = inputShape[0]
        val channels = inputShape[1]
        val height = inputShape[2]
        val width = inputShape[3]

        // Calculate output size
        val scaleH = scales.getOrElse(2) { 1.0f }
        val scaleW = scales.getOrElse(3) { 1.0f }
        val outH = (height * scaleH).toInt()
        val outW = (width * scaleW).toInt()

        val output = when (mode.lowercase()) {
            "nearest" -> {
                // Permute to NHWC for resize
                val nhwc = sd.permute("${opName}_nhwc", input, 0, 2, 3, 1)
                val sizeVar = sd.constant("${opName}_size", Nd4j.createFromArray(outH, outW))
                val resized = sd.image().imageResize("${opName}_resize", nhwc, sizeVar, ImageResizeMethod.ResizeNearest)
                sd.permute("${opName}_nchw", resized, 0, 3, 1, 2)
            }
            "linear", "bilinear" -> {
                val nhwc = sd.permute("${opName}_nhwc", input, 0, 2, 3, 1)
                val resized = sd.image().resizeBiLinear("${opName}_resize", nhwc, outH, outW, false, false)
                sd.permute("${opName}_nchw", resized, 0, 3, 1, 2)
            }
            else -> {
                // Default to nearest
                val nhwc = sd.permute("${opName}_nhwc", input, 0, 2, 3, 1)
                val sizeVar = sd.constant("${opName}_size_default", Nd4j.createFromArray(outH, outW))
                val resized = sd.image().imageResize("${opName}_resize", nhwc, sizeVar, ImageResizeMethod.ResizeNearest)
                sd.permute("${opName}_nchw", resized, 0, 3, 1, 2)
            }
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun upsample3D(
        sd: SameDiff,
        opName: String,
        input: SDVariable,
        scales: FloatArray,
        mode: String,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        val inputShape = input.shape ?: return mapOf(outputNames[0] to listOf(input))

        // For 3D, process each depth slice
        val scaleD = scales.getOrElse(2) { 1.0f }
        val scaleH = scales.getOrElse(3) { 1.0f }
        val scaleW = scales.getOrElse(4) { 1.0f }

        val depth = inputShape[2]
        val height = inputShape[3]
        val width = inputShape[4]

        val outD = (depth * scaleD).toInt()
        val outH = (height * scaleH).toInt()
        val outW = (width * scaleW).toInt()

        // Simplified: process depth dimension separately
        // First upsample HW, then D
        val reshaped = sd.reshape("${opName}_reshape", input,
            inputShape[0], inputShape[1] * depth, height, width)

        val nhwc = sd.permute("${opName}_nhwc", reshaped, 0, 2, 3, 1)
        val resizedHW = if (mode == "nearest") {
            val sizeVar = sd.constant("${opName}_sizeHW", Nd4j.createFromArray(outH, outW))
            sd.image().imageResize("${opName}_resizeHW", nhwc, sizeVar, ImageResizeMethod.ResizeNearest)
        } else {
            sd.image().resizeBiLinear("${opName}_resizeHW", nhwc, outH, outW, false, false)
        }
        val backToNCHW = sd.permute("${opName}_nchw", resizedHW, 0, 3, 1, 2)

        // Reshape and resize depth
        val unflattened = sd.reshape("${opName}_unflatten", backToNCHW,
            inputShape[0], inputShape[1], depth, outH.toLong(), outW.toLong())

        // For depth dimension, use tile or repeat
        val output = if (scaleD > 1.0f) {
            val repeatFactor = scaleD.toInt()
            // Repeat along depth dimension
            val permuted = sd.permute("${opName}_permD", unflattened, 0, 1, 3, 4, 2)  // [N,C,H,W,D]
            val tiled = sd.tile("${opName}_tileD", permuted, 1, 1, 1, 1, repeatFactor)
            sd.permute("${opName}_output", tiled, 0, 1, 4, 2, 3)  // [N,C,D*scale,H,W]
        } else {
            unflattened
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun getScalesFromInput(
        sd: SameDiff,
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): FloatArray {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto
        if (tensorProto != null) {
            return when {
                tensorProto.floatDataCount > 0 -> tensorProto.floatDataList.toFloatArray()
                tensorProto.rawData.size() > 0 -> {
                    val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                    buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    val floatBuffer = buffer.asFloatBuffer()
                    FloatArray(floatBuffer.remaining()) { floatBuffer.get() }
                }
                else -> floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
            }
        }
        return floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
    }

    private fun getScalesFromAttribute(attributes: Map<String, Any>): FloatArray {
        val scalesAttr = attributes["scales"]
        return when (scalesAttr) {
            is FloatArray -> scalesAttr
            is List<*> -> scalesAttr.map { (it as Number).toFloat() }.toFloatArray()
            else -> floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
        }
    }
}
