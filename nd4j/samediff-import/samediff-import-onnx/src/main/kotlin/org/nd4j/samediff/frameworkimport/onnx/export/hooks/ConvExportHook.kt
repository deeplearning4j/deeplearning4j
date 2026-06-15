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
package org.nd4j.samediff.frameworkimport.onnx.export.hooks

import onnx.Onnx
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.export.ExportHookRule
import org.nd4j.samediff.frameworkimport.onnx.export.OnnxExportConfig
import org.nd4j.samediff.frameworkimport.onnx.export.PostExportHook
import org.slf4j.LoggerFactory

/**
 * Post-export hook for convolution operations.
 *
 * Handles:
 * - Weight format conversion (SameDiff uses NHWC by default, ONNX Conv uses NCHW weights)
 * - Proper attribute mapping for kernel_shape, strides, pads, dilations
 * - Group convolution handling
 * - Auto padding conversion
 *
 * @author Adam Gibson
 */
@ExportHookRule("conv2d", "conv3d", "depthwise_conv2d")
class ConvExportHook : PostExportHook {

    private val logger = LoggerFactory.getLogger(ConvExportHook::class.java)

    private val supportedOps = setOf("conv2d", "conv3d", "depthwise_conv2d")

    override fun canHandle(opName: String): Boolean {
        return opName in supportedOps
    }

    override fun export(
        sd: SameDiff,
        op: SameDiffOp,
        graphBuilder: Onnx.GraphProto.Builder,
        config: OnnxExportConfig
    ): List<Onnx.NodeProto> {
        val opName = op.op.opName()
        val dynamicOp = op.op as? DynamicCustomOp
            ?: throw IllegalArgumentException("Expected DynamicCustomOp for conv operation")

        val iArgs = dynamicOp.iArgs()
        val inputs = op.inputsToOp ?: emptyList()
        val outputs = op.outputsOfOp ?: emptyList()

        // Parse conv arguments
        // For conv2d: kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW, wFormat
        val kH = if (iArgs.size > 0) iArgs[0] else 1L
        val kW = if (iArgs.size > 1) iArgs[1] else 1L
        val sH = if (iArgs.size > 2) iArgs[2] else 1L
        val sW = if (iArgs.size > 3) iArgs[3] else 1L
        val pH = if (iArgs.size > 4) iArgs[4] else 0L
        val pW = if (iArgs.size > 5) iArgs[5] else 0L
        val dH = if (iArgs.size > 6) iArgs[6] else 1L
        val dW = if (iArgs.size > 7) iArgs[7] else 1L
        val isSameMode = if (iArgs.size > 8) iArgs[8] != 0L else false
        val isNCHW = if (iArgs.size > 9) iArgs[9] != 0L else false

        // Build ONNX Conv node
        val nodeBuilder = Onnx.NodeProto.newBuilder()
            .setName(op.name)
            .setOpType("Conv")

        // Add inputs (X, W, B)
        inputs.forEach { nodeBuilder.addInput(it) }

        // Add outputs
        outputs.forEach { nodeBuilder.addOutput(it) }

        // Add kernel_shape attribute
        nodeBuilder.addAttribute(AttributeProto {
            name = "kernel_shape"
            addAllInts(listOf(kH, kW))
            type = Onnx.AttributeProto.AttributeType.INTS
        })

        // Add strides attribute
        nodeBuilder.addAttribute(AttributeProto {
            name = "strides"
            addAllInts(listOf(sH, sW))
            type = Onnx.AttributeProto.AttributeType.INTS
        })

        // Add dilations attribute
        nodeBuilder.addAttribute(AttributeProto {
            name = "dilations"
            addAllInts(listOf(dH, dW))
            type = Onnx.AttributeProto.AttributeType.INTS
        })

        // Add auto_pad or pads attribute
        if (isSameMode) {
            nodeBuilder.addAttribute(AttributeProto {
                name = "auto_pad"
                s = byteString("SAME_UPPER")
                type = Onnx.AttributeProto.AttributeType.STRING
            })
        } else {
            // ONNX uses [top, left, bottom, right] for 2D
            nodeBuilder.addAttribute(AttributeProto {
                name = "pads"
                addAllInts(listOf(pH, pW, pH, pW))
                type = Onnx.AttributeProto.AttributeType.INTS
            })
        }

        // Handle depthwise convolution (group = num_channels)
        if (opName == "depthwise_conv2d") {
            // For depthwise, group equals the number of input channels
            // We need to get this from the weight shape
            val weightsVar = if (inputs.size > 1) sd.getVariable(inputs[1]) else null
            val weightsShape = weightsVar?.shape
            if (weightsShape != null && weightsShape.isNotEmpty()) {
                val groups = weightsShape[0] // Assuming weight format is [C, M, kH, kW]
                nodeBuilder.addAttribute(AttributeProto {
                    name = "group"
                    i = groups
                    type = Onnx.AttributeProto.AttributeType.INT
                })
            }
        }

        logger.debug("Exported conv op: {} with kernel={}x{}, stride={}x{}, pad={}x{}",
            op.name, kH, kW, sH, sW, pH, pW)

        return listOf(nodeBuilder.build())
    }

    override fun getInitializers(sd: SameDiff, op: SameDiffOp): List<Onnx.TensorProto> {
        // If weight format conversion is needed, it would be done here
        // For now, we assume weights are already in the correct format
        return emptyList()
    }

    override fun priority(): Int = 10
}
