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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX ConvTranspose operation.
 *
 * ONNX ConvTranspose spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
 *
 * Transposed convolution (also known as deconvolution or fractionally-strided convolution).
 *
 * Inputs:
 * - X: Input tensor of shape [N, C, H, W] (or [N, C, D, H, W] for 3D)
 * - W: Weight tensor of shape [C, M/group, kH, kW]
 * - B: Optional bias tensor of shape [M]
 *
 * Attributes:
 * - auto_pad: Padding mode ("NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID")
 * - dilations: Dilation values
 * - group: Number of groups
 * - kernel_shape: Kernel size
 * - output_padding: Additional padding for output
 * - output_shape: Explicit output shape
 * - pads: Padding values [top, left, bottom, right]
 * - strides: Stride values
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["ConvTranspose"], frameworkName = "onnx")
class ConvTranspose : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])
        val weights = sd.getVariable(op.inputsToOp[1])
        val bias = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null

        // Get attributes with defaults
        val autoPad = attributes.getOrDefault("auto_pad", "NOTSET") as? String ?: "NOTSET"
        val group = (attributes.getOrDefault("group", 1L) as Number).toInt()

        // Get strides (default: [1, 1])
        val strides = getIntList(attributes, "strides", listOf(1, 1))
        val sH = strides.getOrElse(0) { 1 }
        val sW = strides.getOrElse(1) { 1 }

        // Get dilations (default: [1, 1])
        val dilations = getIntList(attributes, "dilations", listOf(1, 1))
        val dH = dilations.getOrElse(0) { 1 }
        val dW = dilations.getOrElse(1) { 1 }

        // Get pads (default: [0, 0, 0, 0])
        val pads = getIntList(attributes, "pads", listOf(0, 0, 0, 0))
        val pH = pads.getOrElse(0) { 0 }
        val pW = pads.getOrElse(1) { 0 }

        // Get kernel shape (optional, can be inferred from weights)
        val kernelShape = getIntList(attributes, "kernel_shape", listOf())
        val kH = kernelShape.getOrElse(0) { 0 }
        val kW = kernelShape.getOrElse(1) { 0 }

        // Get output padding (default: [0, 0])
        val outputPadding = getIntList(attributes, "output_padding", listOf(0, 0))

        // Determine if same mode padding
        val isSameMode = autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER"

        // Build deconv config
        val config = DeConv2DConfig.builder()
            .kH(kH.toLong())
            .kW(kW.toLong())
            .sH(sH.toLong())
            .sW(sW.toLong())
            .pH(pH.toLong())
            .pW(pW.toLong())
            .dH(dH.toLong())
            .dW(dW.toLong())
            .isSameMode(isSameMode)
            .dataFormat("NCHW")
            .build()

        // Perform deconvolution
        val output = if (bias != null) {
            val deconv = sd.cnn.deconv2d("${opName}_deconv", input, weights, config)
            // Add bias - reshape bias from [C] to [1, C, 1, 1] for broadcasting
            val biasReshaped = sd.reshape("${opName}_biasReshaped", bias, 1, -1, 1, 1)
            sd.math.add(outputNames[0], deconv, biasReshaped)
        } else {
            sd.cnn.deconv2d(outputNames[0], input, weights, config)
        }

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
