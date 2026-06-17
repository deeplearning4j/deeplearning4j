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
 * Implementation of Microsoft ONNX NhwcConv operation.
 *
 * NhwcConv performs convolution with NHWC (channels-last) format.
 * This is the same as regular Conv but specifically for NHWC data format.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#NhwcConv
 *
 * Inputs:
 * - X: Input tensor (NHWC format)
 * - W: Weight tensor (OHWI format for NHWC)
 * - B: Optional bias tensor
 *
 * Attributes:
 * - auto_pad: Auto padding mode
 * - dilations: Dilation values
 * - group: Number of groups
 * - kernel_shape: Kernel shape
 * - pads: Padding values
 * - strides: Stride values
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["NhwcConv"], frameworkName = "onnx")
class NhwcConv : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input = sd.getVariable(op.inputsToOp[0])
        val weights = sd.getVariable(op.inputsToOp[1])

        // Optional bias
        val hasBias = op.inputsToOp.size > 2 && op.inputsToOp[2] != null
        val bias = if (hasBias) sd.getVariable(op.inputsToOp[2]) else null

        // Get convolution attributes
        val strides = getIntListAttribute(attributes, "strides", listOf(1, 1))
        val dilations = getIntListAttribute(attributes, "dilations", listOf(1, 1))
        val pads = getIntListAttribute(attributes, "pads", listOf(0, 0, 0, 0))
        val group = (attributes.getOrDefault("group", 1) as Number).toInt()
        val kernelShape = getIntListAttribute(attributes, "kernel_shape", listOf())
        val autoPad = attributes.getOrDefault("auto_pad", "NOTSET") as? String ?: "NOTSET"

        // Determine kernel size from kernel_shape attribute (required in ONNX) or use default
        // The kernel_shape attribute should always be provided in ONNX models
        val kH = if (kernelShape.isNotEmpty()) kernelShape[0].toLong() else 3L
        val kW = if (kernelShape.size > 1) kernelShape[1].toLong() else kH  // Default to square kernel

        // Calculate padding based on auto_pad
        var pH = pads[0].toLong()
        var pW = pads[1].toLong()

        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
            // For SAME padding, calculate dynamically or use 0 for now
            // Actual padding will be computed at runtime based on input size
            pH = 0
            pW = 0
        }

        // Build Conv2D config for NHWC format
        val paddingMode = if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") PaddingMode.SAME else PaddingMode.VALID
        val config = Conv2DConfig.builder()
            .kH(kH)
            .kW(kW)
            .sH(strides[0].toLong())
            .sW(strides[1].toLong())
            .dH(dilations[0].toLong())
            .dW(dilations[1].toLong())
            .pH(pH)
            .pW(pW)
            .dataFormat("NHWC")
            .paddingMode(paddingMode)
            .build()

        // Perform convolution
        // Note: Input is NHWC, weights might need reordering
        var result = sd.cnn.conv2d(input, weights, config)

        // Add bias if present
        if (bias != null) {
            result = result.add(bias)
        }

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }

    private fun getIntListAttribute(attributes: Map<String, Any>, name: String, default: List<Int>): List<Int> {
        val value = attributes[name] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toInt() }
            is LongArray -> value.map { it.toInt() }
            is IntArray -> value.toList()
            else -> default
        }
    }
}
