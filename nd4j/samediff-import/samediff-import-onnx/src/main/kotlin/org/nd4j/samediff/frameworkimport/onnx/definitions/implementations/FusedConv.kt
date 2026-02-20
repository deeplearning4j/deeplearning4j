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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Microsoft ONNX FusedConv operation.
 *
 * FusedConv combines convolution with optional activation function:
 * output = activation(conv(X, W) + B + Z)
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#FusedConv
 *
 * Inputs:
 * - X: Input tensor (NCHW format)
 * - W: Weight tensor
 * - B: Optional bias tensor
 * - Z: Optional tensor for residual/skip connection
 *
 * Attributes:
 * - activation: Activation function name (Relu, Sigmoid, Tanh, LeakyRelu, etc.)
 * - activation_params: Parameters for activation (e.g., alpha for LeakyRelu)
 * - auto_pad: Auto padding mode
 * - dilations: Dilation values
 * - group: Number of groups
 * - kernel_shape: Kernel shape
 * - pads: Padding values
 * - strides: Stride values
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["FusedConv"], frameworkName = "onnx")
class FusedConv : PreImportHook {

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

        // Optional Z (residual connection)
        val hasZ = op.inputsToOp.size > 3 && op.inputsToOp[3] != null
        val z = if (hasZ) sd.getVariable(op.inputsToOp[3]) else null

        // Get convolution attributes
        val strides = getIntListAttribute(attributes, "strides", listOf(1, 1))
        val dilations = getIntListAttribute(attributes, "dilations", listOf(1, 1))
        val pads = getIntListAttribute(attributes, "pads", listOf(0, 0, 0, 0))
        val group = (attributes.getOrDefault("group", 1) as Number).toInt()
        val activation = attributes.getOrDefault("activation", "") as? String ?: ""
        val activationParams = getFloatListAttribute(attributes, "activation_params", listOf())

        // Get kernel shape from attribute (standard ONNX practice)
        val kernelShape = getIntListAttribute(attributes, "kernel_shape", listOf(3, 3))
        val kH = kernelShape[0].toLong()
        val kW = if (kernelShape.size > 1) kernelShape[1].toLong() else kH

        // Build Conv2D config
        val config = Conv2DConfig.builder()
            .kH(kH)
            .kW(kW)
            .sH(strides[0].toLong())
            .sW(strides[1].toLong())
            .dH(dilations[0].toLong())
            .dW(dilations[1].toLong())
            .pH(pads[0].toLong())
            .pW(pads[1].toLong())
            .dataFormat("NCHW")
            .build()

        // Perform convolution
        var result = sd.cnn.conv2d(input, weights, config)

        // Add bias if present
        if (bias != null) {
            result = result.add(bias)
        }

        // Add residual connection if present
        if (z != null) {
            result = result.add(z)
        }

        // Apply activation
        result = when (activation.lowercase()) {
            "relu" -> sd.nn.relu(result, 0.0)
            "sigmoid" -> sd.nn.sigmoid(result)
            "tanh" -> sd.math.tanh(result)
            "leakyrelu" -> {
                val alpha = if (activationParams.isNotEmpty()) activationParams[0].toDouble() else 0.01
                sd.nn.leakyRelu(result, alpha)
            }
            "elu" -> {
                // Note: sd.nn.elu doesn't take alpha parameter, uses default 1.0
                sd.nn.elu("_elu", result)
            }
            "hardsigmoid" -> sd.nn.hardSigmoid(result)
            "hardswish" -> {
                // HardSwish: x * max(0, min(1, (x + 3) / 6))
                val shifted = sd.math.add(result, 3.0)
                val scaled = sd.math.div(shifted, 6.0)
                val clipped = sd.math.clipByValue(scaled, 0.0, 1.0)
                sd.math.mul(result, clipped)
            }
            "softplus" -> sd.nn.softplus(result)
            "softsign" -> sd.nn.softsign(result)
            else -> result  // No activation or unknown activation
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

    private fun getFloatListAttribute(attributes: Map<String, Any>, name: String, default: List<Float>): List<Float> {
        val value = attributes[name] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toFloat() }
            is FloatArray -> value.toList()
            is DoubleArray -> value.map { it.toFloat() }
            else -> default
        }
    }
}
