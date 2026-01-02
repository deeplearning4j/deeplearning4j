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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX QLinearConv operation.
 *
 * ONNX QLinearConv spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
 *
 * Quantized convolution with linear quantization parameters.
 * Similar to Conv but with quantized inputs and outputs.
 *
 * Inputs:
 * - x: Quantized input tensor [N, C, H, W]
 * - x_scale: Input scale
 * - x_zero_point: Input zero point
 * - w: Quantized weight tensor
 * - w_scale: Weight scale
 * - w_zero_point: Weight zero point
 * - y_scale: Output scale
 * - y_zero_point: Output zero point
 * - B: Optional bias (int32)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["QLinearConv"], frameworkName = "onnx")
class QLinearConv : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])
        val xScale = sd.getVariable(op.inputsToOp[1])
        val xZeroPoint = sd.getVariable(op.inputsToOp[2])
        val w = sd.getVariable(op.inputsToOp[3])
        val wScale = sd.getVariable(op.inputsToOp[4])
        val wZeroPoint = sd.getVariable(op.inputsToOp[5])
        val yScale = sd.getVariable(op.inputsToOp[6])
        val yZeroPoint = sd.getVariable(op.inputsToOp[7])
        val bias = if (op.inputsToOp.size > 8) sd.getVariable(op.inputsToOp[8]) else null
        
        // Get conv attributes
        val autoPad = attributes.getOrDefault("auto_pad", "NOTSET") as? String ?: "NOTSET"
        val group = (attributes.getOrDefault("group", 1L) as Number).toInt()
        val strides = getIntList(attributes, "strides", listOf(1, 1))
        val dilations = getIntList(attributes, "dilations", listOf(1, 1))
        val pads = getIntList(attributes, "pads", listOf(0, 0, 0, 0))
        val kernelShape = getIntList(attributes, "kernel_shape", listOf())
        
        // Dequantize input: x_float = (x - x_zero_point) * x_scale
        val xFloat = x.castTo(DataType.FLOAT)
        val xzpFloat = xZeroPoint.castTo(DataType.FLOAT)
        val xDequant = sd.math.mul("${opName}_x_dequant", 
            sd.math.sub("${opName}_x_shift", xFloat, xzpFloat), 
            xScale)
        
        // Dequantize weights: w_float = (w - w_zero_point) * w_scale
        val wFloat = w.castTo(DataType.FLOAT)
        val wzpFloat = wZeroPoint.castTo(DataType.FLOAT)
        val wDequant = sd.math.mul("${opName}_w_dequant", 
            sd.math.sub("${opName}_w_shift", wFloat, wzpFloat), 
            wScale)
        
        // Build conv config
        val paddingMode = if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") PaddingMode.SAME else PaddingMode.VALID
        val config = Conv2DConfig.builder()
            .kH(kernelShape.getOrElse(0) { 0 }.toLong())
            .kW(kernelShape.getOrElse(1) { 0 }.toLong())
            .sH(strides.getOrElse(0) { 1 }.toLong())
            .sW(strides.getOrElse(1) { 1 }.toLong())
            .pH(pads.getOrElse(0) { 0 }.toLong())
            .pW(pads.getOrElse(1) { 0 }.toLong())
            .dH(dilations.getOrElse(0) { 1 }.toLong())
            .dW(dilations.getOrElse(1) { 1 }.toLong())
            .paddingMode(paddingMode)
            .dataFormat("NCHW")
            .build()
        
        // Perform convolution
        var convResult = sd.cnn.conv2d("${opName}_conv", xDequant, wDequant, config)
        
        // Add bias if present
        if (bias != null) {
            val biasFloat = bias.castTo(DataType.FLOAT)
            val biasReshaped = sd.reshape("${opName}_biasReshape", biasFloat, 1, -1, 1, 1)
            convResult = sd.math.add("${opName}_biased", convResult, biasReshaped)
        }
        
        // Requantize output: y = round(result / y_scale) + y_zero_point
        val scaled = sd.math.div("${opName}_scaled", convResult, yScale)
        val rounded = sd.math.round("${opName}_rounded", scaled)
        val yzpFloat = yZeroPoint.castTo(DataType.FLOAT)
        val shifted = sd.math.add("${opName}_shifted", rounded, yzpFloat)
        
        // Clamp and cast to output dtype
        val outputDtype = yZeroPoint.dataType()
        val (minVal, maxVal) = when (outputDtype) {
            DataType.INT8 -> Pair(-128.0, 127.0)
            DataType.UINT8 -> Pair(0.0, 255.0)
            else -> Pair(0.0, 255.0)
        }
        val clamped = sd.math.clipByValue("${opName}_clamped", shifted, minVal, maxVal)
        val output = clamped.castTo(outputDtype).rename(outputNames[0])
        
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
