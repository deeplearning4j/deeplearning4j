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
 * Implementation of ONNX ConvInteger operation.
 *
 * ONNX ConvInteger spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger
 *
 * Performs integer convolution with zero-point handling.
 * The computation is: Y = (X - x_zero_point) conv (W - w_zero_point)
 *
 * Inputs:
 * - x: Quantized input tensor [N, C, H, W]
 * - w: Quantized weight tensor
 * - x_zero_point: Zero point for input (optional)
 * - w_zero_point: Zero point for weights (optional)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["ConvInteger"], frameworkName = "onnx")
class ConvInteger : PreImportHook {

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
        val w = sd.getVariable(op.inputsToOp[1])
        val xZeroPoint = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        val wZeroPoint = if (op.inputsToOp.size > 3) sd.getVariable(op.inputsToOp[3]) else null
        
        // Get conv attributes
        val autoPad = attributes.getOrDefault("auto_pad", "NOTSET") as? String ?: "NOTSET"
        val group = (attributes.getOrDefault("group", 1L) as Number).toInt()
        val strides = getIntList(attributes, "strides", listOf(1, 1))
        val dilations = getIntList(attributes, "dilations", listOf(1, 1))
        val pads = getIntList(attributes, "pads", listOf(0, 0, 0, 0))
        val kernelShape = getIntList(attributes, "kernel_shape", listOf())
        
        // Convert inputs to INT32 for computation
        var xInt = x.castTo(DataType.INT32)
        var wInt = w.castTo(DataType.INT32)
        
        // Subtract zero points
        if (xZeroPoint != null) {
            val xzp = xZeroPoint.castTo(DataType.INT32)
            xInt = sd.math.sub("${opName}_x_shift", xInt, xzp)
        }
        
        if (wZeroPoint != null) {
            val wzp = wZeroPoint.castTo(DataType.INT32)
            wInt = sd.math.sub("${opName}_w_shift", wInt, wzp)
        }
        
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
        
        // Perform convolution (cast to float, then back to int32)
        val xFloat = xInt.castTo(DataType.FLOAT)
        val wFloat = wInt.castTo(DataType.FLOAT)
        val convResult = sd.cnn.conv2d("${opName}_conv", xFloat, wFloat, config)
        
        // Convert back to INT32
        val output = convResult.castTo(DataType.INT32).rename(outputNames[0])
        
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
