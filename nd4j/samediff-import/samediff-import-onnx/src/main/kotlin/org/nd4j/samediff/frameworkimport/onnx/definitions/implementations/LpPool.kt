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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX LpPool operation.
 *
 * ONNX LpPool spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool
 *
 * Lp pooling consumes an input tensor and applies lp pool pooling across the
 * tensor according to kernel sizes, stride sizes, and pad lengths.
 *
 * Inputs:
 * - X: Input tensor of shape [N, C, H, W]
 *
 * Attributes:
 * - auto_pad: Padding mode ("NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID")
 * - kernel_shape: Kernel size
 * - p: Exponent of Lp norm (default: 2)
 * - pads: Padding values
 * - strides: Stride values
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["LpPool"], frameworkName = "onnx")
class LpPool : PreImportHook {

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
        
        // Get input
        val input = sd.getVariable(op.inputsToOp[0])
        
        // Get attributes
        val p = (attributes.getOrDefault("p", 2L) as Number).toInt()
        val autoPad = attributes.getOrDefault("auto_pad", "NOTSET") as? String ?: "NOTSET"
        
        // Get kernel shape
        val kernelShape = getIntList(attributes, "kernel_shape", listOf(1, 1))
        val kH = kernelShape.getOrElse(0) { 1 }.toLong()
        val kW = kernelShape.getOrElse(1) { 1 }.toLong()
        
        // Get strides (default: [1, 1])
        val strides = getIntList(attributes, "strides", listOf(1, 1))
        val sH = strides.getOrElse(0) { 1 }.toLong()
        val sW = strides.getOrElse(1) { 1 }.toLong()
        
        // Get pads (default: [0, 0, 0, 0])
        val pads = getIntList(attributes, "pads", listOf(0, 0, 0, 0))
        val pH = pads.getOrElse(0) { 0 }.toLong()
        val pW = pads.getOrElse(1) { 0 }.toLong()
        
        // Determine padding mode
        val paddingMode = if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") PaddingMode.SAME else PaddingMode.VALID

        // LpPool is implemented as:
        // For p=2: sqrt(avgPool(x^2) * kernel_size)
        // For general p: (avgPool(|x|^p) * kernel_size)^(1/p)
        val poolConfig = Pooling2DConfig.builder()
            .kH(kH)
            .kW(kW)
            .sH(sH)
            .sW(sW)
            .pH(pH)
            .pW(pW)
            .paddingMode(paddingMode)
            .isNHWC(false)
            .build()
        
        val kernelSize = kH * kW
        
        val output = if (p == 2) {
            // L2 pool: sqrt(avgPool(x^2) * kernel_size)
            val squared = sd.math.square(input)
            val avgPooled = sd.cnn.avgPooling2d("${opName}_avgpool", squared, poolConfig)
            val sumPooled = sd.math.mul("${opName}_sum", avgPooled, kernelSize.toDouble())
            sd.math.sqrt(outputNames[0], sumPooled)
        } else {
            // General Lp pool: (avgPool(|x|^p) * kernel_size)^(1/p)
            val absInput = sd.math.abs(input)
            val powered = sd.math.pow("${opName}_pow", absInput, p.toDouble())
            val avgPooled = sd.cnn.avgPooling2d("${opName}_avgpool", powered, poolConfig)
            val sumPooled = sd.math.mul("${opName}_sum", avgPooled, kernelSize.toDouble())
            sd.math.pow(outputNames[0], sumPooled, 1.0 / p)
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
