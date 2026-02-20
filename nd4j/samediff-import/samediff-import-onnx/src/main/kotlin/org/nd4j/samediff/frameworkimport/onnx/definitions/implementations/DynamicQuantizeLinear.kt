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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX DynamicQuantizeLinear operation.
 *
 * ONNX DynamicQuantizeLinear spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear
 *
 * Computes scale and zero-point dynamically from input, then quantizes to uint8.
 * scale = (max(x) - min(x)) / 255
 * zero_point = saturate(round(-min(x) / scale))
 * y = saturate(round(x / scale) + zero_point)
 *
 * Inputs:
 * - x: Input tensor to quantize
 *
 * Outputs:
 * - y: Quantized output (uint8)
 * - y_scale: Computed scale (float)
 * - y_zero_point: Computed zero point (uint8)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["DynamicQuantizeLinear"], frameworkName = "onnx")
class DynamicQuantizeLinear : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])
        
        // Compute min and max of input
        val xMin = sd.min("${opName}_min", x)
        val xMax = sd.max("${opName}_max", x)

        // Constants for clamping
        val zero = sd.constant(0.0)
        val c255 = sd.constant(255.0)
        val epsilon = sd.constant(1e-10)

        // Clamp min to <= 0 and max to >= 0 to ensure zero is representable
        val qMin = sd.math.min("${opName}_qmin", xMin, zero)
        val qMax = sd.math.max("${opName}_qmax", xMax, zero)

        // Compute scale: (max - min) / 255
        val range = sd.math.sub("${opName}_range", qMax, qMin)
        val scale = sd.math.div("${opName}_scale", range, c255)

        // Avoid division by zero
        val safeScale = sd.math.max("${opName}_safeScale", scale, epsilon)
        
        // Compute zero point: round(-min / scale), clamped to [0, 255]
        val zeroPointFloat = sd.math.div("${opName}_zpf", sd.math.neg(qMin), safeScale)
        val zeroPointRounded = sd.math.round("${opName}_zpr", zeroPointFloat)
        val zeroPointClamped = sd.math.clipByValue("${opName}_zpc", zeroPointRounded, 0.0, 255.0)
        val zeroPoint = zeroPointClamped.castTo(DataType.UINT8)
        
        // Quantize: round(x / scale) + zero_point, clamped to [0, 255]
        val scaled = sd.math.div("${opName}_scaled", x, safeScale)
        val rounded = sd.math.round("${opName}_rounded", scaled)
        val shifted = sd.math.add("${opName}_shifted", rounded, zeroPointClamped)
        val clamped = sd.math.clipByValue("${opName}_clamped", shifted, 0.0, 255.0)
        val quantized = clamped.castTo(DataType.UINT8)
        
        // Rename outputs
        val y = quantized.rename(outputNames[0])
        val yScale = safeScale.rename(outputNames.getOrElse(1) { "${opName}_y_scale" })
        val yZeroPoint = zeroPoint.rename(outputNames.getOrElse(2) { "${opName}_y_zero_point" })
        
        return mapOf(
            outputNames[0] to listOf(y),
            outputNames.getOrElse(1) { "${opName}_y_scale" } to listOf(yScale),
            outputNames.getOrElse(2) { "${opName}_y_zero_point" } to listOf(yZeroPoint)
        )
    }
}
