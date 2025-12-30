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
 * Implementation of ONNX QuantizeLinear operation.
 *
 * ONNX QuantizeLinear spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
 *
 * Quantizes a float tensor: y = saturate(round(x / scale) + zero_point)
 *
 * Inputs:
 * - x: Input tensor to quantize
 * - y_scale: Scale factor(s)
 * - y_zero_point: Zero point(s) (optional)
 *
 * Attributes:
 * - axis: Axis for per-channel quantization (default: 1)
 * - saturate: Whether to saturate output (default: 1)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["QuantizeLinear"], frameworkName = "onnx")
class QuantizeLinear : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])  // Input to quantize
        val scale = sd.getVariable(op.inputsToOp[1])  // Scale
        val zeroPoint = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        
        // Get attributes
        val axis = (attributes.getOrDefault("axis", 1L) as Number).toInt()
        val saturate = (attributes.getOrDefault("saturate", 1L) as Number).toInt()
        
        // Determine output dtype from zero_point (default: UINT8)
        val outputDtype = zeroPoint?.dataType() ?: DataType.UINT8
        
        // Quantize: y = round(x / scale) + zero_point
        val scaled = sd.math.div("${opName}_scaled", x, scale)
        val rounded = sd.math.round("${opName}_rounded", scaled)
        
        var result = if (zeroPoint != null) {
            val zeroPointFloat = zeroPoint.castTo(rounded.dataType())
            sd.math.add("${opName}_shifted", rounded, zeroPointFloat)
        } else {
            rounded
        }
        
        // Saturate to output dtype range
        if (saturate == 1) {
            val (minVal, maxVal) = when (outputDtype) {
                DataType.INT8 -> Pair(-128.0, 127.0)
                DataType.UINT8 -> Pair(0.0, 255.0)
                DataType.INT16 -> Pair(-32768.0, 32767.0)
                DataType.UINT16 -> Pair(0.0, 65535.0)
                DataType.INT32 -> Pair(-2147483648.0, 2147483647.0)
                else -> Pair(0.0, 255.0)  // Default to UINT8 range
            }
            result = sd.math.clipByValue("${opName}_saturated", result, minVal, maxVal)
        }
        
        // Cast to output dtype
        val output = result.castTo(outputDtype).rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
