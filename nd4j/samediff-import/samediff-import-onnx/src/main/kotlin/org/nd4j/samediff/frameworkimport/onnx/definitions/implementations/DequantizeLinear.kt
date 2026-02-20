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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX DequantizeLinear operation.
 *
 * ONNX DequantizeLinear spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
 *
 * Dequantizes a quantized tensor: y = (x - zero_point) * scale
 *
 * Inputs:
 * - x: Quantized input tensor
 * - x_scale: Scale factor(s)
 * - x_zero_point: Zero point(s) (optional)
 *
 * Attributes:
 * - axis: Axis for per-channel quantization (default: 1)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["DequantizeLinear"], frameworkName = "onnx")
class DequantizeLinear : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])  // Quantized input
        val scale = sd.getVariable(op.inputsToOp[1])  // Scale
        val zeroPoint = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        
        // Get axis (default: 1)
        val axis = (attributes.getOrDefault("axis", 1L) as Number).toInt()
        
        // Convert input to float for computation
        val xFloat = x.castTo(scale.dataType())
        
        // Dequantize: y = (x - zero_point) * scale
        val output = if (zeroPoint != null) {
            val zeroPointFloat = zeroPoint.castTo(scale.dataType())
            val shifted = sd.math.sub("${opName}_shift", xFloat, zeroPointFloat)
            sd.math.mul(outputNames[0], shifted, scale)
        } else {
            sd.math.mul(outputNames[0], xFloat, scale)
        }
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
