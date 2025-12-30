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
 * Implementation of ONNX QLinearMatMul operation.
 *
 * ONNX QLinearMatMul spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul
 *
 * Quantized matrix multiplication with linear quantization parameters.
 * Y = requantize(dequantize(A) * dequantize(B))
 *
 * Inputs:
 * - a: Quantized first matrix
 * - a_scale: Scale for A
 * - a_zero_point: Zero point for A
 * - b: Quantized second matrix
 * - b_scale: Scale for B
 * - b_zero_point: Zero point for B
 * - y_scale: Scale for output
 * - y_zero_point: Zero point for output
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["QLinearMatMul"], frameworkName = "onnx")
class QLinearMatMul : PreImportHook {

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
        val a = sd.getVariable(op.inputsToOp[0])
        val aScale = sd.getVariable(op.inputsToOp[1])
        val aZeroPoint = sd.getVariable(op.inputsToOp[2])
        val b = sd.getVariable(op.inputsToOp[3])
        val bScale = sd.getVariable(op.inputsToOp[4])
        val bZeroPoint = sd.getVariable(op.inputsToOp[5])
        val yScale = sd.getVariable(op.inputsToOp[6])
        val yZeroPoint = sd.getVariable(op.inputsToOp[7])
        
        // Dequantize A: a_float = (a - a_zero_point) * a_scale
        val aFloat = a.castTo(DataType.FLOAT)
        val azpFloat = aZeroPoint.castTo(DataType.FLOAT)
        val aDequant = sd.math.mul("${opName}_a_dequant", 
            sd.math.sub("${opName}_a_shift", aFloat, azpFloat), 
            aScale)
        
        // Dequantize B: b_float = (b - b_zero_point) * b_scale
        val bFloat = b.castTo(DataType.FLOAT)
        val bzpFloat = bZeroPoint.castTo(DataType.FLOAT)
        val bDequant = sd.math.mul("${opName}_b_dequant", 
            sd.math.sub("${opName}_b_shift", bFloat, bzpFloat), 
            bScale)
        
        // Matrix multiplication
        val matmulResult = sd.mmul("${opName}_matmul", aDequant, bDequant)
        
        // Requantize: y = round(result / y_scale) + y_zero_point
        val scaled = sd.math.div("${opName}_scaled", matmulResult, yScale)
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
}
