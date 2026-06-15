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
 * Implementation of ONNX MatMulInteger operation.
 *
 * ONNX MatMulInteger spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger
 *
 * Matrix multiplication with integer inputs. The inputs are dequantized, multiplied,
 * and the result is returned as int32.
 *
 * Y = (A - a_zero_point) * (B - b_zero_point)
 *
 * Inputs:
 * - A: First matrix (M x K), int8 or uint8
 * - B: Second matrix (K x N), int8 or uint8
 * - a_zero_point: Zero point for A (optional)
 * - b_zero_point: Zero point for B (optional)
 *
 * Output:
 * - Y: Result matrix (M x N), int32
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["MatMulInteger"], frameworkName = "onnx")
class MatMulInteger : PreImportHook {

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
        val a = sd.getVariable(op.inputsToOp[0])  // First matrix
        val b = sd.getVariable(op.inputsToOp[1])  // Second matrix
        val aZeroPoint = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        val bZeroPoint = if (op.inputsToOp.size > 3) sd.getVariable(op.inputsToOp[3]) else null
        
        // Convert to INT32 for computation to avoid overflow
        var aInt = a.castTo(DataType.INT32)
        var bInt = b.castTo(DataType.INT32)
        
        // Subtract zero points if provided
        if (aZeroPoint != null) {
            val azp = aZeroPoint.castTo(DataType.INT32)
            aInt = sd.math.sub("${opName}_a_shifted", aInt, azp)
        }
        
        if (bZeroPoint != null) {
            val bzp = bZeroPoint.castTo(DataType.INT32)
            bInt = sd.math.sub("${opName}_b_shifted", bInt, bzp)
        }
        
        // Perform matrix multiplication
        // Cast to float for mmul, then back to int32
        val aFloat = aInt.castTo(DataType.FLOAT)
        val bFloat = bInt.castTo(DataType.FLOAT)
        val result = sd.mmul("${opName}_mmul", aFloat, bFloat)
        
        // Convert back to INT32
        val output = result.castTo(DataType.INT32).rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
