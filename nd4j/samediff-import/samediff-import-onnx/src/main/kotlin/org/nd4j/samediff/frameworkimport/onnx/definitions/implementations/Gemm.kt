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
 *  * Unless required by applicabl  e law or agreed to in writing, software
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
 * A port of cast.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/cast.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Gemm"],frameworkName = "onnx")
class Gemm : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm
        // ONNX Gemm: Y = alpha * A' * B' + beta * C
        // This is actually linear - PyTorch exports nn.Linear as Gemm

        val inputA = sd.getVariable(op.inputsToOp[0])
        val inputB = sd.getVariable(op.inputsToOp[1])
        val alphaAttr = attributes.getOrDefault("alpha", 1.0f)
        val betaAttr = attributes.getOrDefault("beta", 1.0f)
        val alpha = if (alphaAttr is Float) alphaAttr.toDouble() else alphaAttr as Double
        val beta = if (betaAttr is Float) betaAttr.toDouble() else betaAttr as Double
        val transA = attributes.getOrDefault("transA", 0L) as Long
        val transB = attributes.getOrDefault("transB", 0L) as Long

        if(op.inputsToOp.size > 2) {
            val biasVar = sd.getVariable(op.inputsToOp[2])

            // Compute: alpha * A' * B' + beta * C
            // When alpha=1 and beta=1, this simplifies to linear (xw_plus_b)
            if (alpha == 1.0 && beta == 1.0) {
                val outputVar = sd.nn().linear(outputNames[0],
                    inputA, inputB, biasVar, transA > 0, transB > 0, false)
                return mapOf(outputVar.name() to listOf(outputVar))
            } else {
                // General case: alpha * matmul(A, B) + beta * C
                // matmul signature: (a, b, alpha, beta, transA, transB)
                // We use alpha for scaling the matmul result, beta=0 since we handle bias separately
                var matmulResult = sd.linalg().matmul(inputA, inputB, alpha, 0.0, transA > 0, transB > 0)
                var scaledBias = biasVar
                if (beta != 1.0) {
                    scaledBias = biasVar.mul(beta)
                }
                val outputVar = matmulResult.add(scaledBias)
                return mapOf(outputNames[0] to listOf(sd.updateVariableNameAndReference(outputVar, outputNames[0])))
            }
        } else {
            // No bias case - just matmul with alpha (beta not applicable without C)
            val outputVar = sd.linalg().matmul(inputA, inputB, alpha, 0.0, transA > 0, transB > 0)
            return mapOf(outputVar.name() to listOf(outputVar))
        }
    }


}
