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
 * Implementation of Microsoft ONNX FusedGemm operation.
 *
 * FusedGemm performs GEMM with optional activation:
 * output = activation(alpha * A @ B + beta * C)
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#FusedGemm
 *
 * Inputs:
 * - A: First matrix
 * - B: Second matrix
 * - C: Optional bias matrix
 *
 * Attributes:
 * - alpha: Scalar multiplier for A @ B (default: 1.0)
 * - beta: Scalar multiplier for C (default: 1.0)
 * - transA: Transpose A (default: 0)
 * - transB: Transpose B (default: 0)
 * - activation: Activation function name
 * - activation_alpha: Alpha parameter for activation
 * - activation_beta: Beta parameter for activation
 * - activation_gamma: Gamma parameter for activation
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["FusedGemm"], frameworkName = "onnx")
class FusedGemm : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        var a = sd.getVariable(op.inputsToOp[0])
        var b = sd.getVariable(op.inputsToOp[1])

        // Optional bias
        val hasC = op.inputsToOp.size > 2 && op.inputsToOp[2] != null
        val c = if (hasC) sd.getVariable(op.inputsToOp[2]) else null

        // Get attributes
        val alpha = (attributes.getOrDefault("alpha", 1.0) as Number).toDouble()
        val beta = (attributes.getOrDefault("beta", 1.0) as Number).toDouble()
        val transA = (attributes.getOrDefault("transA", 0) as Number).toInt() != 0
        val transB = (attributes.getOrDefault("transB", 0) as Number).toInt() != 0
        val activation = attributes.getOrDefault("activation", "") as? String ?: ""
        val activationAlpha = (attributes.getOrDefault("activation_alpha", 0.01) as Number).toDouble()
        val activationBeta = (attributes.getOrDefault("activation_beta", 0.5) as Number).toDouble()

        // Handle transpose
        if (transA) {
            a = sd.permute(a, 1, 0)
        }
        if (transB) {
            b = sd.permute(b, 1, 0)
        }

        // Compute A @ B
        var result = sd.mmul(a, b)

        // Apply alpha scaling
        if (alpha != 1.0) {
            result = sd.math.mul(result, alpha)
        }

        // Add bias
        if (c != null) {
            val scaledC = if (beta != 1.0) {
                sd.math.mul(c, beta)
            } else {
                c
            }
            result = sd.math.add(result, scaledC)
        }

        // Apply activation
        result = when (activation.lowercase()) {
            "relu" -> sd.nn.relu(result, 0.0)
            "sigmoid" -> sd.nn.sigmoid(result)
            "tanh" -> sd.math.tanh(result)
            "leakyrelu" -> sd.nn.leakyRelu(result, activationAlpha)
            "elu" -> sd.nn.elu("_elu", result)
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
            "thresholdedrelu" -> {
                // ThresholdedRelu: y = x if x > alpha else 0
                val threshold = sd.constant(activationAlpha)
                sd.where(sd.gt(result, threshold), result, sd.zerosLike(result))
            }
            "affine" -> {
                // Affine: y = alpha * x + beta
                sd.math.add(sd.math.mul(result, activationAlpha), activationBeta)
            }
            "scaledtanh" -> {
                // ScaledTanh: y = alpha * tanh(beta * x)
                sd.math.mul(sd.math.tanh(sd.math.mul(result, activationBeta)), activationAlpha)
            }
            else -> result  // No activation or unknown activation
        }

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }
}
