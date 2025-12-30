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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX SVMRegressor operation (ML domain).
 *
 * ONNX SVMRegressor spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.SVMRegressor
 *
 * Support Vector Machine regressor for regression tasks.
 *
 * Inputs:
 * - X: Input features [N, F]
 *
 * Attributes:
 * - coefficients: SVM coefficients
 * - kernel_params: Kernel parameters (gamma, coef0, degree)
 * - kernel_type: LINEAR, POLY, RBF, SIGMOID
 * - n_supports: Number of support vectors
 * - one_class: Whether one-class SVM (default: 0)
 * - post_transform: NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT
 * - rho: Bias terms
 * - support_vectors: Support vectors
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["SVMRegressor"], frameworkName = "onnx")
class SVMRegressor : PreImportHook {

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

        // Get attributes
        val coefficients = getFloatArray(attributes, "coefficients")
        val supportVectors = getFloatArray(attributes, "support_vectors")
        val rho = getFloatArray(attributes, "rho")
        val kernelType = attributes.getOrDefault("kernel_type", "LINEAR") as? String ?: "LINEAR"
        val kernelParams = getFloatArray(attributes, "kernel_params") // [gamma, coef0, degree]
        val nSupports = (attributes.getOrDefault("n_supports", 0L) as Number).toInt()
        val oneClass = (attributes.getOrDefault("one_class", 0L) as Number).toInt()
        val postTransform = attributes.getOrDefault("post_transform", "NONE") as? String ?: "NONE"

        val output = if (kernelType.uppercase() == "LINEAR" && coefficients != null) {
            // Linear SVM: y = X * W + b
            val numFeatures = x.shape[1].toInt()
            val numTargets = if (coefficients.size > numFeatures)
                coefficients.size / numFeatures else 1

            val weightsArr = Nd4j.createFromArray(*coefficients)
                .reshape(numTargets.toLong(), numFeatures.toLong()).transpose()
            val weights = sd.constant("${opName}_weights", weightsArr)

            val bias = if (rho != null) {
                sd.constant("${opName}_bias", Nd4j.createFromArray(*rho).mul(-1.0))
            } else {
                sd.constant("${opName}_bias", Nd4j.zeros(numTargets))
            }

            sd.math.add("${opName}_linear", sd.mmul(x, weights), bias)
        } else if (supportVectors != null && coefficients != null) {
            // Kernel SVM
            val gamma = kernelParams?.getOrElse(0) { 1.0f } ?: 1.0f
            val coef0 = kernelParams?.getOrElse(1) { 0.0f } ?: 0.0f
            val degree = kernelParams?.getOrElse(2) { 3.0f } ?: 3.0f

            // Compute kernel matrix between X and support vectors
            val numSV = if (nSupports > 0) nSupports else supportVectors.size / (x.shape[1].toInt())
            val svArr = Nd4j.createFromArray(*supportVectors)
                .reshape(numSV.toLong(), x.shape[1])
            val sv = sd.constant("${opName}_sv", svArr)

            // Compute kernel values based on kernel type
            val kernelMatrix = when (kernelType.uppercase()) {
                "RBF" -> {
                    // K(x, sv) = exp(-gamma * ||x - sv||^2)
                    val xExpanded = sd.expandDims("${opName}_x_exp", x, 1)
                    val svExpanded = sd.expandDims("${opName}_sv_exp", sv, 0)
                    val diff = sd.math.sub("${opName}_diff", xExpanded, svExpanded)
                    val sqDist = sd.math.sum("${opName}_sq_dist", sd.math.square(diff), false, 2)
                    sd.math.exp("${opName}_kernel", sd.math.mul(sqDist, -gamma.toDouble()))
                }
                "POLY" -> {
                    // K(x, sv) = (gamma * x.sv + coef0)^degree
                    val dotProd = sd.mmul("${opName}_dot", x, sd.transpose(sv))
                    val scaled = sd.math.add(sd.math.mul(dotProd, gamma.toDouble()), coef0.toDouble())
                    sd.math.pow("${opName}_kernel", scaled, degree.toDouble())
                }
                "SIGMOID" -> {
                    // K(x, sv) = tanh(gamma * x.sv + coef0)
                    val dotProd = sd.mmul("${opName}_dot", x, sd.transpose(sv))
                    sd.math.tanh("${opName}_kernel",
                        sd.math.add(sd.math.mul(dotProd, gamma.toDouble()), coef0.toDouble()))
                }
                else -> {
                    // LINEAR: K(x, sv) = x.sv
                    sd.mmul("${opName}_kernel", x, sd.transpose(sv))
                }
            }

            // Multiply by coefficients and sum
            val coefArr = Nd4j.createFromArray(*coefficients)
            val coef = sd.constant("${opName}_coef", coefArr)

            var result = sd.mmul("${opName}_result", kernelMatrix, coef.reshape(-1, 1))

            // Add bias (negative rho)
            if (rho != null) {
                val biasVal = sd.constant("${opName}_rho", Nd4j.createFromArray(*rho).mul(-1.0))
                result = sd.math.add("${opName}_with_bias", result, biasVal)
            }

            result
        } else {
            // Fallback: return zeros
            sd.zerosLike("${opName}_output", x)
        }

        // Apply post-transform if specified
        val transformedOutput = when (postTransform.uppercase()) {
            "SOFTMAX" -> sd.nn.softmax("${opName}_softmax", output, -1)
            "LOGISTIC" -> sd.nn.sigmoid("${opName}_sigmoid", output)
            "PROBIT" -> sd.math.tanh("${opName}_probit", output)
            else -> output
        }

        val finalOutput = transformedOutput.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
    }

    private fun getFloatArray(attributes: Map<String, Any>, key: String): FloatArray? {
        val value = attributes[key] ?: return null
        return when (value) {
            is FloatArray -> value
            is DoubleArray -> value.map { it.toFloat() }.toFloatArray()
            is List<*> -> value.map { (it as Number).toFloat() }.toFloatArray()
            else -> null
        }
    }
}
