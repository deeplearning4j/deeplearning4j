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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX SVMClassifier operation (ML domain).
 *
 * ONNX SVMClassifier spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.SVMClassifier
 *
 * Support Vector Machine classifier for binary and multiclass classification.
 *
 * Inputs:
 * - X: Input features [N, F]
 *
 * Attributes:
 * - coefficients: SVM coefficients
 * - kernel_params: Kernel parameters (gamma, coef0, degree)
 * - kernel_type: LINEAR, POLY, RBF, SIGMOID
 * - post_transform: NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT
 * - prob_a, prob_b: Probability calibration parameters
 * - rho: Bias terms
 * - support_vectors: Support vectors
 * - vectors_per_class: Number of support vectors per class
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["SVMClassifier"], frameworkName = "onnx")
class SVMClassifier : PreImportHook {

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
        val classLabelsInt = getLongArray(attributes, "classlabels_ints")
        val postTransform = attributes.getOrDefault("post_transform", "NONE") as? String ?: "NONE"
        val vectorsPerClass = getLongArray(attributes, "vectors_per_class")
        
        // For LINEAR kernel, this is essentially a linear classifier
        // For other kernels, we need to compute kernel values with support vectors
        
        val numClasses = classLabelsInt?.size ?: 2
        
        val scores = if (kernelType.uppercase() == "LINEAR" && coefficients != null) {
            // Linear SVM: decision = X * W + b
            val numFeatures = x.shape[1].toInt()
            val weightsArr = Nd4j.createFromArray(*coefficients)
                .reshape(numClasses.toLong() - 1, numFeatures.toLong()).transpose()
            val weights = sd.constant("${opName}_weights", weightsArr)
            
            val bias = if (rho != null) {
                sd.constant("${opName}_bias", Nd4j.createFromArray(*rho).mul(-1.0))
            } else {
                sd.constant("${opName}_bias", Nd4j.zeros(numClasses - 1))
            }
            
            sd.math.add("${opName}_scores", sd.mmul(x, weights), bias)
        } else if (supportVectors != null && coefficients != null) {
            // Kernel SVM - simplified RBF implementation
            val gamma = kernelParams?.getOrElse(0) { 1.0f } ?: 1.0f
            
            // Compute kernel matrix between X and support vectors
            // For RBF: K(x, sv) = exp(-gamma * ||x - sv||^2)
            val numSV = supportVectors.size / (x.shape[1].toInt())
            val svArr = Nd4j.createFromArray(*supportVectors)
                .reshape(numSV.toLong(), x.shape[1])
            val sv = sd.constant("${opName}_sv", svArr)
            
            // Compute squared distances
            val xExpanded = sd.expandDims("${opName}_x_exp", x, 1)
            val svExpanded = sd.expandDims("${opName}_sv_exp", sv, 0)
            val diff = sd.math.sub("${opName}_diff", xExpanded, svExpanded)
            val sqDist = sd.math.sum("${opName}_sq_dist", sd.math.square(diff), false, 2)
            
            // Apply RBF kernel
            val kernelMatrix = sd.math.exp("${opName}_kernel", 
                sd.math.mul(sqDist, -gamma.toDouble()))
            
            // Multiply by coefficients
            val coefArr = Nd4j.createFromArray(*coefficients)
            val coef = sd.constant("${opName}_coef", coefArr)
            
            sd.mmul("${opName}_scores", kernelMatrix, coef.reshape(-1, 1))
        } else {
            // Fallback: return zeros
            sd.zerosLike("${opName}_scores", x)
        }
        
        // Get predictions
        val predictions = if (numClasses == 2) {
            // Binary classification: sign of score
            val sign = sd.math.sign(scores)
            val zero = sd.constant(0.0)
            sd.math.max("${opName}_pred", sign, zero).castTo(DataType.INT64)
        } else {
            sd.argmax("${opName}_pred", scores, false, -1)
        }
        
        // Map to class labels
        val output = if (classLabelsInt != null && classLabelsInt.isNotEmpty()) {
            val labelsArr = Nd4j.createFromArray(*classLabelsInt)
            val labels = sd.constant("${opName}_labels", labelsArr)
            sd.gather(outputNames[0], labels, predictions.castTo(DataType.INT64), 0)
        } else {
            predictions.rename(outputNames[0])
        }
        
        // Compute probabilities if needed
        val probs = when (postTransform.uppercase()) {
            "SOFTMAX" -> sd.nn.softmax("${opName}_probs", scores, -1)
            "LOGISTIC" -> sd.nn.sigmoid("${opName}_probs", scores)
            else -> scores
        }
        val probsOutput = probs.rename(outputNames.getOrElse(1) { "${opName}_Z" })
        
        return mapOf(
            outputNames[0] to listOf(output),
            outputNames.getOrElse(1) { "${opName}_Z" } to listOf(probsOutput)
        )
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
    
    private fun getLongArray(attributes: Map<String, Any>, key: String): LongArray? {
        val value = attributes[key] ?: return null
        return when (value) {
            is LongArray -> value
            is IntArray -> value.map { it.toLong() }.toLongArray()
            is List<*> -> value.map { (it as Number).toLong() }.toLongArray()
            else -> null
        }
    }
}
