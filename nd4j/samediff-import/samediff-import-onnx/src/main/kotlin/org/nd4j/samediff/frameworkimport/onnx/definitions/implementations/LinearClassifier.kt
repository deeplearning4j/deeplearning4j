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
 * Implementation of ONNX LinearClassifier operation (ML domain).
 *
 * ONNX LinearClassifier spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.LinearClassifier
 *
 * Linear classifier: scores = X * coefficients + intercepts
 * Then applies post_transform (softmax, logistic, etc.) to get probabilities.
 *
 * Inputs:
 * - X: Input features [N, F]
 *
 * Attributes:
 * - coefficients: Coefficient matrix
 * - intercepts: Bias/intercept values
 * - classlabels_ints/strings: Class labels
 * - multi_class: Multi-class handling mode
 * - post_transform: NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["LinearClassifier"], frameworkName = "onnx")
class LinearClassifier : PreImportHook {

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
        
        // Get coefficients and intercepts
        val coefficients = getFloatArray(attributes, "coefficients")
        val intercepts = getFloatArray(attributes, "intercepts")
        val classLabelsInt = getLongArray(attributes, "classlabels_ints")
        val postTransform = attributes.getOrDefault("post_transform", "NONE") as? String ?: "NONE"
        
        // Determine number of classes
        val numClasses = intercepts?.size ?: classLabelsInt?.size ?: 2
        val numFeatures = if (coefficients != null && numClasses > 0) 
            coefficients.size / numClasses else 0
        
        // Create weight matrix [features, classes]
        val weightsArr = if (coefficients != null && numFeatures > 0) {
            Nd4j.createFromArray(*coefficients).reshape(numClasses.toLong(), numFeatures.toLong()).transpose()
        } else {
            Nd4j.zeros(1, numClasses)
        }
        val weights = sd.constant("${opName}_weights", weightsArr)
        
        // Create intercepts vector
        val biasArr = if (intercepts != null) {
            Nd4j.createFromArray(*intercepts)
        } else {
            Nd4j.zeros(numClasses)
        }
        val bias = sd.constant("${opName}_bias", biasArr)
        
        // Compute scores: X * W + b
        val scores = sd.math.add("${opName}_scores", sd.mmul(x, weights), bias)
        
        // Apply post-transform to get probabilities
        val probs = when (postTransform.uppercase()) {
            "SOFTMAX" -> sd.nn.softmax("${opName}_probs", scores, -1)
            "LOGISTIC" -> sd.nn.sigmoid("${opName}_probs", scores)
            "SOFTMAX_ZERO" -> sd.nn.softmax("${opName}_probs", scores, -1)
            else -> scores
        }
        
        // Get predicted class (argmax)
        val predictions = sd.argmax("${opName}_argmax", scores, false, -1)
        
        // Map predictions to class labels if provided
        val output = if (classLabelsInt != null && classLabelsInt.isNotEmpty()) {
            val labelsArr = Nd4j.createFromArray(*classLabelsInt)
            val labels = sd.constant("${opName}_labels", labelsArr)
            sd.gather(outputNames[0], labels, predictions.castTo(DataType.INT64), 0)
        } else {
            predictions.rename(outputNames[0])
        }
        
        // Return both class and probabilities
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
