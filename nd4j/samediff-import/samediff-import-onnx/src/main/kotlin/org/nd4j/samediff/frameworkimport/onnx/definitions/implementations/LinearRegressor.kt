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
 * Implementation of ONNX LinearRegressor operation (ML domain).
 *
 * ONNX LinearRegressor spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.LinearRegressor
 *
 * Linear regression: Y = X * coefficients + intercepts
 * Optionally applies post_transform.
 *
 * Inputs:
 * - X: Input features [N, F]
 *
 * Attributes:
 * - coefficients: Coefficient values (flattened)
 * - intercepts: Intercept/bias values
 * - targets: Number of targets (default: 1)
 * - post_transform: NONE or PROBIT
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["LinearRegressor"], frameworkName = "onnx")
class LinearRegressor : PreImportHook {

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
        val intercepts = getFloatArray(attributes, "intercepts")
        val targets = (attributes.getOrDefault("targets", 1L) as Number).toInt()
        val postTransform = attributes.getOrDefault("post_transform", "NONE") as? String ?: "NONE"
        
        // Determine dimensions
        val numTargets = intercepts?.size ?: targets
        val numFeatures = if (coefficients != null && numTargets > 0) 
            coefficients.size / numTargets else 0
        
        // Create weight matrix [features, targets]
        val weightsArr = if (coefficients != null && numFeatures > 0) {
            Nd4j.createFromArray(*coefficients).reshape(numTargets.toLong(), numFeatures.toLong()).transpose()
        } else {
            Nd4j.zeros(1, numTargets)
        }
        val weights = sd.constant("${opName}_weights", weightsArr)
        
        // Create intercepts vector
        val biasArr = if (intercepts != null) {
            Nd4j.createFromArray(*intercepts)
        } else {
            Nd4j.zeros(numTargets)
        }
        val bias = sd.constant("${opName}_bias", biasArr)
        
        // Compute output: Y = X * W + b
        var output = sd.math.add("${opName}_linear", sd.mmul(x, weights), bias)
        
        // Apply post-transform if specified
        if (postTransform.uppercase() == "PROBIT") {
            // Probit transform: inverse of normal CDF
            // Approximation using tanh
            output = sd.math.tanh("${opName}_probit", output)
        }
        
        output = output.rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
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
