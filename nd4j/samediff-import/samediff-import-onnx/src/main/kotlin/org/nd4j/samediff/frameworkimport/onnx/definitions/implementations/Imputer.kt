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
 * Implementation of ONNX Imputer operation (ML domain).
 *
 * ONNX Imputer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Imputer
 *
 * Replaces missing values (NaN or specified value) with imputed values.
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - imputed_value_floats: Replacement values (per feature or single value)
 * - imputed_value_int64s: Replacement values for integer inputs
 * - replaced_value_float: Value to replace (default: NaN)
 * - replaced_value_int64: Value to replace for integer inputs
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Imputer"], frameworkName = "onnx")
class Imputer : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])
        
        // Get imputed values
        val imputedFloats = getFloatArray(attributes, "imputed_value_floats")
        val imputedInts = getLongArray(attributes, "imputed_value_int64s")
        
        // Get replaced value (value to look for and replace)
        val replacedFloat = attributes["replaced_value_float"] as? Number
        val replacedInt = attributes["replaced_value_int64"] as? Number
        
        // Create mask for values to replace
        val mask = if (replacedFloat != null && !java.lang.Double.isNaN(replacedFloat.toDouble())) {
            // Replace specific value
            sd.eq("${opName}_mask", input, replacedFloat.toDouble())
        } else {
            // Replace NaN values
            sd.math.isNaN("${opName}_mask", input)
        }
        
        // Create imputed values tensor
        val imputedVar = when {
            imputedFloats != null && imputedFloats.isNotEmpty() -> {
                if (imputedFloats.size == 1) {
                    // Single value for all features
                    sd.constant("${opName}_imputed", Nd4j.scalar(imputedFloats[0]))
                } else {
                    // Per-feature values
                    sd.constant("${opName}_imputed", Nd4j.createFromArray(*imputedFloats))
                }
            }
            imputedInts != null && imputedInts.isNotEmpty() -> {
                if (imputedInts.size == 1) {
                    sd.constant("${opName}_imputed", Nd4j.scalar(imputedInts[0].toFloat()))
                } else {
                    sd.constant("${opName}_imputed", Nd4j.createFromArray(*imputedInts.map { it.toFloat() }.toFloatArray()))
                }
            }
            else -> {
                // Default to 0
                sd.constant("${opName}_imputed", Nd4j.scalar(0.0f))
            }
        }
        
        // Apply imputation: where mask is true, use imputed value; otherwise keep original
        val maskFloat = mask.castTo(input.dataType())
        val oneConst = sd.constant("${opName}_one", 1.0)
        val invMask = sd.math.sub("${opName}_invMask", oneConst, maskFloat)

        // output = input * (1 - mask) + imputed * mask
        val kept = sd.math.mul("${opName}_kept", input, invMask)
        val replaced = sd.math.mul("${opName}_replaced", imputedVar, maskFloat)
        val output = sd.math.add(outputNames[0], kept, replaced)
        
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
