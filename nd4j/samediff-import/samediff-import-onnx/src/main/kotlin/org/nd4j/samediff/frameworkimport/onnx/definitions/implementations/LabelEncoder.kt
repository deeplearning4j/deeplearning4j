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
 * Implementation of ONNX LabelEncoder operation (ML domain).
 *
 * ONNX LabelEncoder spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.LabelEncoder
 *
 * Maps input values to encoded integer labels or vice versa.
 *
 * Inputs:
 * - X: Input tensor (strings or integers)
 *
 * Attributes:
 * - keys_floats/keys_int64s/keys_strings: Input values to map from
 * - values_floats/values_int64s/values_strings: Output values to map to
 * - default_float/default_int64/default_string: Default value for unknown inputs
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["LabelEncoder"], frameworkName = "onnx")
class LabelEncoder : PreImportHook {

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
        
        // Get keys and values
        val keysInt = getLongArray(attributes, "keys_int64s")
        val keysFloat = getFloatArray(attributes, "keys_floats")
        val valuesInt = getLongArray(attributes, "values_int64s")
        val valuesFloat = getFloatArray(attributes, "values_floats")
        
        // Get default value
        val defaultInt = (attributes["default_int64"] as? Number)?.toLong() ?: -1L
        val defaultFloat = (attributes["default_float"] as? Number)?.toFloat() ?: -1.0f
        
        // For integer keys -> integer values encoding
        if (keysInt != null && keysInt.isNotEmpty()) {
            val values = valuesInt ?: LongArray(keysInt.size) { it.toLong() }
            
            // Create lookup table: for each key, find matching values
            // Start with default value
            var result = sd.onesLike("${opName}_default", input).mul(defaultInt.toDouble())
            
            // For each key-value pair, update result where input matches key
            for (i in keysInt.indices) {
                val key = keysInt[i]
                val value = values.getOrElse(i) { i.toLong() }
                val mask = sd.eq("${opName}_mask_$i", input, key.toDouble())
                val maskFloat = mask.castTo(result.dataType())
                val invMask = sd.math.sub("${opName}_inv_$i", 1.0, maskFloat)
                result = sd.math.add("${opName}_upd_$i",
                    sd.math.mul(result, invMask),
                    sd.math.mul(maskFloat, value.toDouble())
                )
            }
            
            val output = result.castTo(DataType.INT64).rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        }
        
        // For float keys -> float values encoding
        if (keysFloat != null && keysFloat.isNotEmpty()) {
            val values = valuesFloat ?: FloatArray(keysFloat.size) { it.toFloat() }
            
            var result = sd.onesLike("${opName}_default", input).mul(defaultFloat.toDouble())
            
            for (i in keysFloat.indices) {
                val key = keysFloat[i]
                val value = values.getOrElse(i) { i.toFloat() }
                val mask = sd.eq("${opName}_mask_$i", input, key.toDouble())
                val maskFloat = mask.castTo(result.dataType())
                val invMask = sd.math.sub("${opName}_inv_$i", 1.0, maskFloat)
                result = sd.math.add("${opName}_upd_$i",
                    sd.math.mul(result, invMask),
                    sd.math.mul(maskFloat, value.toDouble())
                )
            }
            
            val output = result.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        }
        
        // Default: return input as-is
        val output = input.rename(outputNames[0])
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
