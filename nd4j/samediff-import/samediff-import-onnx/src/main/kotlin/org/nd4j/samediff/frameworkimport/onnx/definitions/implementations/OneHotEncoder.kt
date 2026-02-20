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
 * Implementation of ONNX OneHotEncoder operation (ML domain).
 *
 * ONNX OneHotEncoder spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.OneHotEncoder
 *
 * Encodes categorical features as one-hot numeric arrays.
 *
 * Inputs:
 * - X: Input tensor of integers
 *
 * Attributes:
 * - cats_int64s: List of categories (integers)
 * - cats_strings: List of categories (strings) - not supported in numeric mode
 * - zeros: If true, include category 0 in encoding (default: true)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["OneHotEncoder"], frameworkName = "onnx")
class OneHotEncoder : PreImportHook {

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
        
        // Get categories
        val catsInt = getLongArray(attributes, "cats_int64s")
        val zeros = (attributes.getOrDefault("zeros", 1L) as Number).toInt() == 1
        
        // Determine depth (number of categories)
        val depthInt = if (catsInt != null && catsInt.isNotEmpty()) {
            catsInt.size
        } else {
            // If no categories specified, use a default value
            // since we need static int for oneHot depth
            val inputShape = input.shape
            if (inputShape != null && inputShape.isNotEmpty()) {
                inputShape[inputShape.size - 1].toInt().coerceAtLeast(10)
            } else {
                10  // Default depth
            }
        }
        
        // Convert input to indices if categories are specified
        val indices = if (catsInt != null && catsInt.isNotEmpty()) {
            // Map input values to category indices
            var result = sd.zerosLike("${opName}_idx", input)
            for (i in catsInt.indices) {
                val cat = catsInt[i]
                val mask = sd.eq("${opName}_cat_$i", input, cat.toDouble())
                val maskInt = mask.castTo(DataType.INT64)
                result = sd.math.add("${opName}_idx_$i", result, sd.math.mul(maskInt, i.toLong().toDouble()))
            }
            result.castTo(DataType.INT32)
        } else {
            input.castTo(DataType.INT32)
        }
        
        // Create one-hot encoding
        val oneHot = sd.oneHot("${opName}_onehot", indices, depthInt, -1, 1.0, 0.0, DataType.FLOAT)

        // Flatten last dimensions for 2D output: [N, D] -> [N, num_categories]
        val output = sd.reshape(outputNames[0], oneHot, -1L, depthInt.toLong())
        
        return mapOf(outputNames[0] to listOf(output))
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
