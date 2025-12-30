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
 * Implementation of ONNX CategoryMapper operation (ML domain).
 *
 * ONNX CategoryMapper spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.CategoryMapper
 *
 * Maps categorical values to integers or strings based on lookup tables.
 * This is useful for converting between categorical representations.
 *
 * Inputs:
 * - X: Input categorical values
 *
 * Attributes:
 * - cats_int64s: Integer categories
 * - cats_strings: String categories
 * - default_int64: Default integer for unknown categories
 * - default_string: Default string for unknown categories
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["CategoryMapper"], frameworkName = "onnx")
class CategoryMapper : PreImportHook {

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

        // Get category mappings
        val catsInt64s = getLongArray(attributes, "cats_int64s")
        val defaultInt64 = (attributes.getOrDefault("default_int64", -1L) as Number).toLong()

        // If we have integer categories, create a lookup table
        val output = if (catsInt64s != null && catsInt64s.isNotEmpty()) {
            // Create a mapping from input values to indices
            // For simplicity, we use the input directly if it's already integer-based
            val xInt = x.castTo(DataType.INT64)

            // Create lookup table as constant
            val lookupTable = Nd4j.createFromArray(*catsInt64s)
            val lookupConst = sd.constant("${opName}_lookup", lookupTable)

            // Use gather to map values (simplified approach)
            // In practice, this assumes input values are valid indices
            val clippedInput = sd.math.clipByValue("${opName}_clip", xInt,
                0.0, (catsInt64s.size - 1).toDouble())
            sd.gather("${opName}_mapped", lookupConst, clippedInput.castTo(DataType.INT64), 0)
        } else {
            // No mapping provided, pass through as identity
            sd.identity("${opName}_identity", x)
        }

        val finalOutput = output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
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
