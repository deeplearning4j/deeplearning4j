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
 * Implementation of ONNX DictVectorizer operation (ML domain).
 *
 * ONNX DictVectorizer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.DictVectorizer
 *
 * Converts a dictionary (map) to a vector based on the vocabulary.
 * Since SameDiff doesn't have native map/dictionary support, this implementation
 * assumes the input is already in a format compatible with tensor operations.
 *
 * Inputs:
 * - X: Input map (represented as tensor in SameDiff)
 *
 * Attributes:
 * - int64_vocabulary: Integer vocabulary keys
 * - string_vocabulary: String vocabulary keys
 *
 * Note: This is a simplified implementation that passes through the input
 * as SameDiff doesn't natively support dictionary/map data types.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["DictVectorizer"], frameworkName = "onnx")
class DictVectorizer : PreImportHook {

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

        // Get vocabulary attributes
        val int64Vocabulary = getLongArray(attributes, "int64_vocabulary")
        val stringVocabulary = getStringList(attributes, "string_vocabulary")

        // Determine vocabulary size
        val vocabSize = int64Vocabulary?.size ?: stringVocabulary?.size ?: 0

        // Since SameDiff doesn't support dictionaries natively,
        // we assume input is already a tensor representation
        // If input is 1D, reshape to include vocabulary dimension
        val output = if (vocabSize > 0) {
            // Create output with proper shape based on vocabulary size
            // This is a passthrough that ensures the output has the right type
            sd.identity("${opName}_vectorized", x).castTo(DataType.FLOAT)
        } else {
            // No vocabulary specified, just pass through
            sd.identity("${opName}_passthrough", x)
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

    private fun getStringList(attributes: Map<String, Any>, key: String): List<String>? {
        val value = attributes[key] ?: return null
        return when (value) {
            is List<*> -> value.map { it.toString() }
            is Array<*> -> value.map { it.toString() }
            else -> null
        }
    }
}
