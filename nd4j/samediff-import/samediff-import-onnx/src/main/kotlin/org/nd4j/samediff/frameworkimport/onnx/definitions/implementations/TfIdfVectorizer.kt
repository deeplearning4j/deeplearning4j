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
 * Implementation of ONNX TfIdfVectorizer operation (ML domain).
 *
 * ONNX TfIdfVectorizer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TfIdfVectorizer
 *
 * Computes TF-IDF (Term Frequency-Inverse Document Frequency) features.
 * This is commonly used for text feature extraction in NLP.
 *
 * Inputs:
 * - X: Input tensor (string or integer tokens)
 *
 * Attributes:
 * - max_gram_length: Maximum n-gram length
 * - max_skip_count: Maximum skip distance
 * - min_gram_length: Minimum n-gram length
 * - mode: TF, IDF, or TFIDF
 * - ngram_counts: N-gram counts per document
 * - ngram_indexes: N-gram vocabulary indexes
 * - pool_int64s/pool_strings: Vocabulary pool
 * - weights: IDF weights
 *
 * Note: Full TF-IDF computation requires text processing not available in SameDiff.
 * This implementation provides a simplified numeric-based approach.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["TfIdfVectorizer"], frameworkName = "onnx")
class TfIdfVectorizer : PreImportHook {

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
        val mode = attributes.getOrDefault("mode", "TF") as? String ?: "TF"
        val weights = getFloatArray(attributes, "weights")
        val ngramIndexes = getLongArray(attributes, "ngram_indexes")

        // Determine output vocabulary size
        val vocabSize = ngramIndexes?.size ?: weights?.size ?: 1

        // For integer inputs, we can compute a simplified TF representation
        // by counting occurrences using one-hot encoding and summing

        val xInt = x.castTo(DataType.INT64)

        // Create output based on mode
        val output = when (mode.uppercase()) {
            "TF" -> {
                // Term frequency: count occurrences
                // Simplified: use one-hot encoding if input is token indices
                if (vocabSize > 1) {
                    val oneHot = sd.oneHot("${opName}_onehot", xInt, vocabSize)
                    sd.math.sum("${opName}_tf", oneHot, false, 1)
                } else {
                    x.castTo(DataType.FLOAT)
                }
            }
            "IDF" -> {
                // Inverse document frequency
                if (weights != null && weights.isNotEmpty()) {
                    val idfWeights = sd.constant("${opName}_idf_weights",
                        Nd4j.createFromArray(*weights))
                    // Apply IDF weights via gather
                    val clipped = sd.math.clipByValue(xInt.castTo(DataType.FLOAT),
                        0.0, (weights.size - 1).toDouble())
                    sd.gather("${opName}_idf", idfWeights, clipped.castTo(DataType.INT64), 0)
                } else {
                    x.castTo(DataType.FLOAT)
                }
            }
            "TFIDF" -> {
                // TF * IDF combined
                if (vocabSize > 1 && weights != null) {
                    val oneHot = sd.oneHot("${opName}_onehot", xInt, vocabSize)
                    val tf = sd.math.sum("${opName}_tf", oneHot, false, 1)
                    val idfWeights = sd.constant("${opName}_idf_weights",
                        Nd4j.createFromArray(*weights))
                    sd.math.mul("${opName}_tfidf", tf, idfWeights)
                } else {
                    x.castTo(DataType.FLOAT)
                }
            }
            else -> x.castTo(DataType.FLOAT)
        }

        val finalOutput = output.rename(outputNames[0])

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
