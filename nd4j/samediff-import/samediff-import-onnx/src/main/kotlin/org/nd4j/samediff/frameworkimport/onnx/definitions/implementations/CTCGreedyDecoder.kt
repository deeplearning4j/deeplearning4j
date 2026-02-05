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
 * Implementation of CTC Greedy Decoder operation for OCR models.
 *
 * CTCGreedyDecoder performs greedy decoding on the output of a CTC network.
 * It selects the most likely token at each timestep and merges repeated tokens.
 *
 * This is commonly used in OCR models like PaddleOCR for text recognition.
 *
 * Inputs:
 * - inputs: Logits from the network [batch, time, num_classes] or [time, batch, num_classes]
 * - sequence_length: Optional sequence lengths [batch]
 *
 * Attributes:
 * - merge_repeated: Whether to merge repeated tokens (default: true)
 * - blank_index: Index of the blank label (default: 0 or num_classes-1)
 *
 * Outputs:
 * - decoded_indices: Indices of decoded tokens
 * - decoded_values: Values of decoded tokens
 * - decoded_shape: Shape of the decoded output
 * - log_probability: Log probability of the decoded sequence
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["CTCGreedyDecoder", "CTC_greedy_decoder"], frameworkName = "onnx")
class CTCGreedyDecoder : PreImportHook {

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

        // Get input logits - shape [batch, time, num_classes] or [time, batch, num_classes]
        val inputs = sd.getVariable(op.inputsToOp[0])

        // Optional sequence length input
        val hasSeqLen = op.inputsToOp.size > 1 && op.inputsToOp[1] != null && op.inputsToOp[1].isNotEmpty()
        val sequenceLength = if (hasSeqLen) sd.getVariable(op.inputsToOp[1]) else null

        // Get attributes
        val mergeRepeated = (attributes.getOrDefault("merge_repeated", 1L) as Number).toInt() == 1
        val blankIndex = (attributes.getOrDefault("blank_index", -1L) as Number).toInt()

        // Get input shape
        val inputRank = inputs.shape?.size ?: 3

        // Determine if input is [batch, time, classes] or [time, batch, classes]
        // We assume [batch, time, num_classes] format (more common in ONNX)
        val batchSize = sd.sizeAt("${opName}_batchSize", inputs, 0)
        val timeSteps = sd.sizeAt("${opName}_timeSteps", inputs, 1)
        val numClasses = sd.sizeAt("${opName}_numClasses", inputs, 2)

        // Determine blank index (often last class or 0)
        val actualBlankIndex = if (blankIndex < 0) {
            // Use num_classes - 1 as blank
            sd.math.sub("${opName}_blankIndex", numClasses, sd.constant(1L))
        } else {
            sd.constant("${opName}_blankIndex", blankIndex.toLong())
        }

        // Apply softmax to get probabilities if not already applied
        val probs = sd.nn.softmax("${opName}_probs", inputs, -1)

        // Get argmax for each timestep - greedy selection
        val bestPath = sd.argmax("${opName}_bestPath", probs, false, -1)
        val bestPathLong = sd.castTo("${opName}_bestPathLong", bestPath, DataType.INT64)

        // Get max log probabilities for each timestep
        val maxProbs = sd.max("${opName}_maxProbs", probs, false, -1)
        val logProbs = sd.math.log("${opName}_logProbs", maxProbs)

        // Sum log probabilities along time dimension for total log probability
        val totalLogProb = sd.sum("${opName}_totalLogProb", logProbs, false, 1)

        // Create decoded output
        // For greedy decoding with merge_repeated:
        // 1. Remove consecutive duplicates
        // 2. Remove blank tokens

        if (mergeRepeated) {
            // Shift the path to detect changes
            val zero = sd.constant("${opName}_zero", 0L)
            val one = sd.constant("${opName}_one", 1L)

            // Pad with -1 at the beginning to detect first token
            val minusOne = sd.constant("${opName}_minusOne", -1L)
            val paddedShape = sd.stack("${opName}_paddedShape", 0, batchSize, one)
            val padding = sd.fill("${opName}_padding", paddedShape, DataType.INT64, -1.0)

            // Concatenate padding with best path
            val paddedPath = sd.concat("${opName}_paddedPath", 1, padding, bestPathLong)

            // Create shifted version (without last element)
            val timeMinusOne = sd.math.sub("${opName}_timeMinusOne", timeSteps, one)
            val shiftedPath = sd.stridedSlice("${opName}_shiftedPath", paddedPath,
                longArrayOf(0, 0),
                longArrayOf(Long.MAX_VALUE, Long.MAX_VALUE),
                longArrayOf(1, 1),
                0, 0, 0, 0, 1)

            // Create mask where token differs from previous
            val diffMask = sd.neq("${opName}_diffMask", bestPathLong, shiftedPath)
            val diffMaskFloat = sd.castTo("${opName}_diffMaskFloat", diffMask, DataType.FLOAT)

            // Create mask where token is not blank
            val blankIndexExpanded = sd.expandDims(sd.expandDims(actualBlankIndex, 0), 0)
            val repeatCounts = sd.stack("${opName}_repeatCounts", 0,
                sd.castTo(batchSize, DataType.INT64), sd.castTo(timeSteps, DataType.INT64))
            val blankIndexTiled = sd.tile("${opName}_blankIndexTiled", blankIndexExpanded, repeatCounts)
            val notBlankMask = sd.neq("${opName}_notBlankMask", bestPathLong,
                sd.castTo(blankIndexTiled, DataType.INT64))
            val notBlankMaskFloat = sd.castTo("${opName}_notBlankMaskFloat", notBlankMask, DataType.FLOAT)

            // Combined mask: different from previous AND not blank
            val validMask = sd.math.mul("${opName}_validMask", diffMaskFloat, notBlankMaskFloat)

            // Apply mask to get decoded sequence
            // Multiply path by mask, keeping only valid tokens
            val maskedPath = sd.math.mul("${opName}_maskedPath",
                sd.castTo(bestPathLong, DataType.FLOAT), validMask)

            // The decoded output preserves positions but zeros out invalid tokens
            val decodedValues = sd.castTo("${opName}_decodedValues", maskedPath, DataType.INT64)
            decodedValues.rename(outputNames[0])

            val results = mutableMapOf(outputNames[0] to listOf(decodedValues))

            // Output log probability if requested
            if (outputNames.size > 1) {
                totalLogProb.rename(outputNames[1])
                results[outputNames[1]] = listOf(totalLogProb)
            }

            return results
        } else {
            // Without merge_repeated, just remove blanks
            val blankIndexExpanded2 = sd.expandDims(sd.expandDims(actualBlankIndex, 0), 0)
            val repeatCounts2 = sd.stack("${opName}_repeatCounts2", 0,
                sd.castTo(batchSize, DataType.INT64), sd.castTo(timeSteps, DataType.INT64))
            val blankIndexTiled2 = sd.tile("${opName}_blankTiled", blankIndexExpanded2, repeatCounts2)
            val notBlankMask = sd.neq("${opName}_notBlankMask", bestPathLong,
                sd.castTo(blankIndexTiled2, DataType.INT64))
            val notBlankMaskFloat = sd.castTo("${opName}_notBlankMaskFloat", notBlankMask, DataType.FLOAT)

            val maskedPath = sd.math.mul("${opName}_maskedPath",
                sd.castTo(bestPathLong, DataType.FLOAT), notBlankMaskFloat)
            val decodedValues = sd.castTo("${opName}_decodedValues", maskedPath, DataType.INT64)

            decodedValues.rename(outputNames[0])

            val results = mutableMapOf(outputNames[0] to listOf(decodedValues))

            if (outputNames.size > 1) {
                totalLogProb.rename(outputNames[1])
                results[outputNames[1]] = listOf(totalLogProb)
            }

            return results
        }
    }
}
