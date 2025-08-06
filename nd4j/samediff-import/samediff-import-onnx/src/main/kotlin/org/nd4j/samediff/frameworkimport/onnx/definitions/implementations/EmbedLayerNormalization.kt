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
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX EmbedLayerNormalization operation.
 *
 * This operation combines embedding lookups for word, position, and optionally segment embeddings,
 * sums them together, and applies layer normalization.
 *
 * Typical transformer architecture embedding layer as used in BERT-like models.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["EmbedLayerNormalization"],frameworkName = "onnx")
class EmbedLayerNormalization: PreImportHook {

    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>,
        isFinalOutput: Boolean,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): HookResult {
        val ret = HookResult(outputVariables = handleOutputs(
            outputNames,
            sd,
            op,
            attributes,
            mappingRegistry,
            importGraph,
            dynamicVariables
        ), proceedWithInit = false)

        // Don't try to rename ops in this case since we create complex sub-graphs
        return ret
    }

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        // ONNX EmbedLayerNormalization inputs:
        // input_ids: [batch_size, sequence_length] - Token indices
        // segment_ids: [batch_size, sequence_length] - Token type indices (optional, can be null)
        // word_embedding: [vocab_size, hidden_size] - Word embedding matrix
        // position_embedding: [max_position_embeddings, hidden_size] - Position embedding matrix
        // segment_embedding: [type_vocab_size, hidden_size] - Segment embedding matrix (optional)
        // gamma: [hidden_size] - Layer normalization weight
        // beta: [hidden_size] - Layer normalization bias
        // mask: [batch_size, sequence_length] - Attention mask (optional)

        val inputIds = sd.getVariable(op.inputsToOp[0])
        var segmentIds: SDVariable? = null
        var maskInput: SDVariable? = null

        // Handle optional segment_ids input
        if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null) {
            segmentIds = sd.getVariable(op.inputsToOp[1])
        }

        val wordEmbedding = sd.getVariable(op.inputsToOp[2])
        val positionEmbedding = sd.getVariable(op.inputsToOp[3])

        var segmentEmbedding: SDVariable? = null
        var gamma: SDVariable
        var beta: SDVariable

        // Handle variable number of inputs - segment embedding may or may not be present
        when {
            op.inputsToOp.size >= 7 -> {
                // All inputs present: input_ids, segment_ids, word_emb, pos_emb, segment_emb, gamma, beta, [mask]
                segmentEmbedding = sd.getVariable(op.inputsToOp[4])
                gamma = sd.getVariable(op.inputsToOp[5])
                beta = sd.getVariable(op.inputsToOp[6])
                if (op.inputsToOp.size > 7) {
                    maskInput = sd.getVariable(op.inputsToOp[7])
                }
            }
            op.inputsToOp.size >= 6 -> {
                // No segment embedding: input_ids, segment_ids, word_emb, pos_emb, gamma, beta, [mask]
                gamma = sd.getVariable(op.inputsToOp[4])
                beta = sd.getVariable(op.inputsToOp[5])
                if (op.inputsToOp.size > 6) {
                    maskInput = sd.getVariable(op.inputsToOp[6])
                }
            }
            else -> {
                // Minimum required inputs: input_ids, [segment_ids], word_emb, pos_emb, gamma, beta
                gamma = sd.getVariable(op.inputsToOp[op.inputsToOp.size - 2])
                beta = sd.getVariable(op.inputsToOp[op.inputsToOp.size - 1])
            }
        }

        val epsilon = attributes.getOrDefault("epsilon", 1e-12) as Number

        // Get sequence length for position embeddings
        val seqLen = sd.sizeAt(inputIds,1)

        // Ensure inputIds are integer type for gather operation
        val inputIdsInt = if (inputIds.dataType().isIntType()) {
            inputIds
        } else {
            inputIds.castTo(DataType.INT32)
        }

        // Word embedding lookup: [batch_size, seq_len, hidden_size]
        val wordEmbedded = sd.gather(wordEmbedding, inputIdsInt, 0)

        // Position embedding lookup
        // Create position indices [0, 1, 2, ..., seq_len-1]
        val positionIds = sd.range(sd.constant(0.0), seqLen.castTo(DataType.DOUBLE), sd.constant(1.0), DataType.INT32)
        val positionEmbedded = sd.gather(positionEmbedding, positionIds, 0)

        // Position embeddings are [seq_len, hidden_size]
        // We need to expand to [1, seq_len, hidden_size] to broadcast with [batch_size, seq_len, hidden_size]
        val positionEmbeddedExpanded = sd.expandDims(positionEmbedded, 0)

        // Sum word and position embeddings (broadcasting handles the batch dimension)
        var embeddings = wordEmbedded.add(positionEmbeddedExpanded)

        // Add segment embeddings if present
        if (segmentIds != null && segmentEmbedding != null) {
            // Ensure segmentIds are integer type for gather operation
            val segmentIdsInt = if (segmentIds.dataType().isIntType()) {
                segmentIds
            } else {
                segmentIds.castTo(DataType.INT32)
            }
            val segmentEmbedded = sd.gather(segmentEmbedding, segmentIdsInt, 0)
            embeddings = embeddings.add(segmentEmbedded)
        }

        // Apply Layer Normalization: LayerNorm(embeddings)
        // Layer norm is typically applied over the last dimension (hidden_size)
        val normalizedEmbeddings = sd.nn().layerNorm(outputNames[0], embeddings, gamma, beta, false, -1)

        // Handle mask output if needed
        val outputVars = mutableListOf<SDVariable>()
        outputVars.add(normalizedEmbeddings)

        // If mask was provided, compute mask index (typically used for attention)
        if (maskInput != null && outputNames.size > 1) {
            // For now, just pass through the mask as-is since mask processing can vary
            // In some cases, this might involve computing valid sequence lengths or other mask transformations
            outputVars.add(maskInput)
        }

        return if (outputNames.size > 1) {
            // Always create the second output even if no mask input
            val maskOutput = maskInput ?: sd.zerosLike(inputIds).castTo(DataType.INT32)
            mapOf(
                outputNames[0] to listOf(normalizedEmbeddings.rename(outputNames[0])),
                outputNames[1] to listOf(maskOutput.rename(outputNames[1]))
            )
        } else {
            mapOf(outputNames[0] to listOf(normalizedEmbeddings))
        }
    }
}