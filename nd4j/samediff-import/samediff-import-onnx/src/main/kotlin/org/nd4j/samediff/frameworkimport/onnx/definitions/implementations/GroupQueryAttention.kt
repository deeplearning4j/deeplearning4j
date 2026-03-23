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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Microsoft ONNX GroupQueryAttention (GQA) operation.
 *
 * GroupQueryAttention is an efficient variant of multi-head attention where
 * key-value heads are shared across multiple query heads. This is used in
 * models like LLaMA 2, Mistral, etc.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#GroupQueryAttention
 *
 * Inputs:
 * - query: Query tensor [batch, seq_len, num_heads * head_dim]
 * - key: Key tensor [batch, kv_seq_len, num_kv_heads * head_dim]
 * - value: Value tensor [batch, kv_seq_len, num_kv_heads * head_dim]
 * - past_key: Optional cached key
 * - past_value: Optional cached value
 * - seqlens_k: Optional sequence lengths for key
 * - total_sequence_length: Optional total sequence length
 * - cos_cache: Optional rotary embedding cos cache
 * - sin_cache: Optional rotary embedding sin cache
 *
 * Attributes:
 * - num_heads: Number of query heads
 * - kv_num_heads: Number of key-value heads (< num_heads for GQA)
 * - scale: Optional scaling factor
 * - local_window_size: Optional local attention window size
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GroupQueryAttention"], frameworkName = "onnx")
class GroupQueryAttention : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val query = sd.getVariable(op.inputsToOp[0])
        var key = sd.getVariable(op.inputsToOp[1])
        var value = sd.getVariable(op.inputsToOp[2])

        // Optional inputs
        val hasPastKey = op.inputsToOp.size > 3 && op.inputsToOp[3] != null
        val pastKey = if (hasPastKey) sd.getVariable(op.inputsToOp[3]) else null

        val hasPastValue = op.inputsToOp.size > 4 && op.inputsToOp[4] != null
        val pastValue = if (hasPastValue) sd.getVariable(op.inputsToOp[4]) else null

        // Get attributes
        val numHeads = (attributes.getOrDefault("num_heads", 1) as Number).toInt()
        val kvNumHeads = (attributes.getOrDefault("kv_num_heads", numHeads) as Number).toInt()
        val scaleAttr = attributes["scale"]

        // Compute head_dim statically from attributes (num_heads and scale are known at import time)
        // scale = 1/sqrt(head_dim), so head_dim = 1/scale^2
        val headDim: Long = if (scaleAttr != null) {
            val s = (scaleAttr as Number).toDouble()
            Math.round(1.0 / (s * s))
        } else {
            // Fallback: query hidden_size / num_heads (static shape only as last resort)
            val queryShape = query.shape
            if (queryShape != null && queryShape.size >= 3) {
                queryShape[2] / numHeads
            } else {
                throw IllegalStateException("Cannot determine head_dim: no scale attribute and query shape unavailable")
            }
        }

        // Calculate scale
        val scale = if (scaleAttr != null) {
            (scaleAttr as Number).toDouble()
        } else {
            1.0 / kotlin.math.sqrt(headDim.toDouble())
        }

        val queryHiddenSize = (numHeads * headDim).toLong()

        // Number of query heads per kv head (for repeating kv heads)
        val numGroupsPerKvHead = numHeads / kvNumHeads

        // Use dynamic shapes via sd.shape() per CLAUDE.md rules
        val queryShapeVar = sd.shape(query)
        val batchSizeVar = sd.reshape(sd.slice(queryShapeVar, intArrayOf(0), 1), 1)
        val seqLenVar = sd.reshape(sd.slice(queryShapeVar, intArrayOf(1), 1), 1)

        // Reshape query: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        // Use -1 for batch dimension to support dynamic shapes
        val queryReshaped = sd.reshape(query, -1, 0, numHeads.toLong(), headDim)
        // Transpose to [batch, num_heads, seq, head_dim]
        val queryTransposed = sd.permute(queryReshaped, 0, 2, 1, 3)

        // Reshape key/value: [batch, kv_seq, kv_hidden] -> [batch, kv_seq, kv_num_heads, head_dim]
        val keyReshaped = sd.reshape(key, -1, 0, kvNumHeads.toLong(), headDim)
        val valueReshaped = sd.reshape(value, -1, 0, kvNumHeads.toLong(), headDim)

        // Transpose to [batch, kv_num_heads, kv_seq, head_dim]
        var keyTransposed = sd.permute(keyReshaped, 0, 2, 1, 3)
        var valueTransposed = sd.permute(valueReshaped, 0, 2, 1, 3)

        // Handle past key/value
        if (pastKey != null) {
            keyTransposed = sd.concat(2, pastKey, keyTransposed)
        }
        if (pastValue != null) {
            valueTransposed = sd.concat(2, pastValue, valueTransposed)
        }

        // Save pre-expansion key/value for present outputs (KV cache)
        // These have shape [batch, kv_num_heads, total_seq, head_dim] which is correct for caching
        val presentKey = keyTransposed
        val presentValue = valueTransposed

        // Expand key/value heads to match query heads (for GQA)
        if (numGroupsPerKvHead > 1) {
            // Repeat key and value heads
            // [batch, kv_num_heads, kv_seq, head_dim] -> [batch, num_heads, kv_seq, head_dim]
            keyTransposed = repeatKvHeads(sd, keyTransposed, numGroupsPerKvHead, kvNumHeads)
            valueTransposed = repeatKvHeads(sd, valueTransposed, numGroupsPerKvHead, kvNumHeads)
        }

        // Compute attention scores: Q @ K^T
        val keyTranspose = sd.permute(keyTransposed, 0, 1, 3, 2)
        var attentionScores = sd.mmul(queryTransposed, keyTranspose)

        // Apply scale
        attentionScores = sd.math.mul(attentionScores, scale)

        // Apply softmax
        val attentionProbs = sd.nn.softmax(attentionScores, -1)

        // Compute output: attention_probs @ V
        var output = sd.mmul(attentionProbs, valueTransposed)

        // Transpose back: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim]
        output = sd.permute(output, 0, 2, 1, 3)

        // Reshape to [batch, seq, hidden] using dynamic concat for shape
        val outputShapeVar = sd.concat(0, batchSizeVar, seqLenVar,
            sd.constant(org.nd4j.linalg.factory.Nd4j.createFromArray(queryHiddenSize)))
        output = sd.reshape(output, outputShapeVar)

        output.rename(outputNames[0])

        val results = mutableMapOf(outputNames[0] to listOf(output))

        // Output present key/value for caching if requested
        // IMPORTANT: Use pre-expansion key/value (kv_num_heads, NOT num_heads)
        // so the KV cache shape stays [batch, kv_num_heads, total_seq, head_dim]
        if (outputNames.size > 1 && hasPastKey) {
            presentKey.rename(outputNames[1])
            results[outputNames[1]] = listOf(presentKey)
        }
        if (outputNames.size > 2 && hasPastValue) {
            presentValue.rename(outputNames[2])
            results[outputNames[2]] = listOf(presentValue)
        }

        return results
    }

    /**
     * Repeat key/value heads to match the number of query heads.
     * Uses dynamic shapes via sd.shape() to support variable sequence lengths.
     *
     * @param kvNumHeads number of KV heads (known statically from attributes)
     */
    private fun repeatKvHeads(sd: SameDiff, x: SDVariable, numRepeats: Int, kvNumHeads: Int): SDVariable {
        if (numRepeats == 1) return x

        // x shape: [batch, kv_num_heads, seq, head_dim]
        // Expand: [batch, kv_num_heads, 1, seq, head_dim]
        val expanded = sd.expandDims(x, 2)

        // Tile along the new dimension: [batch, kv_num_heads, numRepeats, seq, head_dim]
        val tiled = sd.tile(expanded, 1, 1, numRepeats, 1, 1)

        // Reshape to [batch, num_heads, seq, head_dim] using dynamic shape
        // batch and seq are dynamic, num_heads and head_dim are static
        val numTotalHeads = (kvNumHeads * numRepeats).toLong()
        // Extract shape from the tiled tensor itself (rank 5: [batch, kvNumHeads, numRepeats, seq, headDim])
        // This is more reliable than reading from x, because tiled is in the main computation path
        // and its shape is always correct after expandDims+tile.
        val tiledShapeVar = sd.shape(tiled)
        val batchVar = sd.reshape(sd.slice(tiledShapeVar, intArrayOf(0), 1), 1)
        val seqVar = sd.reshape(sd.slice(tiledShapeVar, intArrayOf(3), 1), 1)
        val headDimVar = sd.reshape(sd.slice(tiledShapeVar, intArrayOf(4), 1), 1)
        val targetShape = sd.concat(0, batchVar,
            sd.constant(org.nd4j.linalg.factory.Nd4j.createFromArray(numTotalHeads)),
            seqVar, headDimVar)
        return sd.reshape(tiled, targetShape)
    }
}
