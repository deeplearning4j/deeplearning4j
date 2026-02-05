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
 * Implementation of Microsoft ONNX MultiHeadAttention operation.
 *
 * MultiHeadAttention computes multi-head attention with separate Q, K, V inputs:
 * output = softmax(Q @ K^T / sqrt(head_dim)) @ V
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#MultiHeadAttention
 *
 * Inputs:
 * - query: Query tensor [batch, seq_len, hidden_size]
 * - key: Key tensor [batch, kv_seq_len, hidden_size]
 * - value: Value tensor [batch, kv_seq_len, hidden_size]
 * - bias: Optional bias for QKV projections
 * - key_padding_mask: Optional mask for padding
 * - relative_position_bias: Optional relative position bias
 * - past_key: Optional cached key for incremental decoding
 * - past_value: Optional cached value for incremental decoding
 *
 * Attributes:
 * - num_heads: Number of attention heads
 * - mask_filter_value: Value for masked positions (default: -10000)
 * - scale: Scaling factor (default: 1/sqrt(head_dim))
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["MultiHeadAttention"], frameworkName = "onnx")
class MultiHeadAttention : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        var query = sd.getVariable(op.inputsToOp[0])
        var key = sd.getVariable(op.inputsToOp[1])
        var value = sd.getVariable(op.inputsToOp[2])

        // Optional inputs
        val hasBias = op.inputsToOp.size > 3 && op.inputsToOp[3] != null
        val bias = if (hasBias) sd.getVariable(op.inputsToOp[3]) else null

        val hasKeyPaddingMask = op.inputsToOp.size > 4 && op.inputsToOp[4] != null
        val keyPaddingMask = if (hasKeyPaddingMask) sd.getVariable(op.inputsToOp[4]) else null

        val hasRelativePosBias = op.inputsToOp.size > 5 && op.inputsToOp[5] != null
        val relativePosBias = if (hasRelativePosBias) sd.getVariable(op.inputsToOp[5]) else null

        val hasPastKey = op.inputsToOp.size > 6 && op.inputsToOp[6] != null
        val pastKey = if (hasPastKey) sd.getVariable(op.inputsToOp[6]) else null

        val hasPastValue = op.inputsToOp.size > 7 && op.inputsToOp[7] != null
        val pastValue = if (hasPastValue) sd.getVariable(op.inputsToOp[7]) else null

        // Get attributes
        val numHeads = (attributes.getOrDefault("num_heads", 1) as Number).toInt()
        val maskFilterValue = (attributes.getOrDefault("mask_filter_value", -10000.0) as Number).toDouble()
        val scaleAttr = attributes["scale"]

        // Get dimensions dynamically - shapes may not be statically known
        val queryShapeVar = sd.shape(query)
        val staticQueryShape = query.shape  // May be null for dynamic shapes

        // Extract dimensions using gather for dynamic shapes
        val idx0 = sd.constant(Nd4j.createFromArray(0))
        val idx1 = sd.constant(Nd4j.createFromArray(1))
        val idx2 = sd.constant(Nd4j.createFromArray(2))

        val batchSizeVar = sd.gather(queryShapeVar, idx0, 0)
        val seqLenVar = sd.gather(queryShapeVar, idx1, 0)
        val hiddenSizeVar = sd.gather(queryShapeVar, idx2, 0)
        val numHeadsConst = sd.constant(Nd4j.createFromArray(numHeads.toLong()))

        // Compute head_dim and scale. Prefer static sources over dynamic sd.math.div
        // because the dynamic division can produce 0 when shape ops evaluate in unexpected order.
        val scale: Double
        val headDimStatic: Long
        if (scaleAttr != null) {
            scale = (scaleAttr as Number).toDouble()
            // Derive head_dim from scale: scale = 1/sqrt(head_dim) → head_dim = 1/scale^2
            headDimStatic = kotlin.math.round(1.0 / (scale * scale)).toLong()
        } else if (staticQueryShape != null && staticQueryShape.size >= 3) {
            headDimStatic = staticQueryShape[2] / numHeads
            scale = 1.0 / kotlin.math.sqrt(headDimStatic.toDouble())
        } else {
            // Default for common head dimensions (64, 128)
            headDimStatic = 64L
            scale = 1.0 / kotlin.math.sqrt(headDimStatic.toDouble())
        }
        // Use static head_dim constant for KV reshape (which already uses -1 for seq_len)
        val headDimConst = sd.constant(Nd4j.createFromArray(headDimStatic))

        // Apply bias if present (typically already projected, so bias is applied to Q, K, V)
        if (bias != null) {
            // Bias is typically [3 * hidden_size] for Q, K, V
            // For dynamic shapes, we use dynamic slicing
            val biasShape = bias.shape
            if (biasShape != null && biasShape.size == 1) {
                val hiddenSize = biasShape[0] / 3
                val qBias = sd.stridedSlice(bias, longArrayOf(0), longArrayOf(hiddenSize), 1L)
                val kBias = sd.stridedSlice(bias, longArrayOf(hiddenSize), longArrayOf(2 * hiddenSize), 1L)
                val vBias = sd.stridedSlice(bias, longArrayOf(2 * hiddenSize), longArrayOf(3 * hiddenSize), 1L)
                query = sd.math.add("_q_bias", query, qBias)
                key = sd.math.add("_k_bias", key, kBias)
                value = sd.math.add("_v_bias", value, vBias)
            }
        }

        // Build reshape target shapes dynamically
        // [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        // Use -1 for head_dim in query (reshape infers it from total size / other dims).
        // Use static headDimConst for KV (which already uses -1 for seq_len, can't have two -1s).
        val negOne = sd.constant(Nd4j.createFromArray(-1L))
        val queryNewShape = sd.stack(0, batchSizeVar, seqLenVar, numHeadsConst, negOne)
        val kvNewShape = sd.stack(0, batchSizeVar, negOne, numHeadsConst, headDimConst)

        // Reshape for multi-head attention
        val queryReshaped = sd.reshape(query, queryNewShape)
        val keyReshaped = sd.reshape(key, kvNewShape)
        val valueReshaped = sd.reshape(value, kvNewShape)

        // Transpose to [batch, num_heads, seq, head_dim]
        val queryTransposed = sd.permute(queryReshaped, 0, 2, 1, 3)
        var keyTransposed = sd.permute(keyReshaped, 0, 2, 1, 3)
        var valueTransposed = sd.permute(valueReshaped, 0, 2, 1, 3)

        // Handle past key/value for incremental decoding
        if (pastKey != null) {
            keyTransposed = sd.concat(2, pastKey, keyTransposed)
        }
        if (pastValue != null) {
            valueTransposed = sd.concat(2, pastValue, valueTransposed)
        }

        // Compute attention scores: Q @ K^T
        val keyTranspose = sd.permute(keyTransposed, 0, 1, 3, 2)
        var attentionScores = sd.mmul(queryTransposed, keyTranspose)

        // Apply scale
        attentionScores = sd.math.mul(attentionScores, scale)

        // Apply relative position bias if present
        if (relativePosBias != null) {
            attentionScores = sd.math.add(attentionScores, relativePosBias)
        }

        // Apply key padding mask if present
        if (keyPaddingMask != null) {
            // Expand mask dimensions for broadcasting
            val maskExpanded = sd.expandDims(sd.expandDims(keyPaddingMask, 1), 2)
            val maskValue = sd.constant(maskFilterValue)
            // Where mask is 0, apply mask value
            attentionScores = sd.where(maskExpanded, attentionScores, maskValue)
        }

        // Apply softmax
        val attentionProbs = sd.nn.softmax(attentionScores, -1)

        // Compute output: attention_probs @ V
        var output = sd.mmul(attentionProbs, valueTransposed)

        // Transpose back: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim]
        output = sd.permute(output, 0, 2, 1, 3)

        // Reshape to [batch, seq, hidden] using dynamic shape
        val outputNewShape = sd.stack(0, batchSizeVar, seqLenVar, hiddenSizeVar)
        output = sd.reshape(output, outputNewShape)

        output.rename(outputNames[0])

        val results = mutableMapOf(outputNames[0] to listOf(output))

        // Output present key/value for caching if requested
        // Check for non-empty output names (empty string means output is not used)
        if (outputNames.size > 1 && outputNames[1].isNotEmpty()) {
            keyTransposed.rename(outputNames[1])
            results[outputNames[1]] = listOf(keyTransposed)
        }
        if (outputNames.size > 2 && outputNames[2].isNotEmpty()) {
            valueTransposed.rename(outputNames[2])
            results[outputNames[2]] = listOf(valueTransposed)
        }

        return results
    }
}
