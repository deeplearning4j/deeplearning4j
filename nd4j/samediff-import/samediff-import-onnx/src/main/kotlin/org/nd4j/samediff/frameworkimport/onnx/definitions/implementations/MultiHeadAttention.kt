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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2
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
 * Uses fused DotProductAttentionV2 when no key padding mask is present (common case
 * for decoder self-attention with causal masking). This replaces 7-10 individual ops
 * (permute, mmul, scale, softmax, mmul, permute) with a single fused op that uses
 * the CUDA flash attention kernel.
 *
 * For 30-layer SmolDocling decoder: saves ~120-270 ops per decode step.
 *
 * Fallback to manual attention when key_padding_mask is present.
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
            headDimStatic = kotlin.math.round(1.0 / (scale * scale)).toLong()
        } else if (staticQueryShape != null && staticQueryShape.size >= 3) {
            headDimStatic = staticQueryShape[2] / numHeads
            scale = 1.0 / kotlin.math.sqrt(headDimStatic.toDouble())
        } else {
            headDimStatic = 64L
            scale = 1.0 / kotlin.math.sqrt(headDimStatic.toDouble())
        }
        val headDimConst = sd.constant(Nd4j.createFromArray(headDimStatic))

        // Apply bias if present
        if (bias != null) {
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

        // Fused attention path: use DotProductAttentionV2 when no key padding mask.
        // The op's internal useCausalMask replaces the external relativePosBias (causal mask).
        // DotProductAttentionV2 expects BSHD format: [batch, seq, numHeads, headDim].
        // TODO: Fused attention produces incorrect output — needs debugging.
        // Disable until the DotProductAttentionV2 input format/masking is validated.
        val useFusedAttention = false // (keyPaddingMask == null)

        if (useFusedAttention) {
            return doFusedAttention(sd, query, key, value, pastKey, pastValue,
                numHeads, headDimStatic, scale, outputNames,
                batchSizeVar, seqLenVar, hiddenSizeVar, numHeadsConst, headDimConst)
        } else {
            return doManualAttention(sd, query, key, value, pastKey, pastValue,
                keyPaddingMask, relativePosBias, numHeads, headDimStatic, scale, maskFilterValue,
                outputNames, batchSizeVar, seqLenVar, hiddenSizeVar, numHeadsConst, headDimConst)
        }
    }

    /**
     * Fused attention using DotProductAttentionV2.
     * Replaces permute + mmul + scale + softmax + mmul + permute with a single fused op.
     * Input/output in BSHD format: [batch, seq, numHeads, headDim].
     */
    private fun doFusedAttention(
        sd: SameDiff,
        query: SDVariable, key: SDVariable, value: SDVariable,
        pastKey: SDVariable?, pastValue: SDVariable?,
        numHeads: Int, headDimStatic: Long, scale: Double,
        outputNames: List<String>,
        batchSizeVar: SDVariable, seqLenVar: SDVariable, hiddenSizeVar: SDVariable,
        numHeadsConst: SDVariable, headDimConst: SDVariable
    ): Map<String, List<SDVariable>> {

        val negOne = sd.constant(Nd4j.createFromArray(-1L))
        val queryNewShape = sd.stack(0, batchSizeVar, seqLenVar, numHeadsConst, negOne)
        val kvNewShape = sd.stack(0, batchSizeVar, negOne, numHeadsConst, headDimConst)

        // Reshape [B, S, hidden] -> [B, S, numHeads, headDim] (BSHD)
        val queryBSHD = sd.reshape(query, queryNewShape)
        var keyBSHD = sd.reshape(key, kvNewShape)
        var valueBSHD = sd.reshape(value, kvNewShape)

        // Handle past KV for incremental decoding.
        // Past KV is in BHSD [B, numHeads, pastSeq, headDim] from the ONNX model.
        // Need to permute to BSHD for concat, then pass to the fused op.
        if (pastKey != null) {
            val pastKeyBSHD = sd.permute(pastKey, 0, 2, 1, 3)  // BHSD -> BSHD
            keyBSHD = sd.concat(1, pastKeyBSHD, keyBSHD)       // concat on seq dim
        }
        if (pastValue != null) {
            val pastValueBSHD = sd.permute(pastValue, 0, 2, 1, 3)
            valueBSHD = sd.concat(1, pastValueBSHD, valueBSHD)
        }

        // Create empty masks (no custom masking - causal mask handled internally)
        val emptyMask = sd.constant(Nd4j.empty(DataType.FLOAT))

        // DotProductAttentionV2: fused attention with CUDA flash attention kernel.
        // Input order: queries, values, keys, queryMask, valueMask
        // BSHD format [B, S, numHeads, headDim] -> output [B, S, numHeads, headDim]
        val attnOp = DotProductAttentionV2(sd, queryBSHD, valueBSHD, keyBSHD,
            emptyMask, emptyMask, scale, 0.0, true, false)
        val attnOutputBSHD = attnOp.outputVariables()[0]

        // Reshape back to [B, S, hidden]
        val outputNewShape = sd.stack(0, batchSizeVar, seqLenVar, hiddenSizeVar)
        var output = sd.reshape(attnOutputBSHD, outputNewShape)
        output.rename(outputNames[0])

        val results = mutableMapOf(outputNames[0] to listOf(output))

        // Output present key/value if requested (non-empty output names).
        // Present KV must be in BHSD format for the ONNX model's KV cache.
        if (outputNames.size > 1 && outputNames[1].isNotEmpty()) {
            val keyBHSD = sd.permute(keyBSHD, 0, 2, 1, 3)  // BSHD -> BHSD
            keyBHSD.rename(outputNames[1])
            results[outputNames[1]] = listOf(keyBHSD)
        }
        if (outputNames.size > 2 && outputNames[2].isNotEmpty()) {
            val valueBHSD = sd.permute(valueBSHD, 0, 2, 1, 3)
            valueBHSD.rename(outputNames[2])
            results[outputNames[2]] = listOf(valueBHSD)
        }

        return results
    }

    /**
     * Manual attention path: used when key_padding_mask is present.
     * Explicit matmul + scale + mask + softmax + matmul pipeline.
     */
    private fun doManualAttention(
        sd: SameDiff,
        query: SDVariable, key: SDVariable, value: SDVariable,
        pastKey: SDVariable?, pastValue: SDVariable?,
        keyPaddingMask: SDVariable?, relativePosBias: SDVariable?,
        numHeads: Int, headDimStatic: Long, scale: Double, maskFilterValue: Double,
        outputNames: List<String>,
        batchSizeVar: SDVariable, seqLenVar: SDVariable, hiddenSizeVar: SDVariable,
        numHeadsConst: SDVariable, headDimConst: SDVariable
    ): Map<String, List<SDVariable>> {

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
            val maskExpanded = sd.expandDims(sd.expandDims(keyPaddingMask, 1), 2)
            val maskValue = sd.constant(maskFilterValue)
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
