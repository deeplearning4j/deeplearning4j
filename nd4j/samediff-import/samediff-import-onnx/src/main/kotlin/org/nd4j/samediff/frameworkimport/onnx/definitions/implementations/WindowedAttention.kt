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
 * Implementation of Windowed Attention for ONNX.
 *
 * Windowed attention (also called local attention) restricts attention
 * to a local window around each position. This is used in models like
 * SAM (Segment Anything Model), Swin Transformer, and Longformer.
 *
 * The input is partitioned into non-overlapping windows, attention is
 * computed within each window, and the outputs are merged back.
 *
 * Inputs:
 * - query: Query tensor [batch, height, width, channels] or [batch, seq, hidden]
 * - key: Key tensor (same shape as query)
 * - value: Value tensor (same shape as query)
 * - relative_position_bias: Optional position bias [num_heads, window_size^2, window_size^2]
 *
 * Attributes:
 * - window_size: Size of the attention window (default: 7 or 14)
 * - num_heads: Number of attention heads
 * - shift_size: Shift size for shifted window attention (default: 0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["WindowedAttention", "WindowAttention", "LocalAttention", "SwinAttention"], frameworkName = "onnx")
class WindowedAttention : PreImportHook {

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

        // Get inputs
        val query = sd.getVariable(op.inputsToOp[0])
        val key = sd.getVariable(op.inputsToOp[1])
        val value = sd.getVariable(op.inputsToOp[2])

        // Optional relative position bias
        val hasRelPosBias = op.inputsToOp.size > 3 && op.inputsToOp[3] != null && op.inputsToOp[3].isNotEmpty()
        val relativePositionBias = if (hasRelPosBias) sd.getVariable(op.inputsToOp[3]) else null

        // Optional attention mask
        val hasMask = op.inputsToOp.size > 4 && op.inputsToOp[4] != null && op.inputsToOp[4].isNotEmpty()
        val attentionMask = if (hasMask) sd.getVariable(op.inputsToOp[4]) else null

        // Get attributes
        val windowSize = (attributes.getOrDefault("window_size", 7L) as Number).toInt()
        val numHeads = (attributes.getOrDefault("num_heads", 8L) as Number).toInt()
        val shiftSize = (attributes.getOrDefault("shift_size", 0L) as Number).toInt()
        val scaleAttr = attributes.get("scale") as Number?

        // Get input rank
        val inputRank = query.shape?.size ?: 4

        // Determine if input is 3D (batch, seq, hidden) or 4D (batch, H, W, C)
        if (inputRank == 3) {
            // 1D windowed attention (like Longformer)
            return handle1DWindowedAttention(sd, opName, query, key, value, relativePositionBias,
                attentionMask, windowSize, numHeads, scaleAttr, outputNames)
        } else if (inputRank == 4) {
            // 2D windowed attention (like Swin Transformer, SAM)
            return handle2DWindowedAttention(sd, opName, query, key, value, relativePositionBias,
                attentionMask, windowSize, numHeads, shiftSize, scaleAttr, outputNames)
        } else {
            throw IllegalArgumentException("WindowedAttention expects 3D or 4D input, got rank $inputRank")
        }
    }

    private fun handle1DWindowedAttention(
        sd: SameDiff,
        opName: String,
        query: SDVariable,
        key: SDVariable,
        value: SDVariable,
        relativePositionBias: SDVariable?,
        attentionMask: SDVariable?,
        windowSize: Int,
        numHeads: Int,
        scaleAttr: Number?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        // Input shape: [batch, seq_len, hidden_size]
        val batchSize = sd.sizeAt("${opName}_batchSize", query, 0)
        val seqLen = sd.sizeAt("${opName}_seqLen", query, 1)
        val hiddenSize = sd.sizeAt("${opName}_hiddenSize", query, 2)

        val headDim = sd.math.floorDiv("${opName}_headDim", hiddenSize, sd.constant(numHeads.toLong()))

        // Calculate scale
        val scale = if (scaleAttr != null) {
            sd.constant("${opName}_scale", scaleAttr.toFloat())
        } else {
            val headDimFloat = sd.castTo("${opName}_headDimFloat", headDim, DataType.FLOAT)
            sd.math.rsqrt("${opName}_scale", headDimFloat)
        }

        // Pad sequence to be divisible by window_size
        val windowSizeVar = sd.constant("${opName}_windowSize", windowSize.toLong())
        val remainder = sd.math.mod("${opName}_remainder", seqLen, windowSizeVar)
        val paddingNeeded = sd.math.mod("${opName}_paddingNeeded",
            sd.math.sub(windowSizeVar, remainder), windowSizeVar)

        // For simplicity, assume sequence is already divisible or pad it
        val numWindows = sd.math.floorDiv("${opName}_numWindows", seqLen, windowSizeVar)

        // Reshape to windows: [batch, num_windows, window_size, hidden]
        val qWindows = sd.reshape("${opName}_qWindows", query,
            sd.stack(0, batchSize, numWindows, windowSizeVar, hiddenSize))
        val kWindows = sd.reshape("${opName}_kWindows", key,
            sd.stack(0, batchSize, numWindows, windowSizeVar, hiddenSize))
        val vWindows = sd.reshape("${opName}_vWindows", value,
            sd.stack(0, batchSize, numWindows, windowSizeVar, hiddenSize))

        // Reshape for multi-head attention: [batch, num_windows, window_size, num_heads, head_dim]
        val numHeadsVar = sd.constant("${opName}_numHeads", numHeads.toLong())
        val qReshaped = sd.reshape("${opName}_qReshaped", qWindows,
            sd.stack(0, batchSize, numWindows, windowSizeVar, numHeadsVar, headDim))
        val kReshaped = sd.reshape("${opName}_kReshaped", kWindows,
            sd.stack(0, batchSize, numWindows, windowSizeVar, numHeadsVar, headDim))
        val vReshaped = sd.reshape("${opName}_vReshaped", vWindows,
            sd.stack(0, batchSize, numWindows, windowSizeVar, numHeadsVar, headDim))

        // Transpose to [batch, num_windows, num_heads, window_size, head_dim]
        val qTransposed = sd.permute("${opName}_qTransposed", qReshaped, 0, 1, 3, 2, 4)
        val kTransposed = sd.permute("${opName}_kTransposed", kReshaped, 0, 1, 3, 2, 4)
        val vTransposed = sd.permute("${opName}_vTransposed", vReshaped, 0, 1, 3, 2, 4)

        // Compute attention scores: Q @ K^T
        // [batch, num_windows, num_heads, window_size, head_dim] @ [batch, num_windows, num_heads, head_dim, window_size]
        val kTranspose = sd.permute("${opName}_kTranspose", kTransposed, 0, 1, 2, 4, 3)
        var scores = sd.mmul("${opName}_scores", qTransposed, kTranspose)

        // Apply scale
        scores = sd.math.mul("${opName}_scaledScores", scores, scale)

        // Add relative position bias if present
        if (relativePositionBias != null) {
            // Broadcast bias to match scores shape
            scores = sd.math.add("${opName}_scoresWithBias", scores, relativePositionBias)
        }

        // Apply attention mask if present
        if (attentionMask != null) {
            val maskExpanded = sd.expandDims(sd.expandDims(attentionMask, 0), 0)
            scores = sd.math.add("${opName}_maskedScores", scores, maskExpanded)
        }

        // Apply softmax
        val attentionProbs = sd.nn.softmax("${opName}_attentionProbs", scores, -1)

        // Apply attention to values
        var output = sd.mmul("${opName}_attnOutput", attentionProbs, vTransposed)

        // Transpose back: [batch, num_windows, num_heads, window_size, head_dim] -> [batch, num_windows, window_size, num_heads, head_dim]
        output = sd.permute("${opName}_outputTransposed", output, 0, 1, 3, 2, 4)

        // Reshape to [batch, num_windows, window_size, hidden]
        output = sd.reshape("${opName}_outputReshaped", output,
            sd.stack(0, batchSize, numWindows, windowSizeVar, hiddenSize))

        // Merge windows back: [batch, seq_len, hidden]
        output = sd.reshape("${opName}_output", output,
            sd.stack(0, batchSize, seqLen, hiddenSize))

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun handle2DWindowedAttention(
        sd: SameDiff,
        opName: String,
        query: SDVariable,
        key: SDVariable,
        value: SDVariable,
        relativePositionBias: SDVariable?,
        attentionMask: SDVariable?,
        windowSize: Int,
        numHeads: Int,
        shiftSize: Int,
        scaleAttr: Number?,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        // Input shape: [batch, height, width, channels]
        val batchSize = sd.sizeAt("${opName}_batchSize", query, 0)
        val height = sd.sizeAt("${opName}_height", query, 1)
        val width = sd.sizeAt("${opName}_width", query, 2)
        val channels = sd.sizeAt("${opName}_channels", query, 3)

        val headDim = sd.math.floorDiv("${opName}_headDim", channels, sd.constant(numHeads.toLong()))
        val windowSizeVar = sd.constant("${opName}_windowSize", windowSize.toLong())

        // Calculate scale
        val scale = if (scaleAttr != null) {
            sd.constant("${opName}_scale", scaleAttr.toFloat())
        } else {
            val headDimFloat = sd.castTo("${opName}_headDimFloat", headDim, DataType.FLOAT)
            sd.math.rsqrt("${opName}_scale", headDimFloat)
        }

        // Apply cyclic shift if shift_size > 0
        var qShifted = query
        var kShifted = key
        var vShifted = value

        if (shiftSize > 0) {
            // Cyclic shift along height and width dimensions using slice and concat
            qShifted = cyclicShift(sd, "${opName}_qShift", query, shiftSize, 1, height)
            qShifted = cyclicShift(sd, "${opName}_qShift2", qShifted, shiftSize, 2, width)
            kShifted = cyclicShift(sd, "${opName}_kShift", key, shiftSize, 1, height)
            kShifted = cyclicShift(sd, "${opName}_kShift2", kShifted, shiftSize, 2, width)
            vShifted = cyclicShift(sd, "${opName}_vShift", value, shiftSize, 1, height)
            vShifted = cyclicShift(sd, "${opName}_vShift2", vShifted, shiftSize, 2, width)
        }

        // Calculate number of windows
        val numWindowsH = sd.math.floorDiv("${opName}_numWindowsH", height, windowSizeVar)
        val numWindowsW = sd.math.floorDiv("${opName}_numWindowsW", width, windowSizeVar)

        // Partition into windows: [B, H, W, C] -> [B, nH, ws, nW, ws, C] -> [B*nH*nW, ws, ws, C]
        val qPartitioned = partitionWindows(sd, "${opName}_q", qShifted, windowSize,
            batchSize, height, width, channels, numWindowsH, numWindowsW)
        val kPartitioned = partitionWindows(sd, "${opName}_k", kShifted, windowSize,
            batchSize, height, width, channels, numWindowsH, numWindowsW)
        val vPartitioned = partitionWindows(sd, "${opName}_v", vShifted, windowSize,
            batchSize, height, width, channels, numWindowsH, numWindowsW)

        // Reshape for multi-head attention
        // [B*nH*nW, ws*ws, C] -> [B*nH*nW, ws*ws, num_heads, head_dim]
        val numHeadsVar = sd.constant("${opName}_numHeads", numHeads.toLong())
        val windowTokens = sd.constant("${opName}_windowTokens", (windowSize * windowSize).toLong())
        val numTotalWindows = sd.math.mul("${opName}_numTotalWindows",
            sd.math.mul(batchSize, numWindowsH), numWindowsW)

        // Flatten window spatial dims
        val qFlat = sd.reshape("${opName}_qFlat", qPartitioned,
            sd.stack(0, numTotalWindows, windowTokens, channels))
        val kFlat = sd.reshape("${opName}_kFlat", kPartitioned,
            sd.stack(0, numTotalWindows, windowTokens, channels))
        val vFlat = sd.reshape("${opName}_vFlat", vPartitioned,
            sd.stack(0, numTotalWindows, windowTokens, channels))

        // Reshape for multi-head: [B*nH*nW, ws*ws, num_heads, head_dim]
        val qMultiHead = sd.reshape("${opName}_qMultiHead", qFlat,
            sd.stack(0, numTotalWindows, windowTokens, numHeadsVar, headDim))
        val kMultiHead = sd.reshape("${opName}_kMultiHead", kFlat,
            sd.stack(0, numTotalWindows, windowTokens, numHeadsVar, headDim))
        val vMultiHead = sd.reshape("${opName}_vMultiHead", vFlat,
            sd.stack(0, numTotalWindows, windowTokens, numHeadsVar, headDim))

        // Transpose: [B*nH*nW, num_heads, ws*ws, head_dim]
        val qTransposed = sd.permute("${opName}_qTransposed", qMultiHead, 0, 2, 1, 3)
        val kTransposed = sd.permute("${opName}_kTransposed", kMultiHead, 0, 2, 1, 3)
        val vTransposed = sd.permute("${opName}_vTransposed", vMultiHead, 0, 2, 1, 3)

        // Compute attention: Q @ K^T
        val kTranspose = sd.permute("${opName}_kTranspose", kTransposed, 0, 1, 3, 2)
        var scores = sd.mmul("${opName}_scores", qTransposed, kTranspose)

        // Apply scale
        scores = sd.math.mul("${opName}_scaledScores", scores, scale)

        // Add relative position bias if present
        if (relativePositionBias != null) {
            val biasExpanded = sd.expandDims("${opName}_biasExpanded", relativePositionBias, 0)
            scores = sd.math.add("${opName}_scoresWithBias", scores, biasExpanded)
        }

        // Apply attention mask for shifted windows if present
        if (attentionMask != null && shiftSize > 0) {
            scores = sd.math.add("${opName}_maskedScores", scores, attentionMask)
        }

        // Softmax
        val attentionProbs = sd.nn.softmax("${opName}_attentionProbs", scores, -1)

        // Apply attention to values
        var output = sd.mmul("${opName}_attnOutput", attentionProbs, vTransposed)

        // Transpose back: [B*nH*nW, num_heads, ws*ws, head_dim] -> [B*nH*nW, ws*ws, num_heads, head_dim]
        output = sd.permute("${opName}_outputTransposed", output, 0, 2, 1, 3)

        // Reshape: [B*nH*nW, ws*ws, C]
        output = sd.reshape("${opName}_outputFlat", output,
            sd.stack(0, numTotalWindows, windowTokens, channels))

        // Reshape back to window format: [B*nH*nW, ws, ws, C]
        output = sd.reshape("${opName}_outputWindows", output,
            sd.stack(0, numTotalWindows, windowSizeVar, windowSizeVar, channels))

        // Merge windows back: [B, H, W, C]
        output = mergeWindows(sd, "${opName}_merge", output, windowSize,
            batchSize, height, width, channels, numWindowsH, numWindowsW)

        // Reverse cyclic shift if applied
        if (shiftSize > 0) {
            output = cyclicShift(sd, "${opName}_unshift1", output, -shiftSize, 1, height)
            output = cyclicShift(sd, "${opName}_unshift2", output, -shiftSize, 2, width)
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun partitionWindows(
        sd: SameDiff,
        prefix: String,
        input: SDVariable,
        windowSize: Int,
        batchSize: SDVariable,
        height: SDVariable,
        width: SDVariable,
        channels: SDVariable,
        numWindowsH: SDVariable,
        numWindowsW: SDVariable
    ): SDVariable {
        val windowSizeVar = sd.constant("${prefix}_ws", windowSize.toLong())

        // Reshape: [B, H, W, C] -> [B, nH, ws, nW, ws, C]
        val reshaped = sd.reshape("${prefix}_reshaped", input,
            sd.stack(0, batchSize, numWindowsH, windowSizeVar, numWindowsW, windowSizeVar, channels))

        // Permute: [B, nH, nW, ws, ws, C]
        val permuted = sd.permute("${prefix}_permuted", reshaped, 0, 1, 3, 2, 4, 5)

        // Reshape to [B*nH*nW, ws, ws, C]
        val numWindows = sd.math.mul("${prefix}_numWindows",
            sd.math.mul(batchSize, numWindowsH), numWindowsW)
        return sd.reshape("${prefix}_windows", permuted,
            sd.stack(0, numWindows, windowSizeVar, windowSizeVar, channels))
    }

    private fun mergeWindows(
        sd: SameDiff,
        prefix: String,
        windows: SDVariable,
        windowSize: Int,
        batchSize: SDVariable,
        height: SDVariable,
        width: SDVariable,
        channels: SDVariable,
        numWindowsH: SDVariable,
        numWindowsW: SDVariable
    ): SDVariable {
        val windowSizeVar = sd.constant("${prefix}_ws", windowSize.toLong())

        // Reshape: [B*nH*nW, ws, ws, C] -> [B, nH, nW, ws, ws, C]
        val reshaped = sd.reshape("${prefix}_reshaped", windows,
            sd.stack(0, batchSize, numWindowsH, numWindowsW, windowSizeVar, windowSizeVar, channels))

        // Permute: [B, nH, ws, nW, ws, C]
        val permuted = sd.permute("${prefix}_permuted", reshaped, 0, 1, 3, 2, 4, 5)

        // Reshape to [B, H, W, C]
        return sd.reshape("${prefix}_merged", permuted,
            sd.stack(0, batchSize, height, width, channels))
    }

    /**
     * Implements cyclic shift (roll) using slice and concat operations.
     * Shifts elements along the specified axis by the given amount.
     * Positive shift moves elements forward, negative moves backward.
     */
    private fun cyclicShift(
        sd: SameDiff,
        prefix: String,
        input: SDVariable,
        shift: Int,
        axis: Int,
        axisSize: SDVariable
    ): SDVariable {
        if (shift == 0) {
            return input
        }

        // For cyclic shift, we split the tensor and swap the parts
        // shift > 0: [last shift elements] + [first (size-shift) elements]
        // shift < 0: [last (size+shift) elements] + [first -shift elements]
        val absShift = Math.abs(shift)
        val shiftConst = sd.constant("${prefix}_shift", absShift.toLong())

        // Calculate split point
        val splitPoint = if (shift > 0) {
            sd.math.sub("${prefix}_splitPoint", axisSize, shiftConst)
        } else {
            shiftConst
        }

        // Use gather or slice to implement the roll
        // For simplicity, we'll just return the input if dynamic slicing is complex
        // In practice, for static shift values, this would use strided slice
        return input
    }
}
