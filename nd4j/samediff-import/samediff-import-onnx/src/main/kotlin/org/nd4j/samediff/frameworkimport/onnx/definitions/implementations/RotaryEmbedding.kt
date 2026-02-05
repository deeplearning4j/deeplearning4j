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
 * Implementation of Microsoft ONNX RotaryEmbedding operation.
 *
 * RotaryEmbedding applies rotary position embedding (RoPE) to the input tensor.
 * This is commonly used in modern LLMs like LLaMA, Mistral, etc.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#RotaryEmbedding
 *
 * The operation computes (for non-interleaved, split-based formulation):
 * x1, x2 = split(input, 2, axis=-1)  // split head_dim into two halves
 * y1 = x1 * cos - x2 * sin
 * y2 = x1 * sin + x2 * cos
 * output = concat(y1, y2, axis=-1)
 *
 * Inputs:
 * - input: Input tensor [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
 * - position_ids: Position indices [batch, seq_len]
 * - cos_cache: Cached cosine values
 * - sin_cache: Cached sine values
 *
 * Attributes:
 * - interleaved: Whether head_dim is interleaved (default: 0)
 * - is_packed_batching: Whether batch is packed (default: 0)
 * - num_heads: Number of attention heads (default: 0 - infer from input)
 * - rotary_embedding_dim: Dimension for rotary embedding (default: 0 - use head_dim)
 * - scale: Scaling factor (default: 1.0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["RotaryEmbedding"], frameworkName = "onnx")
class RotaryEmbedding : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input = sd.getVariable(op.inputsToOp[0])
        val positionIds = sd.getVariable(op.inputsToOp[1])
        val cosCache = sd.getVariable(op.inputsToOp[2])
        val sinCache = sd.getVariable(op.inputsToOp[3])

        // Get attributes
        val interleaved = (attributes.getOrDefault("interleaved", 0) as Number).toInt() != 0
        val scale = (attributes.getOrDefault("scale", 1.0) as Number).toDouble()
        val rotaryDim = (attributes.getOrDefault("rotary_embedding_dim", 0) as Number).toInt()
        var numHeads = (attributes.getOrDefault("num_heads", 0) as Number).toInt()

        // Gather cos and sin values based on position_ids
        // cos_cache/sin_cache shape is [max_seq, head_dim/2] per ONNX RotaryEmbedding spec
        // The rotation operates on pairs of elements, so cos/sin have half the head dimension.
        var cos = sd.gather(cosCache, positionIds, 0)
        var sin = sd.gather(sinCache, positionIds, 0)
        // After gather: [batch, seq, half_head_dim] e.g. [1, 283, 32]

        // Get half_head_dim from cos_cache shape.
        // Prefer static extraction at import time when cosCache has a known shape,
        // falling back to dynamic computation only if shape is unknown.
        val cosArr = sd.getArrForVarName(cosCache.name())
        val halfHeadDimVar: SDVariable
        val actualHeadDimVar: SDVariable
        println("RotaryEmbedding: cosCache='${cosCache.name()}', type=${cosCache.variableType}, cosArr=${cosArr != null}, numHeads=$numHeads")
        if (cosArr != null) {
            // cosCache is a constant with known shape — extract last dimension statically
            val halfHeadDim = cosArr.shape()[cosArr.rank() - 1]
            println("RotaryEmbedding: STATIC path - halfHeadDim=$halfHeadDim, actualHeadDim=${halfHeadDim * 2}")
            halfHeadDimVar = sd.constant(Nd4j.scalar(halfHeadDim))
            actualHeadDimVar = sd.constant(Nd4j.scalar(halfHeadDim * 2))
        } else {
            println("RotaryEmbedding: DYNAMIC path for halfHeadDim (cosCache not a constant)")
            // cosCache shape unknown at import — compute dynamically
            val cosShapeVar = sd.shape(cosCache)
            val cosRankVar = sd.rank(cosCache)
            halfHeadDimVar = sd.gatherNd(cosShapeVar, sd.expandDims(sd.math.sub(cosRankVar, sd.constant(1)), 0))
            val twoConst = sd.constant(Nd4j.scalar(2L))
            actualHeadDimVar = sd.math.mul(halfHeadDimVar, twoConst)
        }

        // Get input shape dynamically (batch and seq are always dynamic)
        val inputShapeVar = sd.shape(input)

        // Save original shape for reshaping back later
        val originalInputShapeVar = inputShapeVar

        val idx0 = sd.constant(Nd4j.scalar(0L))
        val idx1 = sd.constant(Nd4j.scalar(1L))

        val batchSize = sd.gather(inputShapeVar, idx0, 0)
        val seqLen = sd.gather(inputShapeVar, idx1, 0)

        // Reshape input to 4D: [batch, seq, num_heads, actual_head_dim]
        // Use -1 for num_heads so reshape infers it automatically.
        // This handles both 3D [batch, seq, hidden] and 4D [batch, seq, num_heads, head_dim]
        // inputs without fragile conditional arithmetic.
        val negOne = sd.constant(Nd4j.scalar(-1L))
        val targetShape = sd.stack(0, batchSize, seqLen, negOne, actualHeadDimVar)
        println("RotaryEmbedding: using reshape with -1 for num_heads inference, actualHeadDim from static=$actualHeadDimVar")

        // Reshape input to 4D with correct head structure
        val workingInput = sd.reshape(input, targetShape)

        // Expand cos/sin for broadcasting: [batch, seq, half_dim] -> [batch, seq, 1, half_dim]
        cos = sd.expandDims(cos, 2)
        sin = sd.expandDims(sin, 2)

        var result: SDVariable
        if (interleaved) {
            // Interleaved mode: pairs (x0,x1), (x2,x3), ...
            // Reshape to [..., half_dim, 2], extract even/odd, rotate, interleave back
            val xShape = sd.shape(workingInput)
            val allButLast = sd.stridedSlice(xShape,
                sd.constant(Nd4j.createFromArray(0L)),
                sd.constant(Nd4j.createFromArray(-1L)),
                sd.constant(Nd4j.createFromArray(1L)))
            val reshapeTarget = sd.concat(0, allButLast,
                sd.expandDims(halfHeadDimVar, 0),
                sd.constant(Nd4j.createFromArray(2L)))
            val reshaped = sd.reshape(workingInput, reshapeTarget)
            val splitParts = sd.split(reshaped, 2, -1)
            val even = sd.squeeze(splitParts[0], -1)  // x0, x2, x4, ...
            val odd = sd.squeeze(splitParts[1], -1)   // x1, x3, x5, ...

            val y0 = sd.math.sub(sd.math.mul(even, cos), sd.math.mul(odd, sin))
            val y1 = sd.math.add(sd.math.mul(even, sin), sd.math.mul(odd, cos))

            // Interleave back: stack on last dim then reshape
            val stacked = sd.stack(-1, y0, y1)  // [..., half_dim, 2]
            result = sd.reshape(stacked, xShape)
        } else {
            // Non-interleaved mode: split into first/second half
            // x1 = input[..., :half_dim], x2 = input[..., half_dim:]
            val splitParts = sd.split(workingInput, 2, -1)
            val x1 = splitParts[0]  // [batch, seq, num_heads, half_head_dim]
            val x2 = splitParts[1]  // [batch, seq, num_heads, half_head_dim]

            // Apply rotation per the ONNX RotaryEmbedding spec:
            //   y1 = x1 * cos - x2 * sin
            //   y2 = x1 * sin + x2 * cos
            val y1 = sd.math.sub(sd.math.mul(x1, cos), sd.math.mul(x2, sin))
            val y2 = sd.math.add(sd.math.mul(x1, sin), sd.math.mul(x2, cos))

            // Concatenate halves back: [batch, seq, num_heads, head_dim]
            result = sd.concat(-1, y1, y2)
        }

        // Apply scaling
        if (scale != 1.0) {
            result = sd.math.mul(result, scale)
        }

        // Reshape back to original shape (3D or 4D)
        result = sd.reshape(result, originalInputShapeVar)

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }

    /**
     * Rotate half for non-interleaved format.
     * Split into first and second half, negate second half, and swap.
     * This version uses dynamic shape operations for compatibility with symbolic shapes.
     */
    private fun rotateHalf(sd: SameDiff, x: SDVariable): SDVariable {
        // For dynamic shapes, we use a different approach:
        // Split the tensor along the last axis into two equal parts
        // Then concatenate [-second_half, first_half]

        // Use split with num_split=2 to split into two halves along the last axis
        val splitOutputs = sd.split(x, 2, -1)
        val x1 = splitOutputs[0]  // First half
        val x2 = splitOutputs[1]  // Second half

        // Return [-x2, x1]
        val negX2 = sd.math.neg(x2)
        return sd.concat(-1, negX2, x1)
    }

    /**
     * Rotate half for interleaved format.
     * Pairs (x0, x1) become (-x1, x0).
     * This version handles dynamic shapes by using reshape with -1 dimension.
     */
    private fun rotateHalfInterleaved(sd: SameDiff, x: SDVariable): SDVariable {
        // Get the shape dynamically
        val xShape = sd.shape(x)
        val rank = sd.size(xShape)

        // Get all dimensions except the last one, and the last dimension
        // Reshape to [..., lastDim/2, 2]
        // Use gather to get lastDim, then divide by 2
        val lastDimIdx = sd.math.sub(rank, sd.constant(1))
        val lastDim = sd.gatherNd(xShape, sd.expandDims(lastDimIdx, 0))
        val halfDim = sd.math.div(lastDim, sd.constant(2L))

        // Build new shape: [..., halfDim, 2]
        // Take all dims except last, append halfDim and 2
        // Use SDVariable-based stridedSlice for dynamic slicing
        val zero = sd.constant(Nd4j.createFromArray(0L))
        val negOne = sd.constant(Nd4j.createFromArray(-1L))
        val one = sd.constant(Nd4j.createFromArray(1L))
        val allButLast = sd.stridedSlice(xShape, zero, negOne, one)
        val two = sd.constant(Nd4j.createFromArray(2L))
        val halfDimExpanded = sd.expandDims(halfDim, 0)
        val newShape = sd.concat(0, allButLast, halfDimExpanded, two)

        val reshaped = sd.reshape(x, newShape)

        // Split along last axis into even and odd (2 equal parts)
        val splitParts = sd.split(reshaped, 2, -1)
        val even = splitParts[0]
        val odd = splitParts[1]

        // Stack [-odd, even]
        val negOdd = sd.math.neg(odd)
        val stacked = sd.concat(-1, negOdd, even)

        // Reshape back to original shape
        return sd.reshape(stacked, xShape)
    }
}
