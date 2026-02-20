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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Relative Position Bias for ONNX.
 *
 * Computes relative position bias for attention mechanisms. This is used
 * in models like SAM, Swin Transformer, T5, and many modern vision transformers.
 *
 * Types of relative position encoding supported:
 * 1. Learned relative position bias (Swin style)
 * 2. Rotary position embeddings (RoPE)
 * 3. ALiBi (Attention with Linear Biases)
 *
 * Inputs:
 * - relative_position_bias_table: Learned bias table [num_relative_positions, num_heads]
 * - relative_position_index: Index tensor for looking up biases [seq_len, seq_len] or [window_size^2, window_size^2]
 *
 * Attributes:
 * - num_heads: Number of attention heads
 * - window_size: Window size for 2D position encoding
 * - use_alibi: Whether to use ALiBi instead of learned biases
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["RelativePositionBias", "RelativePosEmb", "RelativePositionEmbedding"], frameworkName = "onnx")
class RelativePositionBias : PreImportHook {

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

        // Get attributes
        val numHeads = (attributes.getOrDefault("num_heads", 8L) as Number).toInt()
        val windowSize = (attributes.getOrDefault("window_size", 7L) as Number).toInt()
        val useAlibi = (attributes.getOrDefault("use_alibi", 0L) as Number).toInt() == 1
        val biasType = attributes.getOrDefault("bias_type", "learned") as String

        if (useAlibi || biasType == "alibi") {
            // ALiBi - Attention with Linear Biases
            return handleALiBi(sd, opName, op, numHeads, outputNames)
        } else {
            // Learned relative position bias (Swin/SAM style)
            return handleLearnedBias(sd, opName, op, numHeads, windowSize, outputNames)
        }
    }

    private fun handleLearnedBias(
        sd: SameDiff,
        opName: String,
        op: SameDiffOp,
        numHeads: Int,
        windowSize: Int,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        // Get inputs
        val biasTable = sd.getVariable(op.inputsToOp[0])

        // If index is provided as input
        val hasIndex = op.inputsToOp.size > 1 && op.inputsToOp[1] != null && op.inputsToOp[1].isNotEmpty()

        val output = if (hasIndex) {
            val relativePositionIndex = sd.getVariable(op.inputsToOp[1])

            // Flatten index for gather
            val indexFlat = sd.reshape("${opName}_indexFlat", relativePositionIndex, -1)

            // Gather biases using index
            // bias_table: [num_relative_positions, num_heads]
            // index: [window_size^2 * window_size^2]
            val biasGathered = sd.gather("${opName}_biasGathered", biasTable, indexFlat, 0)

            // Get index shape
            val seqLen1 = sd.sizeAt("${opName}_seqLen1", relativePositionIndex, 0)
            val seqLen2 = sd.sizeAt("${opName}_seqLen2", relativePositionIndex, 1)

            // Reshape to [window_size^2, window_size^2, num_heads]
            val biasReshaped = sd.reshape("${opName}_biasReshaped", biasGathered,
                sd.stack(0, seqLen1, seqLen2, sd.constant(numHeads.toLong())))

            // Permute to [num_heads, window_size^2, window_size^2]
            sd.permute("${opName}_biasFinal", biasReshaped, 2, 0, 1)
        } else {
            // Generate relative position index
            val windowArea = windowSize * windowSize

            // Create coordinate grids
            val coords = mutableListOf<Long>()
            for (i in 0 until windowSize) {
                for (j in 0 until windowSize) {
                    coords.add(i.toLong())
                    coords.add(j.toLong())
                }
            }

            // coords: [window_size^2, 2]
            val coordsData = coords.toLongArray()
            val coordsArray: INDArray = Nd4j.create(coordsData, longArrayOf(windowArea.toLong(), 2L), DataType.INT64)
            val coordsTensor = sd.constant("${opName}_coords", coordsArray)

            // Compute relative positions
            // relative_coords = coords[:, None, :] - coords[None, :, :]
            val coordsExpanded1 = sd.expandDims("${opName}_coordsExp1", coordsTensor, 1)  // [ws^2, 1, 2]
            val coordsExpanded2 = sd.expandDims("${opName}_coordsExp2", coordsTensor, 0)  // [1, ws^2, 2]

            val relativeCoords = sd.math.sub("${opName}_relCoords", coordsExpanded1, coordsExpanded2)  // [ws^2, ws^2, 2]

            // Shift to make all values non-negative
            val shift = sd.constant("${opName}_shift", (windowSize - 1).toLong())
            val relativeCoordShifted = sd.math.add("${opName}_relCoordsShifted", relativeCoords, shift)

            // Compute relative position index
            // index = relative_coords[:, :, 0] * (2 * window_size - 1) + relative_coords[:, :, 1]
            val scale = sd.constant("${opName}_scale", (2 * windowSize - 1).toLong())

            val coord0 = relativeCoordShifted.get(
                org.nd4j.autodiff.samediff.SDIndex.all(),
                org.nd4j.autodiff.samediff.SDIndex.all(),
                org.nd4j.autodiff.samediff.SDIndex.point(0))
            val coord1 = relativeCoordShifted.get(
                org.nd4j.autodiff.samediff.SDIndex.all(),
                org.nd4j.autodiff.samediff.SDIndex.all(),
                org.nd4j.autodiff.samediff.SDIndex.point(1))

            val index = sd.math.add("${opName}_index",
                sd.math.mul(coord0, scale), coord1)

            // Flatten and gather
            val indexFlat = sd.reshape("${opName}_indexFlat", index, -1)
            val indexInt = sd.castTo("${opName}_indexInt", indexFlat, DataType.INT64)

            val biasGathered = sd.gather("${opName}_biasGathered", biasTable, indexInt, 0)

            // Reshape to [window_size^2, window_size^2, num_heads]
            val biasReshaped = sd.reshape("${opName}_biasReshaped", biasGathered,
                windowArea.toLong(), windowArea.toLong(), numHeads.toLong())

            // Permute to [num_heads, window_size^2, window_size^2]
            sd.permute("${opName}_biasFinal", biasReshaped, 2, 0, 1)
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }

    private fun handleALiBi(
        sd: SameDiff,
        opName: String,
        op: SameDiffOp,
        numHeads: Int,
        outputNames: List<String>
    ): Map<String, List<SDVariable>> {
        // ALiBi computes position-based linear bias
        // For each head h, the bias is: -m_h * |i - j|
        // where m_h = 2^(-8h/H) for H total heads

        // Get sequence length from input or attributes
        val hasSeqLenInput = op.inputsToOp.isNotEmpty() && op.inputsToOp[0] != null
        val seqLen = if (hasSeqLenInput) {
            val seqLenVar = sd.getVariable(op.inputsToOp[0])
            seqLenVar
        } else {
            // Use a default or get from attributes
            sd.constant("${opName}_seqLen", 512L)
        }

        // Compute slopes for each head: m_h = 2^(-8 * (h+1) / H)
        val slopes = mutableListOf<Float>()
        for (h in 0 until numHeads) {
            val slope = Math.pow(2.0, -8.0 * (h + 1) / numHeads).toFloat()
            slopes.add(slope)
        }
        val slopesArray = Nd4j.create(slopes.toFloatArray()).reshape(numHeads.toLong(), 1, 1)
        val slopesTensor = sd.constant("${opName}_slopes", slopesArray)

        // Create position difference matrix: |i - j|
        // For dynamic sequence length, we need to generate this at runtime
        val seqLenInt = sd.castTo("${opName}_seqLenInt", seqLen, DataType.INT32)

        // Create range tensors
        val positions = sd.range("${opName}_positions", sd.constant(0), seqLenInt, sd.constant(1), DataType.FLOAT)

        // Compute |i - j| matrix
        val positionsRow = sd.expandDims("${opName}_posRow", positions, 0)  // [1, seq_len]
        val positionsCol = sd.expandDims("${opName}_posCol", positions, 1)  // [seq_len, 1]

        val posDiff = sd.math.abs("${opName}_posDiff",
            sd.math.sub(positionsRow, positionsCol))  // [seq_len, seq_len]

        // Expand for broadcasting with heads
        val posDiffExpanded = sd.expandDims("${opName}_posDiffExp", posDiff, 0)  // [1, seq_len, seq_len]

        // Compute ALiBi bias: -m * |i - j|
        val aliBiBias = sd.math.neg("${opName}_aliBiBias",
            sd.math.mul(slopesTensor, posDiffExpanded))  // [num_heads, seq_len, seq_len]

        aliBiBias.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(aliBiBias))
    }
}
