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

import org.nd4j.autodiff.samediff.SDIndex
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

@PreHookRule(nodeNames = [], opNames = ["Attention"], frameworkName = "onnx")
class Attention: PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        // Detect which version of Attention we're dealing with
        // Microsoft version has "num_heads" attribute and uses input/weights/bias
        // ONNX versions use "q_num_heads"/"kv_num_heads" and Q/K/V inputs
        
        return if (attributes.containsKey("num_heads")) {
            // Microsoft Attention version
            handleMicrosoftAttention(sd, attributes, outputNames, op)
        } else {
            // ONNX Attention version (23 or 24)
            handleOnnxAttention(sd, attributes, outputNames, op)
        }
    }
    
    private fun handleMicrosoftAttention(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp
    ): Map<String, List<SDVariable>> {
        val opName = op.name
        
        val input = sd.getVariable(op.inputsToOp[0])
        val weights = sd.getVariable(op.inputsToOp[1])
        var bias: SDVariable? = null
        if (op.inputsToOp.size > 2 && op.inputsToOp[2] != null && op.inputsToOp[2].isNotEmpty()) {
            bias = sd.getVariable(op.inputsToOp[2])
        }
        var maskIndex: SDVariable? = null
        if (op.inputsToOp.size > 3 && op.inputsToOp[3] != null && op.inputsToOp[3].isNotEmpty()) {
            maskIndex = sd.getVariable(op.inputsToOp[3])
        }
        var past: SDVariable? = null
        if (op.inputsToOp.size > 4 && op.inputsToOp[4] != null && op.inputsToOp[4].isNotEmpty()) {
            past = sd.getVariable(op.inputsToOp[4])
        }
        var relativePositionBias: SDVariable? = null
        if (op.inputsToOp.size > 5 && op.inputsToOp[5] != null && op.inputsToOp[5].isNotEmpty()) {
            relativePositionBias = sd.getVariable(op.inputsToOp[5])
        }
        var pastSequenceLength: SDVariable? = null
        if (op.inputsToOp.size > 6 && op.inputsToOp[6] != null && op.inputsToOp[6].isNotEmpty()) {
            pastSequenceLength = sd.getVariable(op.inputsToOp[6])
        }

        // Get Microsoft-specific attributes
        val numHeads = (attributes["num_heads"] as Number).toInt()
        val unidirectional = (attributes.getOrDefault("unidirectional", 0L) as Number).toInt() == 1
        val doRotary = (attributes.getOrDefault("do_rotary", 0L) as Number).toInt() == 1
        val maskFilterValue = (attributes.getOrDefault("mask_filter_value", -10000.0f) as Number).toFloat()
        val scale = attributes.getOrDefault("scale", null) as Number?
        val qkvHiddenSizes = attributes.getOrDefault("qkv_hidden_sizes", null) as List<Int>?
        val pastPresentShareBuffer = (attributes.getOrDefault("past_present_share_buffer", 0L) as Number).toInt() == 1

        // Get dimensions from input shape
        val inputShape = sd.shape(input).rename("${opName}_inputShape")
        val actualBatchSize = inputShape.get(SDIndex.point(0)).rename("${opName}_actualBatchSize")
        val actualSeqLen = inputShape.get(SDIndex.point(1)).rename("${opName}_actualSeqLen")
        val inputHiddenSize = inputShape.get(SDIndex.point(2)).rename("${opName}_inputHiddenSize")
        
        // Project input to Q, K, V using tensorMmul
        val qkv = sd.tensorMmul("${opName}_qkv", input, weights, intArrayOf(2), 0)
        
        // Add bias if present
        val qkvWithBias = if (bias != null) {
            sd.math.add("${opName}_qkvWithBias", qkv, bias)
        } else qkv
        
        // Get total QKV shape
        val qkvShape = sd.shape(qkvWithBias).rename("${opName}_qkvShape")
        val totalHiddenSize = qkvShape.get(SDIndex.point(2)).rename("${opName}_totalHiddenSize")
        
        // Calculate Q, K, V hidden sizes dynamically
        val qHiddenSize: SDVariable
        val kHiddenSize: SDVariable
        val vHiddenSize: SDVariable
        
        if (qkvHiddenSizes != null && qkvHiddenSizes.isNotEmpty()) {
            // Use provided sizes
            qHiddenSize = sd.constant("${opName}_qHiddenSize", qkvHiddenSizes[0].toLong())
            kHiddenSize = if (qkvHiddenSizes.size > 1) {
                sd.constant("${opName}_kHiddenSize", qkvHiddenSizes[1].toLong()) 
            } else qHiddenSize
            vHiddenSize = if (qkvHiddenSizes.size > 2) {
                sd.constant("${opName}_vHiddenSize", qkvHiddenSizes[2].toLong())
            } else kHiddenSize
        } else {
            // Assume equal split among Q, K, V
            val three = sd.constant("${opName}_three", 3L)
            val hiddenSizePerPart = sd.math.floorDiv("${opName}_hiddenSizePerPart", totalHiddenSize, three)
            qHiddenSize = hiddenSizePerPart
            kHiddenSize = hiddenSizePerPart
            vHiddenSize = hiddenSizePerPart
        }
        
        // Calculate head size
        val numHeadsLong = sd.constant("${opName}_numHeadsLong", numHeads.toLong())
        val headSize = sd.math.floorDiv("${opName}_headSize", qHiddenSize, numHeadsLong)

        // Extract Q, K, V using dynamic slicing with proper intervals
        val zero = sd.constant("${opName}_zero", 0L)
        
        // Q: from 0 to qHiddenSize
        val q = qkvWithBias.get(SDIndex.all(), SDIndex.all(), 
            SDIndex.interval(zero, qHiddenSize)).rename("${opName}_q")
        
        // K: from qHiddenSize to qHiddenSize + kHiddenSize
        val kStart = qHiddenSize
        val kEnd = sd.math.add("${opName}_kEnd", kStart, kHiddenSize)
        val k = qkvWithBias.get(SDIndex.all(), SDIndex.all(), 
            SDIndex.interval(kStart, kEnd)).rename("${opName}_k")
        
        // V: from qHiddenSize + kHiddenSize to end
        val vStart = kEnd
        val v = qkvWithBias.get(SDIndex.all(), SDIndex.all(), 
            SDIndex.interval(vStart, totalHiddenSize)).rename("${opName}_v")
        
        // Reshape Q, K, V to (batch, seq, num_heads, head_size)
        val qReshapeShape = sd.concat("${opName}_qReshapeShape", 0, actualBatchSize, actualSeqLen, numHeadsLong, headSize)
        val qReshaped = sd.reshape("${opName}_qReshaped", q, qReshapeShape)
        
        val kReshapeShape = sd.concat("${opName}_kReshapeShape", 0, actualBatchSize, actualSeqLen, numHeadsLong, headSize)
        val kReshaped = sd.reshape("${opName}_kReshaped", k, kReshapeShape)
        
        // For V, calculate its actual head size in case of dimension mismatch
        val vShape = sd.shape(v).rename("${opName}_vShape")
        val actualVHiddenSize = vShape.get(SDIndex.point(2)).rename("${opName}_actualVHiddenSize")
        val vHeadSize = sd.math.floorDiv("${opName}_vHeadSize", actualVHiddenSize, numHeadsLong)
        
        val vReshapeShape = sd.concat("${opName}_vReshapeShape", 0, actualBatchSize, actualSeqLen, numHeadsLong, vHeadSize)
        val vReshaped = sd.reshape("${opName}_vReshaped", v, vReshapeShape)

        // Transpose to (batch, num_heads, seq, head_size)
        val qTransposed = sd.permute("${opName}_qTransposed", qReshaped, 0, 2, 1, 3)
        val kTransposed = sd.permute("${opName}_kTransposed", kReshaped, 0, 2, 1, 3)
        val vTransposed = sd.permute("${opName}_vTransposed", vReshaped, 0, 2, 1, 3)

        // Get actual head size from transposed shape for scale calculation
        val qTransposedShape = sd.shape(qTransposed).rename("${opName}_qTransposedShape")
        val actualHeadSize = qTransposedShape.get(SDIndex.point(3)).rename("${opName}_actualHeadSize")

        // Handle past state if present
        val kToUse: SDVariable
        val vToUse: SDVariable
        if (past != null) {
            // past has shape (2, batch, num_heads, past_seq_len, head_size)
            // Extract past K and V
            val zeroIdx = sd.expandDims("${opName}_zeroIdx", sd.constant("${opName}_zeroConst", 0), 0)
            val oneIdx = sd.expandDims("${opName}_oneIdx", sd.constant("${opName}_oneConst", 1), 0)
            val pastK = sd.gather("${opName}_pastK", past, zeroIdx, 0)
            val pastV = sd.gather("${opName}_pastV", past, oneIdx, 0)
            
            // Concatenate with current K, V along sequence dimension
            kToUse = sd.concat("${opName}_kToUse", 2, pastK, kTransposed)
            vToUse = sd.concat("${opName}_vToUse", 2, pastV, vTransposed)
        } else {
            kToUse = kTransposed
            vToUse = vTransposed
        }

        // Apply rotary embeddings if needed
        val qFinal = if (doRotary) {
            // TODO: Implement rotary embeddings
            qTransposed
        } else qTransposed
        
        val kFinal = if (doRotary) {
            // TODO: Implement rotary embeddings
            kToUse
        } else kToUse

        // Compute attention scores using matmul with 3D reshaping
        // Reshape from [batch, heads, seq, head_size] to [batch*heads, seq, head_size]
        val batchTimesHeads = sd.math.mul("${opName}_batchTimesHeads", actualBatchSize, numHeadsLong)
        val qSeqLen = sd.shape(qFinal).get(SDIndex.point(2)).rename("${opName}_qSeqLen")
        val kSeqLen = sd.shape(kFinal).get(SDIndex.point(2)).rename("${opName}_kSeqLen")

        val qReshape3D = sd.reshape("${opName}_qReshape3D", qFinal,
            sd.concat("${opName}_qReshape3DShape", 0, batchTimesHeads, qSeqLen, actualHeadSize))
        val kReshape3D = sd.reshape("${opName}_kReshape3D", kFinal,
            sd.concat("${opName}_kReshape3DShape", 0, batchTimesHeads, kSeqLen, actualHeadSize))

        val scores3D = sd.linalg().mmul("${opName}_scores3D", qReshape3D, kReshape3D, false, true, false)
        val scores = sd.reshape("${opName}_scores", scores3D,
            sd.concat("${opName}_scoresShape", 0, actualBatchSize, numHeadsLong, qSeqLen, kSeqLen))
        
        // Apply scale
        val scaleValue = if (scale != null) {
            sd.constant("${opName}_scaleValue", scale.toFloat())
        } else {
            // Default scale is 1/sqrt(head_size)
            val sqrtHeadSize = sd.math.sqrt("${opName}_sqrtHeadSize", sd.castTo("${opName}_headSizeCast", actualHeadSize, DataType.FLOAT))
            sd.math().div("${opName}_scaleValue", sd.constant("${opName}_oneFloat", 1.0f), sqrtHeadSize)
        }
        var scaledScores = sd.math.mul("${opName}_scaledScores", scores, scaleValue)
        
        // Add relative position bias if present
        if (relativePositionBias != null) {
            scaledScores = sd.math.add("${opName}_scaledScoresWithBias", scaledScores, relativePositionBias)
        }
        
        // Apply mask if present
        if (maskIndex != null || unidirectional) {
            val maskValue = sd.constant("${opName}_maskValue", maskFilterValue)
            
            if (maskIndex != null) {
                // Handle different mask formats
                scaledScores = sd.math.add("${opName}_scaledScoresWithMask", scaledScores, maskIndex)
            }
            
            if (unidirectional) {
                // Create causal mask
                val qSeqLen = sd.shape(qFinal).get(SDIndex.point(2)).rename("${opName}_qSeqLen")
                val kSeqLen = sd.shape(kFinal).get(SDIndex.point(2)).rename("${opName}_kSeqLen")
                
                // Create the shape by concatenating dimensions
                val onesShape = sd.concat("${opName}_onesShape", 0, qSeqLen, kSeqLen)
                val ones = sd.create("${opName}_ones", onesShape, DataType.FLOAT)
                ones.assign(1)
                val causalMask = sd.linalg().triu("${opName}_causalMask", ones, 1)
                val causalMaskInverted = sd.math.sub("${opName}_causalMaskInverted", sd.constant("${opName}_oneFloatMask", 1.0f), causalMask)
                val causalMaskExpanded1 = sd.expandDims("${opName}_causalMaskExpanded1", causalMaskInverted, 0)
                val causalMaskExpanded = sd.expandDims("${opName}_causalMaskExpanded", causalMaskExpanded1, 0)
                
                // Apply mask
                val maskSub = sd.math.sub("${opName}_maskSub", sd.constant("${opName}_oneFloatMask2", 1.0f), causalMaskExpanded)
                val maskAdd = sd.math.mul("${opName}_maskAdd", maskSub, maskValue)
                scaledScores = sd.math.add("${opName}_scaledScoresUnidirectional", scaledScores, maskAdd)
            }
        }

        // Apply softmax along last dimension
        val attentionWeights = sd.nn().softmax("${opName}_attentionWeights", scaledScores, 3)

        // Apply attention to values using matmul with 3D reshaping
        val vSeqLen = sd.shape(vToUse).get(SDIndex.point(2)).rename("${opName}_vSeqLen")

        val attentionWeights3D = sd.reshape("${opName}_attentionWeights3D", attentionWeights,
            sd.concat("${opName}_attentionWeights3DShape", 0, batchTimesHeads, qSeqLen, vSeqLen))
        val v3D = sd.reshape("${opName}_v3D", vToUse,
            sd.concat("${opName}_v3DShape", 0, batchTimesHeads, vSeqLen, vHeadSize))

        val attentionOutput3D = sd.linalg().mmul("${opName}_attentionOutput3D", attentionWeights3D, v3D, false, false, false)
        val attentionOutput = sd.reshape("${opName}_attentionOutput", attentionOutput3D,
            sd.concat("${opName}_attentionOutputShape", 0, actualBatchSize, numHeadsLong, qSeqLen, vHeadSize))
        
        // Transpose back to (batch, seq, num_heads, v_head_size)
        val outputTransposed = sd.permute("${opName}_outputTransposed", attentionOutput, 0, 2, 1, 3)
        
        // Reshape to (batch, seq, v_hidden_size) using actual V hidden size
        val outputShape = sd.concat("${opName}_outputShape", 0, actualBatchSize, actualSeqLen, actualVHiddenSize)
        val output = sd.reshape("${opName}_output", outputTransposed, outputShape)
        output.rename(outputNames[0])
        
        // Handle present output if needed
        val outputs = mutableMapOf(outputNames[0] to listOf(output))
        
        if (outputNames.size > 1 && outputNames[1] != null && outputNames[1].isNotEmpty()) {
            // Create present state (2, batch, num_heads, total_seq, head_size)
            val kFinalReshaped = sd.expandDims("${opName}_kFinalReshaped", kFinal, 0)
            val vFinalReshaped = sd.expandDims("${opName}_vFinalReshaped", vToUse, 0)
            val present = sd.concat("${opName}_present", 0, kFinalReshaped, vFinalReshaped)
            present.rename(outputNames[1])
            outputs[outputNames[1]] = listOf(present)
        }
        
        return outputs
    }
    
    private fun handleOnnxAttention(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp
    ): Map<String, List<SDVariable>> {
        val opName = op.name
        
        // Get inputs - Q, K, V are mandatory
        val q = sd.getVariable(op.inputsToOp[0])
        val k = sd.getVariable(op.inputsToOp[1])
        val v = sd.getVariable(op.inputsToOp[2])
        
        // Optional inputs
        var attnMask: SDVariable? = null
        if (op.inputsToOp.size > 3 && op.inputsToOp[3] != null && op.inputsToOp[3].isNotEmpty()) {
            attnMask = sd.getVariable(op.inputsToOp[3])
        }
        var pastKey: SDVariable? = null
        if (op.inputsToOp.size > 4 && op.inputsToOp[4] != null && op.inputsToOp[4].isNotEmpty()) {
            pastKey = sd.getVariable(op.inputsToOp[4])
        }
        var pastValue: SDVariable? = null
        if (op.inputsToOp.size > 5 && op.inputsToOp[5] != null && op.inputsToOp[5].isNotEmpty()) {
            pastValue = sd.getVariable(op.inputsToOp[5])
        }
        
        // For version 24, there's also nonpad_kv_seqlen
        var nonpadKvSeqlen: SDVariable? = null
        if (op.inputsToOp.size > 6 && op.inputsToOp[6] != null && op.inputsToOp[6].isNotEmpty()) {
            nonpadKvSeqlen = sd.getVariable(op.inputsToOp[6])
        }

        // Get attributes with proper defaults
        val isCausal = (attributes.getOrDefault("is_causal", 0L) as Number).toInt() == 1
        val kvNumHeads = attributes.get("kv_num_heads") as Int?
        val qNumHeads = attributes.get("q_num_heads") as Int?
        val qkMatmulOutputMode = (attributes.getOrDefault("qk_matmul_output_mode", 0L) as Number).toInt()
        val scale = attributes.get("scale") as Number?
        val softcap = (attributes.getOrDefault("softcap", 0.0) as Number).toFloat()
        val softmaxPrecision = attributes.get("softmax_precision") as Int?

        // Check if inputs are 3D or 4D
        val qShape = sd.shape(q).rename("${opName}_qShape")
        val qRank = sd.rank(q).rename("${opName}_qRank")
        
        // Get batch size
        val batchSize = qShape.get(SDIndex.point(0)).rename("${opName}_batchSize")
        
        // Handle both 3D and 4D cases
        val qReshaped: SDVariable
        val kReshaped: SDVariable
        val vReshaped: SDVariable
        
        if (qNumHeads != null && kvNumHeads != null) {
            // 3D case - need to reshape to 4D
            val qSeqLen = qShape.get(SDIndex.point(1)).rename("${opName}_qSeqLen")
            val qHiddenSizeDynamic = qShape.get(SDIndex.point(2)).rename("${opName}_qHiddenSize")
            
            // Calculate head sizes dynamically
            val qNumHeadsVar = sd.constant("${opName}_qNumHeadsConst", qNumHeads.toLong())
            val kvNumHeadsVar = sd.constant("${opName}_kvNumHeadsConst", kvNumHeads.toLong())
            
            val qHeadSize = sd.math.floorDiv("${opName}_qHeadSize", qHiddenSizeDynamic, qNumHeadsVar)
            
            // Get K and V shapes
            val kShape = sd.shape(k).rename("${opName}_kShape")
            val kSeqLen = kShape.get(SDIndex.point(1)).rename("${opName}_kSeqLen")
            val kHiddenSize = kShape.get(SDIndex.point(2)).rename("${opName}_kHiddenSize")
            val kHeadSize = sd.math.floorDiv("${opName}_kHeadSize", kHiddenSize, kvNumHeadsVar)
            
            val vShape = sd.shape(v).rename("${opName}_vShape")
            val vSeqLen = vShape.get(SDIndex.point(1)).rename("${opName}_vSeqLen")
            val vHiddenSize = vShape.get(SDIndex.point(2)).rename("${opName}_vHiddenSize")
            val vHeadSize = sd.math.floorDiv("${opName}_vHeadSize", vHiddenSize, kvNumHeadsVar)
            
            // Reshape Q from (batch, seq, hidden) to (batch, seq, num_heads, head_size)
            val qReshapeShape = sd.concat("${opName}_qReshapeShape", 0, batchSize, qSeqLen, qNumHeadsVar, qHeadSize)
            val qTemp = sd.reshape("${opName}_qTemp", q, qReshapeShape)
            // Transpose to (batch, num_heads, seq, head_size)
            qReshaped = sd.permute("${opName}_qReshaped", qTemp, 0, 2, 1, 3)
            
            // Similar for K and V
            val kReshapeShape = sd.concat("${opName}_kReshapeShape", 0, batchSize, kSeqLen, kvNumHeadsVar, kHeadSize)
            val kTemp = sd.reshape("${opName}_kTemp", k, kReshapeShape)
            kReshaped = sd.permute("${opName}_kReshaped", kTemp, 0, 2, 1, 3)
            
            val vReshapeShape = sd.concat("${opName}_vReshapeShape", 0, batchSize, vSeqLen, kvNumHeadsVar, vHeadSize)
            val vTemp = sd.reshape("${opName}_vTemp", v, vReshapeShape)
            vReshaped = sd.permute("${opName}_vReshaped", vTemp, 0, 2, 1, 3)
        } else {
            // 4D case - already in correct format
            qReshaped = q
            kReshaped = k
            vReshaped = v
        }
        
        // Handle past key/value concatenation if provided
        val kToUse = if (pastKey != null) {
            sd.concat("${opName}_kToUse", 2, pastKey, kReshaped)  // Concat along sequence dimension
        } else {
            kReshaped
        }
        
        val vToUse = if (pastValue != null) {
            sd.concat("${opName}_vToUse", 2, pastValue, vReshaped)  // Concat along sequence dimension
        } else {
            vReshaped
        }

        // Get dimensions for computation
        val qShapeReshaped = sd.shape(qReshaped).rename("${opName}_qShapeReshaped")
        val qNumHeadsVar = qShapeReshaped.get(SDIndex.point(1)).rename("${opName}_qNumHeadsVar")
        val qSeqLenVar = qShapeReshaped.get(SDIndex.point(2)).rename("${opName}_qSeqLenVar")
        val headSizeVar = qShapeReshaped.get(SDIndex.point(3)).rename("${opName}_headSizeVar")
        
        // Apply scale to Q and K
        val scaleToUse = if (scale != null) {
            sd.math.sqrt("${opName}_scaleToUse", sd.constant("${opName}_scaleConst", scale.toFloat()))
        } else {
            // Default scale 1/sqrt(head_size)
            val sqrtHeadSize = sd.math.sqrt("${opName}_sqrtHeadSize", sd.castTo("${opName}_headSizeCast", headSizeVar, DataType.FLOAT))
            sd.math().div("${opName}_scaleToUse", sd.constant("${opName}_oneFloat", 1.0f), sqrtHeadSize)
        }
        
        val qScaled = sd.math.mul("${opName}_qScaled", qReshaped, scaleToUse)
        val kScaled = sd.math.mul("${opName}_kScaled", kToUse, scaleToUse)
        
        // Compute Q @ K^T using matmul with 3D reshaping
        val batchTimesQHeads = sd.math.mul("${opName}_batchTimesQHeads", batchSize, qNumHeadsVar)
        val kSeqLenForScores = sd.shape(kScaled).get(SDIndex.point(2)).rename("${opName}_kSeqLenForScores")

        val qScaled3D = sd.reshape("${opName}_qScaled3D", qScaled,
            sd.concat("${opName}_qScaled3DShape", 0, batchTimesQHeads, qSeqLenVar, headSizeVar))
        val kScaled3D = sd.reshape("${opName}_kScaled3D", kScaled,
            sd.concat("${opName}_kScaled3DShape", 0, batchTimesQHeads, kSeqLenForScores, headSizeVar))

        val scores3D = sd.linalg().mmul("${opName}_scores3D", qScaled3D, kScaled3D, false, true, false)
        val scores = sd.reshape("${opName}_scores", scores3D,
            sd.concat("${opName}_scoresShape", 0, batchSize, qNumHeadsVar, qSeqLenVar, kSeqLenForScores))
        
        // Add attention mask if provided
        var scoresWithMask = scores
        if (attnMask != null) {
            scoresWithMask = sd.math.add("${opName}_scoresWithMask", scores, attnMask)
        }
        
        // Apply causal mask if needed
        if (isCausal) {
            val kSeqLen = sd.shape(kToUse).get(SDIndex.point(2)).rename("${opName}_kSeqLen")
            
            // Create causal mask
            val onesShape = sd.concat("${opName}_onesShape", 0, qSeqLenVar, kSeqLen)
            val ones = sd.create("${opName}_ones", onesShape, DataType.FLOAT)
            ones.assign(1)
            val causalMask = sd.linalg().triu("${opName}_causalMask", ones, 1)
            val causalMaskInverted = sd.math.sub("${opName}_causalMaskInverted", sd.constant("${opName}_oneFloatMask", 1.0f), causalMask)
            val causalMaskExpanded1 = sd.expandDims("${opName}_causalMaskExpanded1", causalMaskInverted, 0)
            val causalMaskExpanded = sd.expandDims("${opName}_causalMaskExpanded", causalMaskExpanded1, 0)
            
            // Apply mask by adding negative infinity where mask is 0
            val negInf = sd.constant("${opName}_negInf", -1e9f)
            val maskSub = sd.math.sub("${opName}_maskSub", sd.constant("${opName}_oneFloatMask2", 1.0f), causalMaskExpanded)
            val maskValue = sd.math.mul("${opName}_maskValue", maskSub, negInf)
            scoresWithMask = sd.math.add("${opName}_scoresWithCausalMask", scoresWithMask, maskValue)
        }
        
        // Apply softcap if needed
        var scoresAfterSoftcap = scoresWithMask
        if (softcap > 0) {
            // softcap(x) = softcap * tanh(x / softcap)
            val softcapConst = sd.constant("${opName}_softcapConst", softcap)
            val divBySoftcap = sd.math.div("${opName}_divBySoftcap", scoresWithMask, softcapConst)
            val tanhResult = sd.math.tanh("${opName}_tanhResult", divBySoftcap)
            scoresAfterSoftcap = sd.math.mul("${opName}_scoresAfterSoftcap", tanhResult, softcapConst)
        }
        
        // Apply softmax
        val attentionWeights = sd.nn().softmax("${opName}_attentionWeights", scoresAfterSoftcap, 3)
        
        // Compute attention @ V using matmul with 3D reshaping
        val vSeqLenFinal = sd.shape(vToUse).get(SDIndex.point(2)).rename("${opName}_vSeqLenFinal")
        val vHeadSizeFinal = sd.shape(vToUse).get(SDIndex.point(3)).rename("${opName}_vHeadSizeFinal")

        val attentionWeights3D = sd.reshape("${opName}_attentionWeights3D", attentionWeights,
            sd.concat("${opName}_attentionWeights3DShape", 0, batchTimesQHeads, qSeqLenVar, vSeqLenFinal))
        val v3D = sd.reshape("${opName}_v3D", vToUse,
            sd.concat("${opName}_v3DShape", 0, batchTimesQHeads, vSeqLenFinal, vHeadSizeFinal))

        val output3D = sd.linalg().mmul("${opName}_output3D", attentionWeights3D, v3D, false, false, false)
        val output = sd.reshape("${opName}_output", output3D,
            sd.concat("${opName}_outputShape", 0, batchSize, qNumHeadsVar, qSeqLenVar, vHeadSizeFinal))
        
        // Reshape output back to original format if needed
        val finalOutput = if (qNumHeads != null) {
            // Was 3D input, need to reshape back
            val outputTransposed = sd.permute("${opName}_outputTransposed", output, 0, 2, 1, 3)
            // Get the actual output hidden size from V
            val vShapeOriginal = sd.shape(v).rename("${opName}_vShapeForOutput")
            val vHiddenSizeForOutput = vShapeOriginal.get(SDIndex.point(2)).rename("${opName}_vHiddenSizeForOutput")
            val qSeqLenOriginal = qShape.get(SDIndex.point(1)).rename("${opName}_qSeqLenOriginal")
            val outputShape = sd.concat("${opName}_outputShape", 0, batchSize, qSeqLenOriginal, vHiddenSizeForOutput)
            sd.reshape("${opName}_finalOutput", outputTransposed, outputShape)
        } else {
            output
        }
        
        finalOutput.rename(outputNames[0])
        
        // Handle optional outputs
        val outputs = mutableMapOf(outputNames[0] to listOf(finalOutput))
        
        // present_key output
        if (outputNames.size > 1 && outputNames[1] != null && outputNames[1].isNotEmpty()) {
            kToUse.rename(outputNames[1])
            outputs[outputNames[1]] = listOf(kToUse)
        }
        
        // present_value output
        if (outputNames.size > 2 && outputNames[2] != null && outputNames[2].isNotEmpty()) {
            vToUse.rename(outputNames[2])
            outputs[outputNames[2]] = listOf(vToUse)
        }
        
        // qk_matmul_output
        if (outputNames.size > 3 && outputNames[3] != null && outputNames[3].isNotEmpty()) {
            val qkOutput = when (qkMatmulOutputMode) {
                0 -> scores
                1 -> scoresWithMask
                2 -> scoresAfterSoftcap
                3 -> attentionWeights
                else -> scores
            }
            qkOutput.rename(outputNames[3])
            outputs[outputNames[3]] = listOf(qkOutput)
        }
        
        return outputs
    }
}
