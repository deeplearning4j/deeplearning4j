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

@PreHookRule(nodeNames = [], opNames = ["EmbedLayerNormalization"], frameworkName = "onnx")
class EmbedLayerNormalization : PreImportHook {

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
        return HookResult(
            outputVariables = handleOutputs(
                outputNames,
                sd,
                op,
                attributes,
                mappingRegistry,
                importGraph,
                dynamicVariables
            ),
            proceedWithInit = false
        )
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
        
        // Get inputs
        val inputIds = sd.getVariable(op.inputsToOp[0])
        val wordEmbedding = sd.getVariable(op.inputsToOp[2])
        val positionEmbedding = sd.getVariable(op.inputsToOp[3])
        
        val segmentIds = if (op.inputsToOp.size > 1 && op.inputsToOp[1] != null && op.inputsToOp[1].isNotEmpty()) {
            sd.getVariable(op.inputsToOp[1])
        } else null
        
        val segmentEmbedding: SDVariable?
        val gamma: SDVariable
        val beta: SDVariable
        val mask: SDVariable?
        
        when (op.inputsToOp.size) {
            8 -> {
                segmentEmbedding = sd.getVariable(op.inputsToOp[4])
                gamma = sd.getVariable(op.inputsToOp[5])
                beta = sd.getVariable(op.inputsToOp[6])
                mask = sd.getVariable(op.inputsToOp[7])
            }
            7 -> {
                val var4 = sd.getVariable(op.inputsToOp[4])
                val var4Shape = var4.shape
                if (var4Shape != null && var4Shape.size == 2) {
                    segmentEmbedding = var4
                    gamma = sd.getVariable(op.inputsToOp[5])
                    beta = sd.getVariable(op.inputsToOp[6])
                    mask = null
                } else {
                    segmentEmbedding = null
                    gamma = sd.getVariable(op.inputsToOp[4])
                    beta = sd.getVariable(op.inputsToOp[5])
                    mask = sd.getVariable(op.inputsToOp[6])
                }
            }
            6 -> {
                segmentEmbedding = null
                gamma = sd.getVariable(op.inputsToOp[4])
                beta = sd.getVariable(op.inputsToOp[5])
                mask = null
            }
            else -> {
                segmentEmbedding = null
                gamma = sd.getVariable(op.inputsToOp[op.inputsToOp.size - 2])
                beta = sd.getVariable(op.inputsToOp[op.inputsToOp.size - 1])
                mask = null
            }
        }
        
        // Cast input IDs to INT32 if needed
        val inputIdsInt = if (inputIds.dataType().isIntType()) {
            inputIds
        } else {
            inputIds.castTo(DataType.INT32)
        }
        
        // Perform word embedding lookup using gather
        val wordEmbedded = sd.gather(wordEmbedding, inputIdsInt, 0)
        
        // Instead of trying to dynamically get the sequence length from shape operations,
        // use the position embedding's first dimension as the maximum sequence length
        // This avoids the slice operation that's causing issues
        val positionEmbeddingShape = positionEmbedding.shape
        val maxSeqLen = if (positionEmbeddingShape != null && positionEmbeddingShape.isNotEmpty()) {
            positionEmbeddingShape[0]
        } else {
            512L // Default fallback
        }
        
        // Get actual sequence length from input
        val inputShape = inputIds.shape
        val actualSeqLen = if (inputShape != null && inputShape.size >= 2) {
            inputShape[1] // For [batch, seq] input
        } else if (inputShape != null && inputShape.size == 1) {
            inputShape[0] // For [seq] input
        } else {
            maxSeqLen
        }
        
        // Create position IDs based on actual sequence length
        val seqLenToUse = minOf(actualSeqLen, maxSeqLen)
        val positionIds = sd.range(0.0, seqLenToUse.toDouble(), 1.0, DataType.INT32)
        val positionEmbedded = sd.gather(positionEmbedding, positionIds, 0)
        
        // Expand position embeddings to add batch dimension if needed
        val positionEmbeddedExpanded = sd.expandDims(positionEmbedded, 0)
        
        // Add word and position embeddings (broadcasting will handle the batch dimension)
        var embeddings = wordEmbedded.add(positionEmbeddedExpanded)
        
        // Handle segment embeddings if present
        if (segmentIds != null && segmentEmbedding != null) {
            val segmentIdsInt = if (segmentIds.dataType().isIntType()) {
                segmentIds
            } else {
                segmentIds.castTo(DataType.INT32)
            }
            val segmentEmbedded = sd.gather(segmentEmbedding, segmentIdsInt, 0)
            embeddings = embeddings.add(segmentEmbedded)
        }
        
        // Apply layer normalization using the SameDiff API
        // The last dimension (-1) is used for normalization in transformer embeddings
        val normalized = sd.nn().layerNorm(outputNames[0], embeddings, gamma, beta, false, -1L)
        
        // Prepare outputs
        val outputs = mutableMapOf<String, List<SDVariable>>()
        outputs[outputNames[0]] = listOf(normalized)
        
        if (outputNames.size > 1) {
            val maskOutput = mask ?: sd.zerosLike(inputIds).castTo(DataType.INT32)
            outputs[outputNames[1]] = listOf(maskOutput.rename(outputNames[1]))
        }
        
        if (outputNames.size > 2) {
            outputs[outputNames[2]] = listOf(embeddings.rename(outputNames[2]))
        }
        
        return outputs
    }
}
