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
 * Implementation of ONNX NegativeLogLikelihoodLoss operation.
 *
 * ONNX spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss
 *
 * Computes negative log likelihood loss given log-probabilities and labels.
 * loss = -sum(log_probs[i][labels[i]] * weight[labels[i]]) / normalizer
 *
 * Inputs:
 * - input: Log probabilities of shape [N, C] or [N, C, d1, d2, ..., dk]
 * - target: Target labels of shape [N] or [N, d1, d2, ..., dk]
 * - weight: Optional weights for each class
 *
 * Attributes:
 * - ignore_index: Label to ignore
 * - reduction: "none", "mean", "sum"
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["NegativeLogLikelihoodLoss"], frameworkName = "onnx")
class NegativeLogLikelihoodLoss : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])  // Log probabilities [N, C] or [N, C, d1, ...]
        val target = sd.getVariable(op.inputsToOp[1])  // Target labels [N] or [N, d1, ...]
        val weight = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        
        // Get attributes
        val ignoreIndex = attributes["ignore_index"] as? Number
        val reduction = attributes.getOrDefault("reduction", "mean") as? String ?: "mean"
        
        // Gather log probabilities for correct class
        // For 2D: gather input[i, target[i]] for each i
        val inputRank = sd.rank(input)

        // Use gather_nd to select the log probability of the correct class
        // Create indices of shape [N, 2] where each row is [batch_idx, class_idx]
        val targetShape = target.shape ?: throw IllegalStateException("NegativeLogLikelihoodLoss requires static target shape")
        val batchSize = targetShape[0]
        val batchIndices = sd.range("${opName}_range", sd.constant(0L), sd.constant(batchSize), sd.constant(1L), DataType.INT64)
        val targetLong = target.castTo(DataType.INT64)
        
        // Stack batch indices with target indices: [[0, t0], [1, t1], ...]
        val gatherIndices = sd.stack("${opName}_indices", 1, batchIndices, targetLong)
        
        // Gather the log probabilities for the target classes
        val selectedLogProbs = sd.gatherNd("${opName}_gather", input, gatherIndices)
        
        // Negate to get NLL
        var loss = sd.math.neg("${opName}_neg", selectedLogProbs)
        
        // Apply class weights if provided
        if (weight != null) {
            val classWeights = sd.gather("${opName}_weights", weight, targetLong, 0)
            loss = sd.math.mul("${opName}_weighted", loss, classWeights)
        }
        
        // Handle ignore_index
        if (ignoreIndex != null) {
            val mask = sd.neq("${opName}_mask", target, ignoreIndex.toDouble()).castTo(loss.dataType())
            loss = sd.math.mul("${opName}_masked", loss, mask)
        }

        // Apply reduction
        val output = when (reduction) {
            "none" -> loss.rename(outputNames[0])
            "sum" -> sd.math.sum(outputNames[0], loss)
            "mean" -> {
                if (ignoreIndex != null) {
                    // Mean over non-ignored elements
                    val mask = sd.neq("${opName}_mask2", target, ignoreIndex.toDouble()).castTo(DataType.FLOAT)
                    val count = sd.math.sum("${opName}_count", mask)
                    sd.math.div(outputNames[0], sd.math.sum(loss), count)
                } else {
                    sd.math.mean(outputNames[0], loss)
                }
            }
            else -> sd.math.mean(outputNames[0], loss)
        }
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
