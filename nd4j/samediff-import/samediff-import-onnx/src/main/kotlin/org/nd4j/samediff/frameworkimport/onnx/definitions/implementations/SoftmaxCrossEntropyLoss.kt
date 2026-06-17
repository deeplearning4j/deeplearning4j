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
 * Implementation of ONNX SoftmaxCrossEntropyLoss operation.
 *
 * ONNX spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss
 *
 * Computes softmax cross entropy loss:
 * loss = -sum(labels * log(softmax(scores)))
 *
 * Inputs:
 * - scores: Unnormalized scores of shape [N, C] or [N, C, d1, d2, ..., dk]
 * - labels: Target class indices of shape [N] or [N, d1, d2, ..., dk]
 * - weights: Optional per-class weights
 *
 * Attributes:
 * - ignore_index: Label to ignore in loss calculation
 * - reduction: "none", "mean", "sum"
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["SoftmaxCrossEntropyLoss"], frameworkName = "onnx")
class SoftmaxCrossEntropyLoss : PreImportHook {

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
        val scores = sd.getVariable(op.inputsToOp[0])  // Unnormalized scores [N, C] or [N, C, d1, ...]
        val labels = sd.getVariable(op.inputsToOp[1])  // Target labels [N] or [N, d1, ...]
        val weight = if (op.inputsToOp.size > 2) sd.getVariable(op.inputsToOp[2]) else null
        
        // Get attributes
        val ignoreIndex = attributes["ignore_index"] as? Number
        val reduction = attributes.getOrDefault("reduction", "mean") as? String ?: "mean"
        
        // Compute log softmax for numerical stability
        // log_softmax = scores - log(sum(exp(scores)))
        val logSoftmax = sd.nn.logSoftmax("${opName}_logsoftmax", scores, 1)

        // Gather log probabilities for correct classes
        // Get batch size from shape - extract first dimension
        val labelsShape = sd.shape(labels)
        val batchSize = sd.slice("${opName}_batch_size", labelsShape, intArrayOf(0), 1)
        val batchIndices = sd.range("${opName}_range", sd.constant(0L), batchSize, sd.constant(1L), DataType.INT64)
        val labelsLong = labels.castTo(DataType.INT64)

        // Create gather indices
        val gatherIndices = sd.stack("${opName}_indices", 1, batchIndices, labelsLong)

        // Gather log probabilities for the target classes
        val selectedLogProbs = sd.gatherNd("${opName}_gather", logSoftmax, gatherIndices)

        // Negate to get cross entropy loss
        var loss = sd.math.neg("${opName}_neg", selectedLogProbs)
        
        // Apply class weights if provided
        if (weight != null) {
            val classWeights = sd.gather("${opName}_weights", weight, labelsLong, 0)
            loss = sd.math.mul("${opName}_weighted", loss, classWeights)
        }
        
        // Handle ignore_index
        if (ignoreIndex != null) {
            val ignoreIndexConst = sd.constant(ignoreIndex.toDouble())
            val mask = sd.neq("${opName}_mask", labels.castTo(DataType.DOUBLE), ignoreIndexConst).castTo(loss.dataType())
            loss = sd.math.mul("${opName}_masked", loss, mask)
        }

        // Apply reduction
        val output = when (reduction) {
            "none" -> loss.rename(outputNames[0])
            "sum" -> sd.math.sum(outputNames[0], loss)
            "mean" -> {
                if (ignoreIndex != null) {
                    // Mean over non-ignored elements
                    val ignoreIndexConst2 = sd.constant(ignoreIndex.toDouble())
                    val mask = sd.neq("${opName}_mask2", labels.castTo(DataType.DOUBLE), ignoreIndexConst2).castTo(DataType.FLOAT)
                    val count = sd.math.sum("${opName}_count", mask)
                    sd.math.div(outputNames[0], sd.math.sum(loss), count)
                } else {
                    sd.math.mean(outputNames[0], loss)
                }
            }
            else -> sd.math.mean(outputNames[0], loss)
        }
        
        // This op can have 2 outputs: loss and log_prob
        // If there's a second output name, return log probabilities
        return if (outputNames.size > 1) {
            val logProb = logSoftmax.rename(outputNames[1])
            mapOf(
                outputNames[0] to listOf(output),
                outputNames[1] to listOf(logProb)
            )
        } else {
            mapOf(outputNames[0] to listOf(output))
        }
    }
}
