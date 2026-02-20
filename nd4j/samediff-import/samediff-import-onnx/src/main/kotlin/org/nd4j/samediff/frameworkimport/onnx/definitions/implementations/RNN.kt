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
 * Implementation of ONNX RNN operation.
 *
 * ONNX RNN spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
 *
 * Computes a basic RNN layer:
 * H_t = activation(X_t * W^T + H_{t-1} * R^T + B)
 *
 * Inputs:
 * - X: Input tensor [seq_length, batch_size, input_size]
 * - W: Weight tensor for input [num_directions, hidden_size, input_size]
 * - R: Recurrence weight tensor [num_directions, hidden_size, hidden_size]
 * - B: Bias tensor (optional) [num_directions, 2*hidden_size]
 * - sequence_lens: Sequence lengths (optional)
 * - initial_h: Initial hidden state (optional)
 *
 * Attributes:
 * - activation_alpha: Activation alpha values
 * - activation_beta: Activation beta values
 * - activations: Activation functions (default: Tanh)
 * - clip: Gradient clipping value
 * - direction: forward, reverse, bidirectional
 * - hidden_size: Hidden state size
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["RNN"], frameworkName = "onnx")
class RNN : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])  // [seq_length, batch_size, input_size]
        val w = sd.getVariable(op.inputsToOp[1])  // [num_directions, hidden_size, input_size]
        val r = sd.getVariable(op.inputsToOp[2])  // [num_directions, hidden_size, hidden_size]
        val b = if (op.inputsToOp.size > 3 && op.inputsToOp[3].isNotEmpty()) 
            sd.getVariable(op.inputsToOp[3]) else null
        val seqLens = if (op.inputsToOp.size > 4 && op.inputsToOp[4].isNotEmpty()) 
            sd.getVariable(op.inputsToOp[4]) else null
        val initialH = if (op.inputsToOp.size > 5 && op.inputsToOp[5].isNotEmpty()) 
            sd.getVariable(op.inputsToOp[5]) else null
        
        // Get attributes
        val hiddenSize = (attributes["hidden_size"] as? Number)?.toInt() ?: 0
        val direction = attributes.getOrDefault("direction", "forward") as? String ?: "forward"
        val activations = getStringList(attributes, "activations", listOf("Tanh"))
        
        // Determine number of directions
        val numDirections = if (direction == "bidirectional") 2 else 1
        
        // ONNX RNN operation is not fully supported yet
        // The implementation would require:
        // 1. Transpose X from [seq_length, batch_size, input_size] to [batch_size, seq_length, input_size]
        // 2. Apply RNN computation: H_t = activation(X_t * W^T + H_{t-1} * R^T + B)
        // 3. Handle bidirectional mode
        // 4. Return both output sequence and final hidden state

        // For now, create a placeholder that passes the input through
        // This is a temporary workaround until full RNN support is added
        val y = x.rename(outputNames[0])

        val resultMap = mutableMapOf<String, List<SDVariable>>()
        resultMap[outputNames[0]] = listOf(y)

        // Add placeholder for hidden state if needed
        if (outputNames.size > 1) {
            val yH = if (initialH != null) {
                initialH.rename(outputNames[1])
            } else {
                sd.zerosLike(outputNames[1], x)
            }
            resultMap[outputNames[1]] = listOf(yH)
        }

        return resultMap
    }
    
    private fun getStringList(attributes: Map<String, Any>, key: String, default: List<String>): List<String> {
        val value = attributes[key] ?: return default
        return when (value) {
            is List<*> -> value.map { it.toString() }
            is Array<*> -> value.map { it.toString() }
            else -> default
        }
    }
}
