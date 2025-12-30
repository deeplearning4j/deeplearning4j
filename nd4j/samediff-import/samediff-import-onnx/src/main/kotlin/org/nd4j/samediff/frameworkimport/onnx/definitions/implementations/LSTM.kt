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
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX LSTM operation.
 *
 * ONNX LSTM spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
 *
 * Computes an LSTM layer.
 *
 * Inputs:
 * - X: Input tensor [seq_length, batch_size, input_size]
 * - W: Weight tensor [num_directions, 4*hidden_size, input_size]
 * - R: Recurrence weight tensor [num_directions, 4*hidden_size, hidden_size]
 * - B: Bias tensor (optional) [num_directions, 8*hidden_size]
 * - sequence_lens: Sequence lengths (optional)
 * - initial_h: Initial hidden state (optional)
 * - initial_c: Initial cell state (optional)
 * - P: Peephole weights (optional)
 *
 * Attributes:
 * - activation_alpha: Activation alpha values
 * - activation_beta: Activation beta values
 * - activations: Activation functions (default: Sigmoid, Tanh, Tanh)
 * - clip: Gradient clipping value
 * - direction: forward, reverse, bidirectional
 * - hidden_size: Hidden state size
 * - input_forget: Couple input and forget gates
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["LSTM"], frameworkName = "onnx")
class LSTM : PreImportHook {

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
        val w = sd.getVariable(op.inputsToOp[1])  // [num_directions, 4*hidden_size, input_size]
        val r = sd.getVariable(op.inputsToOp[2])  // [num_directions, 4*hidden_size, hidden_size]
        val b = if (op.inputsToOp.size > 3 && op.inputsToOp[3].isNotEmpty())
            sd.getVariable(op.inputsToOp[3]) else null
        val seqLens = if (op.inputsToOp.size > 4 && op.inputsToOp[4].isNotEmpty())
            sd.getVariable(op.inputsToOp[4]) else null
        val initialH = if (op.inputsToOp.size > 5 && op.inputsToOp[5].isNotEmpty())
            sd.getVariable(op.inputsToOp[5]) else null
        val initialC = if (op.inputsToOp.size > 6 && op.inputsToOp[6].isNotEmpty())
            sd.getVariable(op.inputsToOp[6]) else null
        val peepholes = if (op.inputsToOp.size > 7 && op.inputsToOp[7].isNotEmpty())
            sd.getVariable(op.inputsToOp[7]) else null

        // Get attributes
        val hiddenSize = (attributes["hidden_size"] as? Number)?.toInt() ?: 0
        val direction = attributes.getOrDefault("direction", "forward") as? String ?: "forward"
        val inputForget = (attributes.getOrDefault("input_forget", 0L) as Number).toInt() == 1

        // Determine number of directions
        val numDirections = if (direction == "bidirectional") 2 else 1

        // Transpose X from [seq_length, batch_size, input_size] to [batch_size, seq_length, input_size]
        val xTransposed = sd.permute("${opName}_x_permute", x, 1, 0, 2)

        // Extract weights for first direction
        val wSlice = sd.squeeze(sd.stridedSlice("${opName}_w_slice", w,
            intArrayOf(0), intArrayOf(1), intArrayOf(1)), 0)
        val rSlice = sd.squeeze(sd.stridedSlice("${opName}_r_slice", r,
            intArrayOf(0), intArrayOf(1), intArrayOf(1)), 0)

        // Build LSTM configuration
        val config = LSTMConfiguration.builder()
            .peepHole(peepholes != null)
            .forgetBias(1.0)
            .clippingCellValue(0.0)
            .dataFormat(1)  // NTS format
            .build()

        // Use SameDiff LSTM layer
        val lstmOutputs = sd.rnn.lstmLayer(
            "${opName}_lstm",
            xTransposed,
            initialC,
            initialH,
            null,  // maxTSLength
            wSlice.reshape(hiddenSize.toLong(), 4, -1),
            rSlice.reshape(hiddenSize.toLong(), 4, hiddenSize.toLong()),
            b,
            peepholes,
            config
        )

        // Get outputs and transpose back
        val yAll = lstmOutputs[0]  // All hidden states
        val yH = lstmOutputs[1]    // Last hidden state
        val yC = lstmOutputs[2]    // Last cell state

        // Transpose output back to [seq_length, num_directions, batch_size, hidden_size]
        val yPermuted = sd.permute("${opName}_y_permute", yAll, 1, 0, 2)
        val y = sd.expandDims(outputNames[0], yPermuted, 1)

        val resultMap = mutableMapOf<String, List<SDVariable>>()
        resultMap[outputNames[0]] = listOf(y)

        if (outputNames.size > 1) {
            val yHOut = sd.expandDims(outputNames[1], yH, 0)
            resultMap[outputNames[1]] = listOf(yHOut)
        }

        if (outputNames.size > 2) {
            val yCOut = sd.expandDims(outputNames[2], yC, 0)
            resultMap[outputNames[2]] = listOf(yCOut)
        }

        return resultMap
    }
}
