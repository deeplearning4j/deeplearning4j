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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException

/**
 * Basic import of GRU in to samediff.
 * Implements forward and linear before reset = false only.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["GRU"],frameworkName = "onnx")
class GRU : PreImportHook  {


    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val direction = if(attributes.containsKey("direction")) attributes["forward"]
        else "forward"
        if(direction != "forward") {
            throw IllegalArgumentException("GRU Import: Only forward direction implemented")
        }

        //dl4j: input [time, bS, nIn]
        //onnx: [seq_length, batch_size, input_size]`
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        //
        //onnx: num_directions, 3*hidden_size, input_size
        //dl4j: input-to-hidden  weights, [nIn, 3*nOut]
        val weights = sd.getVariable(op.inputsToOp[1])
        val weightShape = sd.squeeze(weights,0)
        //get rid of num_directions (only 1 is supported)
        //permute to be nIn, 3 * hiddenSize
        val inputWeights = sd.transpose(weightShape)
        //dl4j: hidden-to-hidden weights, [nOut, 3*nOut]
        //onnx: num_directions, 3*hidden_size, hidden_size]
        val r = sd.getVariable(op.inputsToOp[2])
        val nOut = r.shape().get(SDIndex.point(-1))
        val rShape = sd.squeeze(r,0)
        //get rid of num_directions (only 1 is supported)
        //permute to be nIn, 3 * hiddenSize
        val inputR = sd.transpose(rShape)

        //onnx: This tensor has shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed to be 0
        //dl4j: biases, [3*nOut]
        val bias = getBias(op,sd, nOut ,inputR.dataType())
        if(!op.inputsToOp[4].isEmpty()) {
            throw IllegalArgumentException("Custom sequence lengths not implemented.")
        }

        val seqLens = if(!op.inputsToOp[4].isEmpty()) sd.getVariable(op.inputsToOp[4])
        else sd.constant(0) // TODO: fix
        //onnx: num_directions, batch_size, hidden_size
        //dl4j: initial cell output (at time step = 0) [bS, nOut]
        val initialH = if(!op.inputsToOp[5].isEmpty()) sd.getVariable(op.inputsToOp[5]) else sd.constant(0)
        val initialHShape = sd.squeeze(initialH,0)

        val gruOutput = sd.rnn().gru(op.outputsOfOp[0],inputVariable,initialHShape,inputWeights,inputR,bias)


        return mapOf(op.outputsOfOp[0] to listOf(gruOutput))

    }


    private fun getBias(
        op: SameDiffOp,
        sd: SameDiff,
        hiddenLayerSize: SDVariable,
        dt: org.nd4j.linalg.api.buffer.DataType
    ): SDVariable {
        if (op.inputsToOp[3].isNotEmpty()) {
            val onnxBias = sd.getVariable(op.inputsToOp[3])
            //if so we could just halve this and be good to go. If so a splice is good enough.
            val onnxBiasInput = sd.squeeze(onnxBias, 0)
            //we only do the forward pass in this op, we'll only grab the subset of
            //the bias that matters for us
            //get rid of num_directions (only 1 is supported)
            val subsetBias = onnxBiasInput.get(SDIndex.interval(sd.constant(0), onnxBiasInput.length().div(2.0)))
            //permute to be nIn, 3 * hiddenSize
            return subsetBias.castTo(dt)
        }

        val constShape = Nd4j.create(Nd4j.createBuffer(longArrayOf(3, hiddenLayerSize as Long)))
        val constShape2 = sd.constant(constShape)

        return sd.create(constShape2, dt)

    }

}