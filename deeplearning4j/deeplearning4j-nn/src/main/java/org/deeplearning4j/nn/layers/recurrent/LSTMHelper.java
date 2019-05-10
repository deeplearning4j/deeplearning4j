/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Map;

/**
 * Helper for the recurrent LSTM layer (no peephole connections).
 *
 * @author saudet
 */
public interface LSTMHelper extends LayerHelper {
    boolean checkSupported(IActivation gateActivationFn, IActivation activationFn, boolean hasPeepholeConnections);

    Pair<Gradient, INDArray> backpropGradient(final NeuralNetConfiguration conf, final IActivation gateActivationFn,
                                              final INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                                              final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                                              final INDArray epsilon, final boolean truncatedBPTT, final int tbpttBackwardLength,
                                              final FwdPassReturn fwdPass, final boolean forwards, final String inputWeightKey,
                                              final String recurrentWeightKey, final String biasWeightKey,
                                              final Map<String, INDArray> gradientViews, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                                              final boolean hasPeepholeConnections, //True for GravesLSTM, false for LSTM
                                              final LayerWorkspaceMgr workspaceMgr);

    FwdPassReturn activate(final Layer layer, final NeuralNetConfiguration conf, final IActivation gateActivationFn, //Activation function for the gates - sigmoid or hard sigmoid (must be found in range 0 to 1)
                           final INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                           final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                           final INDArray biases, //Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T
                           final boolean training, final INDArray prevOutputActivations, final INDArray prevMemCellState,
                           boolean forBackprop, boolean forwards, final String inputWeightKey, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                           final boolean hasPeepholeConnections, //True for GravesLSTM, false for LSTM
                           final LayerWorkspaceMgr workspaceMgr);
}
