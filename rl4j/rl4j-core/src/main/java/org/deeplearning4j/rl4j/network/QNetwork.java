/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A QNetwork implementation.<br/>
 * Label names: "Q"<br/>
 * Gradient names: "Q"<br/>
 */
public class QNetwork extends BaseNetwork<QNetwork> {

    public QNetwork(ComputationGraph model) {
        this(new ComputationGraphHandler(model, new String[] { CommonLabelNames.QValues }, CommonGradientNames.QValues));
    }

    public QNetwork(MultiLayerNetwork model) {
        this(new MultiLayerNetworkHandler(model, CommonLabelNames.QValues, CommonGradientNames.QValues));
    }

    private QNetwork(INetworkHandler handler) {
        super(handler);
    }

    @Override
    protected NeuralNetOutput packageResult(INDArray[] output) {
        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.QValues, output[0]);

        return result;
    }

    @Override
    public QNetwork clone() {
        return new QNetwork(getNetworkHandler().clone());
    }
}