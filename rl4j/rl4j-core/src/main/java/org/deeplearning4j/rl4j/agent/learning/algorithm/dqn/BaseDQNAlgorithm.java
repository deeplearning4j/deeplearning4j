/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The base of all DQN based algorithms
 *
 * @author Alexandre Boulanger
 *
 */
public abstract class BaseDQNAlgorithm extends BaseTransitionTDAlgorithm {

    private final IOutputNeuralNet targetQNetwork;

    /**
     * In litterature, this corresponds to Q{net}(s(t+1), a)
     */
    protected INDArray qNetworkNextObservation;

    /**
     * In litterature, this corresponds to Q{tnet}(s(t+1), a)
     */
    protected INDArray targetQNetworkNextObservation;

    protected BaseDQNAlgorithm(IOutputNeuralNet qNetwork, IOutputNeuralNet targetQNetwork, double gamma) {
        super(qNetwork, gamma);
        this.targetQNetwork = targetQNetwork;
    }

    protected BaseDQNAlgorithm(IOutputNeuralNet qNetwork, IOutputNeuralNet targetQNetwork, double gamma, double errorClamp) {
        super(qNetwork, gamma, errorClamp);
        this.targetQNetwork = targetQNetwork;
    }

    @Override
    protected void initComputation(INDArray observations, INDArray nextObservations) {
        super.initComputation(observations, nextObservations);

        qNetworkNextObservation = qNetwork.output(nextObservations);
        targetQNetworkNextObservation = targetQNetwork.output(nextObservations);
    }
}
