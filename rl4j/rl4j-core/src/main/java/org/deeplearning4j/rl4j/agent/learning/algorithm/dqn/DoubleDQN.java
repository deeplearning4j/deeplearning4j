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
import org.nd4j.linalg.factory.Nd4j;

/**
 * The Double-DQN algorithm based on "Deep Reinforcement Learning with Double Q-learning" (https://arxiv.org/abs/1509.06461)
 *
 * @author Alexandre Boulanger
 */
public class DoubleDQN extends BaseDQNAlgorithm {

    private static final int ACTION_DIMENSION_IDX = 1;

    // In litterature, this corresponds to: max_{a}Q(s_{t+1}, a)
    private INDArray maxActionsFromQNetworkNextObservation;

    public DoubleDQN(IOutputNeuralNet qNetwork, IOutputNeuralNet targetQNetwork, double gamma) {
        super(qNetwork, targetQNetwork, gamma);
    }

    public DoubleDQN(IOutputNeuralNet qNetwork, IOutputNeuralNet targetQNetwork, double gamma, double errorClamp) {
        super(qNetwork, targetQNetwork, gamma, errorClamp);
    }

    @Override
    protected void initComputation(INDArray observations, INDArray nextObservations) {
        super.initComputation(observations, nextObservations);

        maxActionsFromQNetworkNextObservation = Nd4j.argMax(qNetworkNextObservation, ACTION_DIMENSION_IDX);
    }

    /**
     * In litterature, this corresponds to:<br />
     *      Q(s_t, a_t) = R_{t+1} + \gamma * Q_{tar}(s_{t+1}, max_{a}Q(s_{t+1}, a))
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    @Override
    protected double computeTarget(int batchIdx, double reward, boolean isTerminal) {
        double yTarget = reward;
        if (!isTerminal) {
            yTarget += gamma * targetQNetworkNextObservation.getDouble(batchIdx, maxActionsFromQNetworkNextObservation.getInt(batchIdx));
        }

        return yTarget;
    }
}
