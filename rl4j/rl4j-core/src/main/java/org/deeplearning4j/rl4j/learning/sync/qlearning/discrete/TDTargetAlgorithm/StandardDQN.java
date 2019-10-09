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

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm;

import org.deeplearning4j.rl4j.learning.sync.qlearning.TargetQNetworkSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The Standard DQN algorithm based on "Playing Atari with Deep Reinforcement Learning" (https://arxiv.org/abs/1312.5602)
 *
 * @author Alexandre Boulanger
 */
public class StandardDQN extends BaseDQNAlgorithm {

    private static final int ACTION_DIMENSION_IDX = 1;

    // In litterature, this corresponds to: max_{a}Q_{tar}(s_{t+1}, a)
    private INDArray maxActionsFromQTargetNextObservation;

    public StandardDQN(TargetQNetworkSource qTargetNetworkSource, double gamma) {
        super(qTargetNetworkSource, gamma);
    }

    public StandardDQN(TargetQNetworkSource qTargetNetworkSource, double gamma, double errorClamp) {
        super(qTargetNetworkSource, gamma, errorClamp);
    }

    @Override
    protected void initComputation(INDArray observations, INDArray nextObservations) {
        super.initComputation(observations, nextObservations);

        maxActionsFromQTargetNextObservation = Nd4j.max(targetQNetworkNextObservation, ACTION_DIMENSION_IDX);
    }

    /**
     * In litterature, this corresponds to:<br />
     *      Q(s_t, a_t) = R_{t+1} + \gamma * max_{a}Q_{tar}(s_{t+1}, a)
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    @Override
    protected double computeTarget(int batchIdx, double reward, boolean isTerminal) {
        double yTarget = reward;
        if (!isTerminal) {
            yTarget += gamma * maxActionsFromQTargetNextObservation.getDouble(batchIdx);
        }

        return yTarget;
    }
}
