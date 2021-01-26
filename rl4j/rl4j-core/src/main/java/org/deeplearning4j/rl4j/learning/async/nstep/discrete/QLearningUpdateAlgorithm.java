/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class QLearningUpdateAlgorithm implements UpdateAlgorithm<IDQN> {

    private final int actionSpaceSize;
    private final double gamma;

    public QLearningUpdateAlgorithm(int actionSpaceSize,
                                    double gamma) {

        this.actionSpaceSize = actionSpaceSize;
        this.gamma = gamma;
    }

    @Override
    public Gradient[] computeGradients(IDQN current, List<StateActionReward<Integer>> experience) {
        int size = experience.size();

        StateActionReward<Integer> stateActionReward = experience.get(size - 1);

        INDArray data = stateActionReward.getObservation().getChannelData(0);
        INDArray features = INDArrayHelper.createBatchForShape(size, data.shape());
        INDArray targets = Nd4j.create(size, actionSpaceSize);

        double r;
        if (stateActionReward.isTerminal()) {
            r = 0;
        } else {
            INDArray[] output = null;
            output = current.outputAll(data);
            r = Nd4j.max(output[0]).getDouble(0);
        }

        for (int i = size - 1; i >= 0; i--) {
            stateActionReward = experience.get(i);
            data = stateActionReward.getObservation().getChannelData(0);

            features.putRow(i, data);

            r = stateActionReward.getReward() + gamma * r;
            INDArray[] output = current.outputAll(data);
            INDArray row = output[0];
            row = row.putScalar(stateActionReward.getAction(), r);
            targets.putRow(i, row);
        }

        return current.gradient(features, targets);
    }
}
