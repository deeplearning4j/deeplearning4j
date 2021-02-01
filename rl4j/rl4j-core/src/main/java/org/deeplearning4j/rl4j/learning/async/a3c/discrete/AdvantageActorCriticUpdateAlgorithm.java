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
package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * The Advantage Actor-Critic update algorithm can be used by A2C and A3C algorithms alike
 */
public class AdvantageActorCriticUpdateAlgorithm implements UpdateAlgorithm<IActorCritic> {

    private final int[] shape;
    private final int actionSpaceSize;
    private final double gamma;
    private final boolean recurrent;

    public AdvantageActorCriticUpdateAlgorithm(boolean recurrent,
                                               int[] shape,
                                               int actionSpaceSize,
                                               double gamma) {

        //if recurrent then train as a time serie with a batch size of 1
        this.recurrent = recurrent;
        this.shape = shape;
        this.actionSpaceSize = actionSpaceSize;
        this.gamma = gamma;
    }

    @Override
    public Gradient[] computeGradients(IActorCritic current, List<StateActionReward<Integer>> experience) {
        int size = experience.size();

        int[] nshape = recurrent ? Learning.makeShape(1, shape, size)
                : Learning.makeShape(size, shape);

        INDArray input = Nd4j.create(nshape);
        INDArray targets = recurrent ? Nd4j.create(1, 1, size) : Nd4j.create(size, 1);
        INDArray logSoftmax = recurrent ? Nd4j.zeros(1, actionSpaceSize, size)
                : Nd4j.zeros(size, actionSpaceSize);

        StateActionReward<Integer> stateActionReward = experience.get(size - 1);
        double value;
        if (stateActionReward.isTerminal()) {
            value = 0;
        } else {
            INDArray[] output = current.outputAll(stateActionReward.getObservation().getChannelData(0));
            value = output[0].getDouble(0);
        }

        for (int i = size - 1; i >= 0; --i) {
            stateActionReward = experience.get(i);

            INDArray observationData = stateActionReward.getObservation().getChannelData(0);

            INDArray[] output = current.outputAll(observationData);

            value = stateActionReward.getReward() + gamma * value;
            if (recurrent) {
                input.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)).assign(observationData);
            } else {
                input.putRow(i, observationData);
            }

            //the critic
            targets.putScalar(i, value);

            //the actor
            double expectedV = output[0].getDouble(0);
            double advantage = value - expectedV;
            if (recurrent) {
                logSoftmax.putScalar(0, stateActionReward.getAction(), i, advantage);
            } else {
                logSoftmax.putScalar(i, stateActionReward.getAction(), advantage);
            }
        }

        // targets -> value, critic
        // logSoftmax -> policy, actor
        return current.gradient(input, new INDArray[]{targets, logSoftmax});
    }
}
