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
package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class QLearningUpdateAlgorithm implements UpdateAlgorithm<IDQN> {

    private final int[] shape;
    private final int actionSpaceSize;
    private final double gamma;

    public QLearningUpdateAlgorithm(int[] shape,
                                    int actionSpaceSize,
                                    double gamma) {

        this.shape = shape;
        this.actionSpaceSize = actionSpaceSize;
        this.gamma = gamma;
    }

    @Override
    public Gradient[] computeGradients(IDQN current, List<StateActionPair<Integer>> experience) {
        int size = experience.size();

        int[] nshape = Learning.makeShape(size, shape);
        INDArray input = Nd4j.create(nshape);
        INDArray targets = Nd4j.create(size, actionSpaceSize);

        StateActionPair<Integer> stateActionPair = experience.get(size - 1);

        double r;
        if (stateActionPair.isTerminal()) {
            r = 0;
        } else {
            INDArray[] output = null;
            output = current.outputAll(stateActionPair.getObservation().getData());
            r = Nd4j.max(output[0]).getDouble(0);
        }

        for (int i = size - 1; i >= 0; i--) {
            stateActionPair = experience.get(i);

            input.putRow(i, stateActionPair.getObservation().getData());

            r = stateActionPair.getReward() + gamma * r;
            INDArray[] output = current.outputAll(stateActionPair.getObservation().getData());
            INDArray row = output[0];
            row = row.putScalar(stateActionPair.getAction(), r);
            targets.putRow(i, row);
        }

        return current.gradient(input, targets);
    }
}
