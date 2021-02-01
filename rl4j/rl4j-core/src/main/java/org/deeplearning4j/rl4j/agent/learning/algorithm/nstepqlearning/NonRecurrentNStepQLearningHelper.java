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
package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * A helper class for the n-step Q-Learning update algorithm. The algorithm is the same whether it's used with a RNN or
 * not but, the shape of INDArrays are different. This class handles the non-recurrent case.
 */
public class NonRecurrentNStepQLearningHelper extends NStepQLearningHelper {
    private final int actionSpaceSize;

    public NonRecurrentNStepQLearningHelper(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
    }

    @Override
    public INDArray createLabels(int trainingBatchSize) {
        return Nd4j.create(trainingBatchSize, actionSpaceSize);
    }

    @Override
    public INDArray getExpectedQValues(INDArray allExpectedQValues, int idx) {
        return allExpectedQValues.getRow(idx);
    }

    @Override
    public void setLabels(INDArray labels, long idx, INDArray data) {
        labels.putRow(idx, data);
    }

    @Override
    public INDArray getTargetExpectedQValuesOfLast(IOutputNeuralNet target, List<StateActionReward<Integer>> trainingBatch, Features features) {
        Observation lastObservation = trainingBatch.get(trainingBatch.size() - 1).getObservation();
        return target.output(lastObservation)
                .get(CommonOutputNames.QValues);
    }
}
