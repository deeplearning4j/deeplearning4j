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
package org.deeplearning4j.rl4j.agent.learning.algorithm;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

// TODO: Add support for RNN
/**
 * This the "Algorithm S3 Asynchronous advantage actor-critic" of <i>Asynchronous Methods for Deep Reinforcement Learning</i>
 * @see <a href="https://arxiv.org/pdf/1602.01783.pdf">Asynchronous Methods for Deep Reinforcement Learning on arXiv</a>, page 14
 * <p/>
 * Note: The output of threadCurrent must contain a channel named "value".
 */
public class AdvantageActorCritic implements IUpdateAlgorithm<Gradients, StateActionPair<Integer>> {

    private final ITrainableNeuralNet threadCurrent;

    private final int actionSpaceSize;
    private final double gamma;

    public AdvantageActorCritic(@NonNull ITrainableNeuralNet threadCurrent,
                                int actionSpaceSize,
                                @NonNull Configuration configuration) {
        this.threadCurrent = threadCurrent;
        this.actionSpaceSize = actionSpaceSize;
        gamma = configuration.getGamma();
    }

    @Override
    public Gradients compute(List<StateActionPair<Integer>> trainingBatch) {
        int size = trainingBatch.size();

        INDArray features = INDArrayHelper.createBatchForShape(size, trainingBatch.get(0).getObservation().getData().shape());
        INDArray values = Nd4j.create(size, 1);
        INDArray policy = Nd4j.zeros(size, actionSpaceSize);

        StateActionPair<Integer> stateActionPair = trainingBatch.get(size - 1);
        double value;
        if (stateActionPair.isTerminal()) {
            value = 0;
        } else {
            INDArray valueOutput = threadCurrent.output(stateActionPair.getObservation()).get(CommonOutputNames.ActorCritic.Value);
            value = valueOutput.getDouble(0);
        }

        for (int i = size - 1; i >= 0; --i) {
            stateActionPair = trainingBatch.get(i);

            Observation observation = stateActionPair.getObservation();

            features.putRow(i, observation.getData());

            value = stateActionPair.getReward() + gamma * value;

            //the critic
            values.putScalar(i, value);

            //the actor
            double expectedV = threadCurrent.output(observation)
                    .get(CommonOutputNames.ActorCritic.Value)
                    .getDouble(0);
            double advantage = value - expectedV;
            policy.putScalar(i, stateActionPair.getAction(), advantage);
        }

        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels(CommonLabelNames.ActorCritic.Value, values);
        featuresLabels.putLabels(CommonLabelNames.ActorCritic.Policy, policy);

        return threadCurrent.computeGradients(featuresLabels);
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The discount factor (default is 0.99)
         */
        @Builder.Default
        double gamma = 0.99;
    }
}
