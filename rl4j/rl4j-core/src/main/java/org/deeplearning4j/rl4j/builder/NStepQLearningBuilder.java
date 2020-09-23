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
package org.deeplearning4j.rl4j.builder;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncSharedNetworksUpdateHandler;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IActionSchema;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.INeuralNetPolicy;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.rng.Random;

/**
 * A {@link IAgentLearner} builder that will setup a {@link NStepQLearning n-step Q-Learning} algorithm with these:
 * <li>a epsilon-greedy policy</li>
 * <li>a n-step state-action-reward experience handler</li>
 * <li>a neural net updater that expects gradient update data</li>
 * <li>a n-step Q-Learning gradient conputation algorithm</li>
 */
public class NStepQLearningBuilder extends BaseAsyncAgentLearnerBuilder<NStepQLearningBuilder.Configuration> {

    private final Random rnd;

    public NStepQLearningBuilder(Configuration configuration,
                                 ITrainableNeuralNet neuralNet,
                                 Builder<Environment<Integer>> environmentBuilder,
                                 Builder<TransformProcess> transformProcessBuilder,
                                 Random rnd) {
        super(configuration, neuralNet, environmentBuilder, transformProcessBuilder);

        // TODO: remove once RNN networks states are stored in the experience elements
        Preconditions.checkArgument(!neuralNet.isRecurrent() || configuration.getExperienceHandlerConfiguration().getBatchSize() == Integer.MAX_VALUE,
                "RL with a recurrent network currently only works with whole-trajectory updates. Until RNN are fully supported, please set the batch size of your experience handler to Integer.MAX_VALUE");

        this.rnd = rnd;
    }

    @Override
    protected IPolicy<Integer> buildPolicy() {
        INeuralNetPolicy<Integer> greedyPolicy = new DQNPolicy<Integer>(networks.getThreadCurrentNetwork());
        IActionSchema<Integer> actionSchema = getEnvironment().getSchema().getActionSchema();
        return new EpsGreedy(greedyPolicy, actionSchema, configuration.getPolicyConfiguration(), rnd);
    }

    @Override
    protected IUpdateAlgorithm<Gradients, StateActionPair<Integer>> buildUpdateAlgorithm() {
        IActionSchema<Integer> actionSchema = getEnvironment().getSchema().getActionSchema();
        return new NStepQLearning(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(), actionSchema.getActionSpaceSize(), configuration.getNstepQLearningConfiguration());
    }

    @Override
    protected AsyncSharedNetworksUpdateHandler buildAsyncSharedNetworksUpdateHandler() {
        return new AsyncSharedNetworksUpdateHandler(networks.getGlobalCurrentNetwork(), networks.getTargetNetwork(), configuration.getNeuralNetUpdaterConfiguration());
    }

    @EqualsAndHashCode(callSuper = true)
    @SuperBuilder
    @Data
    public static class Configuration extends BaseAsyncAgentLearnerBuilder.Configuration {
        NStepQLearning.Configuration nstepQLearningConfiguration;
    }
}
