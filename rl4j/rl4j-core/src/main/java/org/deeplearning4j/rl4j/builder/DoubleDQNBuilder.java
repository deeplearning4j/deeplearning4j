/*
 *  ******************************************************************************
 *  *
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
package org.deeplearning4j.rl4j.builder;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.nd4j.linalg.api.rng.Random;

/**
 * A {@link IAgentLearner} builder that will setup a {@link DoubleDQN double-DQN} algorithm in addition to the setup done by {@link BaseDQNAgentLearnerBuilder}.
 */
public class DoubleDQNBuilder extends BaseDQNAgentLearnerBuilder<DoubleDQNBuilder.Configuration> {


    public DoubleDQNBuilder(Configuration configuration,
                            ITrainableNeuralNet neuralNet,
                            Builder<Environment<Integer>> environmentBuilder,
                            Builder<TransformProcess> transformProcessBuilder,
                            Random rnd) {
        super(configuration, neuralNet, environmentBuilder, transformProcessBuilder, rnd);
    }

    @Override
    protected IUpdateAlgorithm<FeaturesLabels, StateActionRewardState<Integer>> buildUpdateAlgorithm() {
        return new DoubleDQN(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(), configuration.getUpdateAlgorithmConfiguration());
    }

    @EqualsAndHashCode(callSuper = true)
    @SuperBuilder
    @Data
    public static class Configuration extends BaseDQNAgentLearnerBuilder.Configuration {
    }
}
