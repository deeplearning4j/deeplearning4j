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
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncGradientsNeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncSharedNetworksUpdateHandler;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.EpsGreedy;

/**
 * A base {@link IAgentLearner} builder that should be helpful in several common asynchronous scenarios. <p/>
 * <b>Note</b>: Classes implementing BaseAsyncAgentLearnerBuilder should be careful not to re-use a stateful and/or non thread-safe dependency
 * through several calls to build(). In doubt, use a new instance.
 * <p/>
 * This will configure these dependencies:
 * <li>a {@link StateActionExperienceHandler}</li>
 * <li>a {@link AsyncGradientsNeuralNetUpdater gradient neural net updater}</li>
 * @param <CONFIGURATION_TYPE> The type of the configuration
 */
public abstract class BaseAsyncAgentLearnerBuilder<CONFIGURATION_TYPE extends BaseAsyncAgentLearnerBuilder.Configuration> extends BaseAgentLearnerBuilder<Integer, StateActionReward<Integer>, Gradients, CONFIGURATION_TYPE> {

    private final AsyncSharedNetworksUpdateHandler asyncSharedNetworksUpdateHandler;

    public BaseAsyncAgentLearnerBuilder(CONFIGURATION_TYPE configuration,
                                        ITrainableNeuralNet neuralNet,
                                        Builder<Environment<Integer>> environmentBuilder,
                                        Builder<TransformProcess> transformProcessBuilder) {
        super(configuration, neuralNet, environmentBuilder, transformProcessBuilder);

        asyncSharedNetworksUpdateHandler = buildAsyncSharedNetworksUpdateHandler();
    }

    @Override
    protected ExperienceHandler<Integer, StateActionReward<Integer>> buildExperienceHandler() {
        return new StateActionExperienceHandler<Integer>(configuration.getExperienceHandlerConfiguration());
    }

    @Override
    protected INeuralNetUpdater<Gradients> buildNeuralNetUpdater() {
        return new AsyncGradientsNeuralNetUpdater(networks.getThreadCurrentNetwork(), asyncSharedNetworksUpdateHandler);
    }

    protected abstract AsyncSharedNetworksUpdateHandler buildAsyncSharedNetworksUpdateHandler();

    @EqualsAndHashCode(callSuper = true)
    @SuperBuilder
    @Data
    public static class Configuration extends BaseAgentLearnerBuilder.Configuration<Integer> {
        EpsGreedy.Configuration policyConfiguration;
        NeuralNetUpdaterConfiguration neuralNetUpdaterConfiguration;
        StateActionExperienceHandler.Configuration experienceHandlerConfiguration;
    }
}
