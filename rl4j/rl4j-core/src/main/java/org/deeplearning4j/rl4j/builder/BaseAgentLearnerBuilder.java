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

import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.behavior.ILearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.behavior.LearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.update.IUpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;

import java.util.List;

/**
 * A base {@link IAgentLearner} builder that should be helpful in several common scenarios. <p/>
 * <b>Note</b>: Classes implementing BaseAgentLearnerBuilder should be careful not to re-use a stateful and/or non thread-safe dependency
 * through several calls to build(). In doubt, use a new instance.
 * @param <ACTION> The type of action
 * @param <EXPERIENCE_TYPE> The type of experiences
 * @param <ALGORITHM_RESULT_TYPE> The response type of {@link org.deeplearning4j.rl4j.network.IOutputNeuralNet IOutputNeuralNet}.output()
 * @param <CONFIGURATION_TYPE> The type of the configuration
 */
public abstract class BaseAgentLearnerBuilder<ACTION, EXPERIENCE_TYPE, ALGORITHM_RESULT_TYPE, CONFIGURATION_TYPE extends BaseAgentLearnerBuilder.Configuration<ACTION>> implements Builder<IAgentLearner<ACTION>> {

    protected final CONFIGURATION_TYPE configuration;
    private final Builder<Environment<ACTION>> environmentBuilder;
    private final Builder<TransformProcess> transformProcessBuilder;
    protected final INetworksHandler networks;

    protected int createdAgentLearnerCount;

    public BaseAgentLearnerBuilder(@NonNull CONFIGURATION_TYPE configuration,
                                   @NonNull ITrainableNeuralNet neuralNet,
                                   @NonNull Builder<Environment<ACTION>> environmentBuilder,
                                   @NonNull Builder<TransformProcess> transformProcessBuilder) {
        this.configuration = configuration;
        this.environmentBuilder = environmentBuilder;
        this.transformProcessBuilder = transformProcessBuilder;

        this.networks = configuration.isAsynchronous()
                ? new AsyncNetworkHandler(neuralNet)
                : new SyncNetworkHandler(neuralNet);
    }

    @Getter(AccessLevel.PROTECTED)
    private Environment<ACTION> environment;

    @Getter(AccessLevel.PROTECTED)
    private TransformProcess transformProcess;

    @Getter(AccessLevel.PROTECTED)
    private IPolicy<ACTION> policy;

    @Getter(AccessLevel.PROTECTED)
    private ExperienceHandler<ACTION, EXPERIENCE_TYPE> experienceHandler;

    @Getter(AccessLevel.PROTECTED)
    private IUpdateAlgorithm<ALGORITHM_RESULT_TYPE, EXPERIENCE_TYPE> updateAlgorithm;

    @Getter(AccessLevel.PROTECTED)
    private INeuralNetUpdater<ALGORITHM_RESULT_TYPE> neuralNetUpdater;

    @Getter(AccessLevel.PROTECTED)
    private IUpdateRule<EXPERIENCE_TYPE> updateRule;

    @Getter(AccessLevel.PROTECTED)
    private ILearningBehavior<ACTION> learningBehavior;

    protected abstract IPolicy<ACTION> buildPolicy();
    protected abstract ExperienceHandler<ACTION, EXPERIENCE_TYPE> buildExperienceHandler();
    protected abstract IUpdateAlgorithm<ALGORITHM_RESULT_TYPE, EXPERIENCE_TYPE> buildUpdateAlgorithm();
    protected abstract INeuralNetUpdater<ALGORITHM_RESULT_TYPE> buildNeuralNetUpdater();
    protected IUpdateRule<EXPERIENCE_TYPE> buildUpdateRule() {
        return new UpdateRule<ALGORITHM_RESULT_TYPE, EXPERIENCE_TYPE>(getUpdateAlgorithm(), getNeuralNetUpdater());
    }
    protected ILearningBehavior<ACTION> buildLearningBehavior() {
        return LearningBehavior.<ACTION, EXPERIENCE_TYPE>builder()
                .experienceHandler(getExperienceHandler())
                .updateRule(getUpdateRule())
                .build();
    }

    protected void resetForNewBuild() {
        networks.resetForNewBuild();
        environment = environmentBuilder.build();
        transformProcess = transformProcessBuilder.build();
        policy = buildPolicy();
        experienceHandler = buildExperienceHandler();
        updateAlgorithm = buildUpdateAlgorithm();
        neuralNetUpdater = buildNeuralNetUpdater();
        updateRule = buildUpdateRule();
        learningBehavior = buildLearningBehavior();

        ++createdAgentLearnerCount;
    }

    protected String getThreadId() {
        return "AgentLearner-" + createdAgentLearnerCount;
    }

    protected IAgentLearner<ACTION> buildAgentLearner() {
        AgentLearner<ACTION> result = new AgentLearner(getEnvironment(), getTransformProcess(), getPolicy(), configuration.getAgentLearnerConfiguration(), getThreadId(), getLearningBehavior());
        if(configuration.getAgentLearnerListeners() != null) {
            for (AgentListener<ACTION> listener : configuration.getAgentLearnerListeners()) {
                result.addListener(listener);
            }
        }

        return result;
    }

    /**
     * Build a properly assembled / configured IAgentLearner.
     * @return a {@link IAgentLearner}
     */
    @Override
    public IAgentLearner<ACTION> build() {
        resetForNewBuild();
        return buildAgentLearner();
    }

    @SuperBuilder
    @Data
    public static class Configuration<ACTION> {
        /**
         * The configuration that will be used to build the {@link AgentLearner}
         */
        AgentLearner.Configuration agentLearnerConfiguration;

        /**
         * A list of {@link AgentListener AgentListeners} that will be added to the AgentLearner. (default = null; no listeners)
         */
        List<AgentListener<ACTION>> agentLearnerListeners;

        /**
         * Tell the builder that the AgentLearners will be used in an asynchronous setup
         */
        boolean asynchronous;
    }
}
