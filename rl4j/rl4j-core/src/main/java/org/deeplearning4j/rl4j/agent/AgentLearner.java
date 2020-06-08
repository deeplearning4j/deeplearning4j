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
package org.deeplearning4j.rl4j.agent;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.ILearningBehavior;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;

/**
 * The ActionLearner is an {@link Agent} that delegate the learning to a {@link ILearningBehavior}.
 * @param <ACTION> The type of the action
 */
public class AgentLearner<ACTION> extends Agent<ACTION> implements IAgentLearner<ACTION> {

    @Getter
    private int totalStepCount = 0;

    private final ILearningBehavior<ACTION> learningBehavior;
    private double rewardAtLastExperience;

    /**
     *
     * @param environment The {@link Environment} to be used
     * @param transformProcess The {@link TransformProcess} to be used to transform the raw observations into usable ones.
     * @param policy The {@link IPolicy} to be used
     * @param maxEpisodeSteps The maximum number of steps an episode can have before being interrupted. Use null to have no max.
     * @param id A user-supplied id to identify the instance.
     * @param learningBehavior The {@link ILearningBehavior} that will be used to supervise the learning.
     */
    public AgentLearner(Environment<ACTION> environment, TransformProcess transformProcess, IPolicy<ACTION> policy, Integer maxEpisodeSteps, String id, @NonNull ILearningBehavior<ACTION> learningBehavior) {
        super(environment, transformProcess, policy, maxEpisodeSteps, id);

        this.learningBehavior = learningBehavior;
    }

    @Override
    protected void reset() {
        super.reset();

        rewardAtLastExperience = 0;
    }

    @Override
    protected void onBeforeEpisode() {
        super.onBeforeEpisode();

        learningBehavior.handleEpisodeStart();
    }

    @Override
    protected void onAfterAction(Observation observationBeforeAction, ACTION action, StepResult stepResult) {
        if(!observationBeforeAction.isSkipped()) {
            double rewardSinceLastExperience = getReward() - rewardAtLastExperience;
            learningBehavior.handleNewExperience(observationBeforeAction, action, rewardSinceLastExperience, stepResult.isTerminal());

            rewardAtLastExperience = getReward();
        }
    }

    @Override
    protected void onAfterEpisode() {
        learningBehavior.handleEpisodeEnd(getObservation());
    }

    @Override
    protected void incrementEpisodeStepCount() {
        super.incrementEpisodeStepCount();
        ++totalStepCount;
    }

    // FIXME: parent is still visible
    public static <ACTION> AgentLearner.Builder<ACTION, AgentLearner<ACTION>> builder(Environment<ACTION> environment,
                                                   TransformProcess transformProcess,
                                                   IPolicy<ACTION> policy,
                                                   ILearningBehavior<ACTION> learningBehavior) {
        return new AgentLearner.Builder<ACTION, AgentLearner<ACTION>>(environment, transformProcess, policy, learningBehavior);
    }

    public static class Builder<ACTION, AGENT_TYPE extends AgentLearner<ACTION>> extends Agent.Builder<ACTION, AGENT_TYPE> {

        private final ILearningBehavior<ACTION> learningBehavior;

        public Builder(@NonNull Environment<ACTION> environment,
                       @NonNull TransformProcess transformProcess,
                       @NonNull IPolicy<ACTION> policy,
                       @NonNull ILearningBehavior<ACTION> learningBehavior) {
            super(environment, transformProcess, policy);

            this.learningBehavior = learningBehavior;
        }

        @Override
        public AGENT_TYPE build() {
            return (AGENT_TYPE)new AgentLearner<ACTION>(environment, transformProcess, policy, maxEpisodeSteps, id, learningBehavior);
        }
    }
}
