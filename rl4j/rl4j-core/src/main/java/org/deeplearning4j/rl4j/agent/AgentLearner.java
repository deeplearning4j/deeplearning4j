/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.agent;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.behavior.ILearningBehavior;
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

    private final ILearningBehavior<ACTION> learningBehavior;
    private double rewardAtLastExperience;

    /**
     *
     * @param environment The {@link Environment} to be used
     * @param transformProcess The {@link TransformProcess} to be used to transform the raw observations into usable ones.
     * @param policy The {@link IPolicy} to be used
     * @param configuration The configuration for the AgentLearner
     * @param id A user-supplied id to identify the instance.
     * @param learningBehavior The {@link ILearningBehavior} that will be used to supervise the learning.
     */
    public AgentLearner(Environment<ACTION> environment,
                        TransformProcess transformProcess,
                        IPolicy<ACTION> policy,
                        Configuration configuration,
                        String id,
                        @NonNull ILearningBehavior<ACTION> learningBehavior) {
        super(environment, transformProcess, policy, configuration, id);

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
    protected void onBeforeStep() {
        learningBehavior.notifyBeforeStep();
    }

    @SuperBuilder
    @Data
    public static class Configuration extends Agent.Configuration {
    }
}
