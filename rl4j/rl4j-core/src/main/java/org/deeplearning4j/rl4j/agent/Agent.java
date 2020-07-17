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

import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.agent.listener.AgentListenerList;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.nd4j.common.base.Preconditions;

import java.util.Map;

/**
 * An agent implementation. The Agent will use a {@link IPolicy} to interact with an {@link Environment} and receive
 * a reward.
 *
 * @param <ACTION> The type of action
 */
public class Agent<ACTION> implements IAgent<ACTION> {
    @Getter
    private final String id;

    @Getter
    private final Environment<ACTION> environment;

    @Getter
    private final IPolicy<ACTION> policy;

    private final TransformProcess transformProcess;

    protected final AgentListenerList<ACTION> listeners;

    private final Integer maxEpisodeSteps;

    @Getter(AccessLevel.PROTECTED)
    private Observation observation;

    @Getter(AccessLevel.PROTECTED)
    private ACTION lastAction;

    @Getter
    private int episodeStepCount;

    @Getter
    private double reward;

    protected boolean canContinue;

    /**
     * @param environment The {@link Environment} to be used
     * @param transformProcess The {@link TransformProcess} to be used to transform the raw observations into usable ones.
     * @param policy The {@link IPolicy} to be used
     * @param configuration The configuration for the agent
     * @param id A user-supplied id to identify the instance.
     */
    public Agent(@NonNull Environment<ACTION> environment,
                 @NonNull TransformProcess transformProcess,
                 @NonNull IPolicy<ACTION> policy,
                 @NonNull Configuration configuration,
                 String id) {
        Preconditions.checkArgument(configuration.getMaxEpisodeSteps() == null || configuration.getMaxEpisodeSteps() > 0, "Configuration: maxEpisodeSteps must be null (no maximum) or greater than 0, got", configuration.getMaxEpisodeSteps());

        this.environment = environment;
        this.transformProcess = transformProcess;
        this.policy = policy;
        this.maxEpisodeSteps = configuration.getMaxEpisodeSteps();
        this.id = id;

        listeners = buildListenerList();
    }

    protected AgentListenerList<ACTION> buildListenerList() {
        return new AgentListenerList<ACTION>();
    }

    /**
     * Add a {@link AgentListener} that will be notified when agent events happens
     * @param listener
     */
    public void addListener(AgentListener listener) {
        listeners.add(listener);
    }

    /**
     * This will run a single episode
     */
    public void run() {
        runEpisode();
    }

    protected void onBeforeEpisode() {
        // Do Nothing
    }

    protected void onAfterEpisode() {
        // Do Nothing
    }

    protected void runEpisode() {
        reset();
        onBeforeEpisode();

        canContinue = listeners.notifyBeforeEpisode(this);

        while (canContinue && !environment.isEpisodeFinished() && (maxEpisodeSteps == null || episodeStepCount < maxEpisodeSteps)) {
            performStep();
        }

        if(!canContinue) {
            return;
        }

        onAfterEpisode();
        listeners.notifyAfterEpisode(this);
    }

    protected void reset() {
        resetEnvironment();
        resetPolicy();
        reward = 0;
        lastAction = getInitialAction();
        canContinue = true;
    }

    protected void resetEnvironment() {
        episodeStepCount = 0;
        Map<String, Object> channelsData = environment.reset();
        this.observation = transformProcess.transform(channelsData, episodeStepCount, false);
    }

    protected void resetPolicy() {
        policy.reset();
    }

    protected ACTION getInitialAction() {
        return environment.getSchema().getActionSchema().getNoOp();
    }

    protected void performStep() {

        onBeforeStep();

        ACTION action = decideAction(observation);

        canContinue = listeners.notifyBeforeStep(this, observation, action);
        if(!canContinue) {
            return;
        }

        StepResult stepResult = act(action);

        onAfterStep(stepResult);

        canContinue = listeners.notifyAfterStep(this, stepResult);
        if(!canContinue) {
            return;
        }

        incrementEpisodeStepCount();
    }

    protected void incrementEpisodeStepCount() {
        ++episodeStepCount;
    }

    protected ACTION decideAction(Observation observation) {
        if (!observation.isSkipped()) {
            lastAction = policy.nextAction(observation);
        }

        return lastAction;
    }

    protected StepResult act(ACTION action) {
        Observation observationBeforeAction = observation;

        StepResult stepResult = environment.step(action);
        observation = convertChannelDataToObservation(stepResult, episodeStepCount + 1);
        reward += computeReward(stepResult);

        onAfterAction(observationBeforeAction, action, stepResult);

        return stepResult;
    }

    protected Observation convertChannelDataToObservation(StepResult stepResult, int episodeStepNumberOfObs) {
        return transformProcess.transform(stepResult.getChannelsData(), episodeStepNumberOfObs, stepResult.isTerminal());
    }

    protected double computeReward(StepResult stepResult) {
        return stepResult.getReward();
    }

    protected void onAfterAction(Observation observationBeforeAction, ACTION action, StepResult stepResult) {
        // Do Nothing
    }

    protected void onAfterStep(StepResult stepResult) {
        // Do Nothing
    }

    protected void onBeforeStep() {
        // Do Nothing
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The maximum number of steps an episode can have before being interrupted. Use null to have no max.
         */
        @lombok.Builder.Default
        Integer maxEpisodeSteps = null; // Default, no max
    }
}