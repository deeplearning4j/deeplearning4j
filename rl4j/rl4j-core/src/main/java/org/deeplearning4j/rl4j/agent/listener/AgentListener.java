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
package org.deeplearning4j.rl4j.agent.listener;

import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;

/**
 * The base definition of all {@link Agent} event listeners
 */
public interface AgentListener<ACTION> {
    enum ListenerResponse {
        /**
         * Tell the {@link Agent} to continue calling the listeners and the processing.
         */
        CONTINUE,

        /**
         * Tell the {@link Agent} to interrupt calling the listeners and stop the processing.
         */
        STOP,
    }

    /**
     * Called when a new episode is about to start.
     * @param agent The agent that generated the event
     *
     * @return A {@link ListenerResponse}.
     */
    AgentListener.ListenerResponse onBeforeEpisode(Agent agent);

    /**
     * Called when a step is about to be taken.
     *
     * @param agent The agent that generated the event
     * @param observation The observation before the action is taken
     * @param action The action that will be performed
     *
     * @return A {@link ListenerResponse}.
     */
    AgentListener.ListenerResponse onBeforeStep(Agent agent, Observation observation, ACTION action);

    /**
     * Called after a step has been taken.
     *
     * @param agent The agent that generated the event
     * @param stepResult The {@link StepResult} result of the step.
     *
     * @return A {@link ListenerResponse}.
     */
    AgentListener.ListenerResponse onAfterStep(Agent agent, StepResult stepResult);

    /**
     * Called after the episode has ended.
     *
     * @param agent The agent that generated the event
     *
     */
    void onAfterEpisode(Agent agent);
}
