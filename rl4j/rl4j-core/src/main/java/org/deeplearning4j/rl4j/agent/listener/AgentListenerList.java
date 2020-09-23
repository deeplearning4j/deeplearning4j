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

import java.util.ArrayList;
import java.util.List;

/**
 * A class that manages a list of {@link AgentListener AgentListeners} listening to an {@link Agent}.
 * @param <ACTION>
 */
public class AgentListenerList<ACTION> {
    protected final List<AgentListener<ACTION>> listeners = new ArrayList<>();

    /**
     * Add a listener at the end of the list
     * @param listener The listener to be added
     */
    public void add(AgentListener<ACTION> listener) {
        listeners.add(listener);
    }

    /**
     * This method will notify all listeners that an episode is about to start. If a listener returns
     * {@link AgentListener.ListenerResponse STOP}, any following listener is skipped.
     *
     * @param agent The agent that generated the event.
     * @return False if the processing should be stopped
     */
    public boolean notifyBeforeEpisode(Agent<ACTION> agent) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onBeforeEpisode(agent) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     *
     * @param agent The agent that generated the event.
     * @param observation The observation before the action is taken
     * @param action The action that will be performed
     * @return False if the processing should be stopped
     */
    public boolean notifyBeforeStep(Agent<ACTION> agent, Observation observation, ACTION action) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onBeforeStep(agent, observation, action) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     *
     * @param agent The agent that generated the event.
     * @param stepResult The {@link StepResult} result of the step.
     * @return False if the processing should be stopped
     */
    public boolean notifyAfterStep(Agent<ACTION> agent, StepResult stepResult) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onAfterStep(agent, stepResult) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * This method will notify all listeners that an episode has finished.
     *
     * @param agent The agent that generated the event.
     */
    public void notifyAfterEpisode(Agent<ACTION> agent) {
        for (AgentListener<ACTION> listener : listeners) {
            listener.onAfterEpisode(agent);
        }
    }

}
