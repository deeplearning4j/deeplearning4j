package org.deeplearning4j.rl4j.agent.listener;

import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;

import java.util.ArrayList;
import java.util.List;

public class AgentListenerList<ACTION> {
    protected final List<AgentListener<ACTION>> listeners = new ArrayList<>();

    /**
     * Add a listener at the end of the list
     * @param listener The listener to be added
     */
    public void add(AgentListener<ACTION> listener) {
        listeners.add(listener);
    }

    public boolean notifyBeforeEpisode(Agent<ACTION> agent) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onBeforeEpisode(agent) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    public boolean notifyBeforeStep(Agent<ACTION> agent, Observation observation, ACTION action) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onBeforeStep(agent, observation, action) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    public boolean notifyAfterStep(Agent<ACTION> agent, StepResult stepResult) {
        for (AgentListener<ACTION> listener : listeners) {
            if (listener.onAfterStep(agent, stepResult) == AgentListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }
}
