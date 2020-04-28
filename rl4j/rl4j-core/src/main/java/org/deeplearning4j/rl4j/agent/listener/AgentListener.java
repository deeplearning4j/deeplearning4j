package org.deeplearning4j.rl4j.agent.listener;

import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;

public interface AgentListener<ACTION> {
    enum ListenerResponse {
        /**
         * Tell the learning process to continue calling the listeners and the training.
         */
        CONTINUE,

        /**
         * Tell the learning process to stop calling the listeners and terminate the training.
         */
        STOP,
    }

    AgentListener.ListenerResponse onBeforeEpisode(Agent agent);
    AgentListener.ListenerResponse onBeforeStep(Agent agent, Observation observation, ACTION action);
    AgentListener.ListenerResponse onAfterStep(Agent agent, StepResult stepResult);
}
