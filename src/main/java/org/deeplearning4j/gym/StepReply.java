package org.deeplearning4j.gym;

import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/6/16.
 */
public class StepReply<T> {

    private T observation;
    private double reward;
    private boolean done;
    private JSONObject info;

    public StepReply(T observation, double reward, boolean done, JSONObject info) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }

    public T getObservation() {
        return observation;
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

    public JSONObject getInfo() {
        return info;
    }
}
