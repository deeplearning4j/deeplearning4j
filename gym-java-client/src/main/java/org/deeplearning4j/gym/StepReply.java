package org.deeplearning4j.gym;

import lombok.Value;
import org.json.JSONObject;

/**
 * @param <T> type of observation
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/6/16.
 *
 *  StepReply is the container for the data returned after each step(action).
 */
@Value
public class StepReply<T> {

    T observation;
    double reward;
    boolean done;
    JSONObject info;

}
