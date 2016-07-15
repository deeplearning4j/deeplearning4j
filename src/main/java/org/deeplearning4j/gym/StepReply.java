package org.deeplearning4j.gym;


import lombok.Value;
import org.json.JSONObject;
/**
 * Created by rubenfiszel on 7/6/16.
 */

@Value
public class StepReply<T> {

    T observation;
    double reward;
    boolean done;
    JSONObject info;

}
