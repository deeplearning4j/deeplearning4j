package org.deeplearning4j.gym.space;

import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public interface ObservationSpace<T> {

    T getValue(JSONObject o, String key);

}
