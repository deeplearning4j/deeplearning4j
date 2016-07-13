package org.deeplearning4j.gym.space;

import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/8/16.
 */
abstract public class ObservationSpace<T> {

    abstract String getInfoName();

    abstract public T getValue(JSONObject o, String key);

}
