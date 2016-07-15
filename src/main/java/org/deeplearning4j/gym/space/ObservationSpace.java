package org.deeplearning4j.gym.space;

import org.json.JSONObject;

/**
 * @author rubenfiszel on 7/8/16.
 *
 * Should contain contextual information about the environment from which Observations are observed and must know how to build an Observation from json.
 *
 * @param <O> the type of Observation
 */
public interface ObservationSpace<O> {

    /**
     * create an Observation from json. Asking for the key and dict instead of JSONObject directly enables to handle Observation encoded as JSONArray (for Box by example).
     *
     * @param jsonObject the jsonObject in which it is contained
     * @param key        the key corresponding to the observation in the parent dict
     * @return
     */
    O getValue(JSONObject jsonObject, String key);

}
