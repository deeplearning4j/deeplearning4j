package org.deeplearning4j.gym.space;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public class BoxSpace implements LowDimensionalSpace<Box> {

    JSONObject n;

    public BoxSpace(JSONObject n) {
        this.n = n;
    }

    public Box getValue(JSONObject o, String key) {
        JSONArray arr = o.getJSONArray(key);
        return new Box(arr);
    }

}
