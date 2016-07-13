package org.deeplearning4j.gym.space;

import org.deeplearning4j.gym.Box;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public class BoxSpace extends ObservationSpace<Box> {


    public BoxSpace(JSONObject n) {

    }

    public String getInfoName(){
        return "Box";
    }

    public Box getValue(JSONObject o, String key) {
        JSONArray arr = o.getJSONArray(key);
        return new Box(arr);
    }

}
