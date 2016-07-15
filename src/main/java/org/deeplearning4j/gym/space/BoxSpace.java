package org.deeplearning4j.gym.space;

import lombok.Value;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * @author rubenfiszel on 7/8/16.
 *
 * The space of {@link Box} observation
 */
@Value
public class BoxSpace implements LowDimensionalSpace<Box> {

    int[] shape;
    double[] low;
    double[] high;

    public BoxSpace(JSONObject jsonObject) {

        JSONArray arr = jsonObject.getJSONArray("shape");
        int lg = arr.length();

        shape = new int[lg];
        for (int i = 0; i < lg; i++) {
            this.shape[i] = arr.getInt(i);
        }

        arr = jsonObject.getJSONArray("low");
        lg = arr.length();

        low = new double[lg];
        for (int i = 0; i < lg; i++) {
            low[i] = arr.getDouble(i);
        }

        arr = jsonObject.getJSONArray("high");
        lg = arr.length();

        high = new double[lg];
        for (int i = 0; i < lg; i++) {
            high[i] = arr.getInt(i);
        }


    }

    public Box getValue(JSONObject o, String key) {
        JSONArray arr = o.getJSONArray(key);
        return new Box(arr);
    }

}
