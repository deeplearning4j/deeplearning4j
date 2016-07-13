package org.deeplearning4j.gym.space;

import org.json.JSONArray;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public class Box extends LowDimensional{

    double[] array;

    public Box(JSONArray arr) {

        int lg = arr.length();
        this.array = new double[lg];

        for(int i = 0; i < lg; i++){
            this.array[i] = arr.getDouble(i);
        }
    }

    public double[] toArray() {
        return array;
    }
}
