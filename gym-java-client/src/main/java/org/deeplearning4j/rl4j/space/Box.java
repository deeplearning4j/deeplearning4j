package org.deeplearning4j.rl4j.space;

import org.json.JSONArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * A Box observation
 *
 * @see <a href="https://gym.openai.com/envs#box2d">https://gym.openai.com/envs#box2d</a>
 */
public class Box implements Encodable {

    private final double[] array;

    public Box(JSONArray arr) {

        int lg = arr.length();
        this.array = new double[lg];

        for (int i = 0; i < lg; i++) {
            this.array[i] = arr.getDouble(i);
        }
    }

    public double[] toArray() {
        return array;
    }
}
