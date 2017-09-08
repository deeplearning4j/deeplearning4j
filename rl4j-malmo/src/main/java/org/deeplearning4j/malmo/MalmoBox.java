package org.deeplearning4j.malmo;

import java.util.Arrays;

import org.deeplearning4j.rl4j.space.Encodable;

/**
 * Encodable state as a simple value array similar to Gym Box model, but without a JSON constructor
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoBox implements Encodable {
    double[] value;

    /**
     * Construct state from an array of doubles
     * @param value state values
     */
    //TODO: If this constructor was added to "Box", we wouldn't need this class at all.
    public MalmoBox(double... value) {
        this.value = value;
    }

    @Override
    public double[] toArray() {
        return value;
    }

    @Override
    public String toString() {
        return Arrays.toString(value);
    }
}
