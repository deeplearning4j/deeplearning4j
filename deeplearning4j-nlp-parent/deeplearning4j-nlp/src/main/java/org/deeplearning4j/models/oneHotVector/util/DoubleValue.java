package org.deeplearning4j.models.oneHotVector.util;

/**
 * A simple wrapper data structure for storing a double
 * as an Object that can be put into lists, maps, etc. and then
 * incremented, decremented, and set.
 *
 * @author Ray Mooney
 */

public class DoubleValue implements Comparable<DoubleValue> {

    public DoubleValue() {
    }

    public DoubleValue(double val) {
        this.value = val;
    }

    /**
     * A numerical value
     */
    protected double value = 0;

    /**
     * Increment and return the new count
     */
    public double increment() {
        return ++value;
    }

    /**
     * Increment by n and return the new count
     */
    public double increment(int n) {
        value = value + n;
        return value;
    }

    /**
     * Increment by n and return the new count
     */
    public double increment(double n) {
        value = value + n;
        return value;
    }

    /**
     * Decrement and return the new count
     */
    public double decrement() {
        return --value;
    }

    /**
     * Decrement by n and return the new count
     */
    public double decrement(int n) {
        value = value - n;
        return value;
    }

    /**
     * Decrement by n and return the new count
     */
    public double decrement(double n) {
        value = value - n;
        return value;
    }

    /**
     * Get the current count
     */
    public double getValue() {
        return value;
    }

    /**
     * Set the current count
     */
    public double setValue(int value) {
        this.value = value;
        return value;
    }

    /**
     * Set the current count
     */
    public double setValue(double value) {
        this.value = value;
        return value;
    }

    /**
     * @return comparison in reverse order
     */
    @Override
    public int compareTo(DoubleValue that) {
        return -(new Double(value).compareTo(new Double(that.value)));
    }
}
