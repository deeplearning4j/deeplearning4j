package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Static class for conditions
 * @author Adam Gibson
 */
public class Conditions {


    public static Condition isInfinite() {
        return new IsInfinite();
    }



    public static Condition isNan() {
        return new IsNaN();
    }

    public static Condition epsEquals(IComplexNumber value) {
        return new EqualsCondition(value);
    }
    public static Condition epsEquals(Number value) {
        return new EqualsCondition(value);
    }

    public static Condition equals(IComplexNumber value) {
        return new EqualsCondition(value);
    }
    public static Condition equals(Number value) {
        return new EqualsCondition(value);
    }

    public static Condition greaterThan(IComplexNumber value) {
        return new GreaterThan(value);
    }
    public static Condition greaterThan(Number value) {
        return new GreaterThan(value);
    }
    public static Condition lessThan(IComplexNumber value) {
        return new GreaterThan(value);
    }
    public static Condition lessThan(Number value) {
        return new LessThan(value);
    }

    public static Condition lessThanOrEqual(IComplexNumber value) {
        return new LessThanOrEqual(value);
    }
    public static Condition lessThanOrEqual(Number value) {
        return new LessThanOrEqual(value);
    }
    public static Condition greaterThanOrEqual(IComplexNumber value) {
        return new GreaterThanOrEqual(value);
    }
    public static Condition greaterThanOEqual(Number value) {
        return new GreaterThanOrEqual(value);
    }

}
