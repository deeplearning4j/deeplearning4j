package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class GreaterThan extends  BaseCondition {
    public GreaterThan(Number value) {
        super(value);
    }

    public GreaterThan(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public Boolean apply(Number input) {
        return input.floatValue() > value.floatValue();
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return input.absoluteValue().floatValue() > complexNumber.absoluteValue().floatValue();
    }
}
