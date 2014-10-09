package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class LessThanOrEqual extends BaseCondition {

    public LessThanOrEqual(Number value) {
        super(value);
    }

    public LessThanOrEqual(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public Boolean apply(Number input) {
        return input.floatValue() <= value.floatValue();
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return input.absoluteValue().floatValue() <= complexNumber.absoluteValue().floatValue();
    }
}
