package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class LessThan extends BaseCondition {

    public LessThan(Number value) {
        super(value);
    }

    public LessThan(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public Boolean apply(Number input) {
        return input.doubleValue() < value.doubleValue();
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return input.absoluteValue().doubleValue() < complexNumber.absoluteValue().doubleValue();
    }
}
