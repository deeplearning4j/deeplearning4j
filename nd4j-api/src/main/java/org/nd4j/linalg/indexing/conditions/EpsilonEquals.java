package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class EpsilonEquals extends BaseCondition {
    public EpsilonEquals(Number value) {
        super(value);
    }

    public EpsilonEquals(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public Boolean apply(Number input) {
        return Math.abs(input.floatValue() - value.floatValue()) < Nd4j.EPS_THRESHOLD;
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Math.abs(input.absoluteValue().floatValue() - input.absoluteValue().floatValue()) < Nd4j.EPS_THRESHOLD;

    }
}
