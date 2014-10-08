package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class IsInfinite extends BaseCondition {
    public IsInfinite(Number value) {
        super(value);
    }

    public IsInfinite(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public Boolean apply(Number input) {
        return Float.isInfinite(input.floatValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Float.isInfinite(input.absoluteValue().floatValue());
    }
}
