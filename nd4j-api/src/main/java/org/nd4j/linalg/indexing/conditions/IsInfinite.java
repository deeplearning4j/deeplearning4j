package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Returns true when the given number is infinite
 * @author Adam Gibson
 */
public class IsInfinite implements Condition {
    @Override
    public Boolean apply(Number input) {
        return Double.isInfinite(input.doubleValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Double.isInfinite(input.absoluteValue().doubleValue());
    }
}
