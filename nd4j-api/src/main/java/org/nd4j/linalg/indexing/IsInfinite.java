package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.indexing.conditions.Condition;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class IsInfinite implements Condition {

    @Override
    public Boolean apply(Number input) {
        return Float.isInfinite(input.floatValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Float.isInfinite(input.absoluteValue().floatValue());
    }
}
