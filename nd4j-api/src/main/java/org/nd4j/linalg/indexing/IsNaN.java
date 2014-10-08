package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class IsNaN implements Condition {

    @Override
    public Boolean apply(Number input) {
        return Float.isNaN(input.floatValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Float.isNaN(input.absoluteValue().floatValue());
    }
}
