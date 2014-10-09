package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Not implements Condition {

    private Condition opposite;

    public Not(Condition condition) {
        this.opposite = condition;
    }

    @Override
    public Boolean apply(Number input) {
        return !opposite.apply(input);
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return !opposite.apply(input);
    }
}
