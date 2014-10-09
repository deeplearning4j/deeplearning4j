package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 10/8/14.
 */
public abstract class BaseCondition implements Condition {
    protected Number value;
    protected IComplexNumber complexNumber;

    public BaseCondition(Number value) {
        this.value = value;
        this.complexNumber = Nd4j.createComplexNumber(value, 0);
    }

    public BaseCondition(IComplexNumber complexNumber) {
        this.complexNumber = complexNumber;
        this.value = complexNumber.absoluteValue();
    }



}
