package org.nd4j.linalg.indexing.conditions;

import org.apache.commons.math3.util.FastMath;

/**Boolean condition on absolute value: abs(x) > value
 */
public class AbsValueGreaterThan extends BaseCondition {
    public AbsValueGreaterThan(Number value){
        super(value);
    }

    @Override
    public Boolean apply(Number input) {
        return FastMath.abs(input.doubleValue()) > value.doubleValue();
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return input.absoluteValue().doubleValue() > value.doubleValue();
    }
}
