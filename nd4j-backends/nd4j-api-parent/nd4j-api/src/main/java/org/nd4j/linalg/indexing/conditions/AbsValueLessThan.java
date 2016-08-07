package org.nd4j.linalg.indexing.conditions;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**Boolean condition on absolute value: abs(x) < value
 */
public class AbsValueLessThan extends BaseCondition {
    public AbsValueLessThan(Number value){
        super(value);
    }

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return 6;
    }

    @Override
    public Boolean apply(Number input) {
        return FastMath.abs(input.doubleValue()) < value.doubleValue();
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return input.absoluteValue().doubleValue() < value.doubleValue();
    }
}
