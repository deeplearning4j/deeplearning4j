package org.nd4j.autodiff.functions.impl.binary.transform.scalar;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractScalarFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class ScalarSet extends AbstractScalarFunction<ArrayField> {
    public ScalarSet() {
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] shape, Object[] extraArgs) {
        super(sameDiff, i_v, shape, extraArgs);
    }

    public ScalarSet(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    public ArrayField doGetValue() {
        return null;
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarSet().name();
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        return null;
    }
}
