package org.nd4j.autodiff.functions.impl.binary.transform.scalar.comparison;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractScalarFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class ScalarGreatherThanOrEqualTo extends AbstractScalarFunction {
    public ScalarGreatherThanOrEqualTo() {
    }

    public ScalarGreatherThanOrEqualTo(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, Object[] extraArgs) {
        super(sameDiff, i_v, shape, extraArgs);
    }

    public ScalarGreatherThanOrEqualTo(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
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
        return new  org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual().name();
    }

    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
        return null;
    }
}
