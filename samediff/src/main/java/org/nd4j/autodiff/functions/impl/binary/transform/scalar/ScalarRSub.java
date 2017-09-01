package org.nd4j.autodiff.functions.impl.binary.transform.scalar;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;

public class ScalarRSub extends AbstractBinaryFunction<ArrayField> {
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
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseSubtraction().name();
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        return null;
    }
}
