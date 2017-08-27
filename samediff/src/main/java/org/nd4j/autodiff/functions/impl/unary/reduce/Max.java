package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Max extends AbstractReduceUnaryFunction<ArrayField> {

    public Max(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }



    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().max(arg().doGetValue(),dimensions);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.Max().name();
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        List<DifferentialFunction<ArrayField>> ret = new ArrayList<>(1);
        ret.add(sameDiff.getFunctionFactory().doGradChoose(this,i_v1.get(0),dimensions));
        return ret;
    }
}
