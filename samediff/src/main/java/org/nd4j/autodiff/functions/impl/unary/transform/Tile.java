package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class Tile extends AbstractUnaryFunction<ArrayField> {

    private int[] repeat;

    public Tile(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v,int[] repeat) {
        super(sameDiff, i_v, new Object[]{repeat});
        this.repeat = repeat;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().tile(arg().getValue(true),repeat);
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return "tile";
    }
}
