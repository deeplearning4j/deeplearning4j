package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Tile extends AbstractUnaryFunction {

    private int[] repeat;

    public Tile(SameDiff sameDiff, DifferentialFunction i_v,int[] repeat) {
        super(sameDiff, i_v, new Object[]{repeat});
        this.repeat = repeat;
    }

    public Tile(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int[] repeat) {
        super(sameDiff, i_v, inPlace);
        this.repeat = repeat;
    }

    @Override
    public ArrayField doGetValue() {
        return a().tile(arg().getValue(true),repeat);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        validateDifferentialFunctionsamedoDiff(i_v);
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return "tile";
    }
}
