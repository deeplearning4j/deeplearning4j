package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;

public class Permute extends AbstractUnaryFunction<ArrayField> {
    protected  int[] dimensions;

    public Permute(SameDiff sameDiff,DifferentialFunction<ArrayField> iX, int... dimensions) {
        super(sameDiff, iX, ArrayUtil.reverseCopy(iX.getValue(true)
                .getInput().getShape()), OpState.OpType.SHAPE, null);
        this.dimensions = dimensions;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().permute(arg().getValue(true),dimensions);
    }

    @Override
    public double getReal() {
        return Math.tan(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return this;
    }



    @Override
    public String functionName() {
        return "permute";
    }

}
