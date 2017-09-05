package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Broadcast extends AbstractUnaryFunction {

    public Broadcast(SameDiff sameDiff, DifferentialFunction i_v, int[] shape) {
        super(sameDiff, i_v, shape,OpState.OpType.SHAPE,new Object[]{shape});
        this.shape = shape;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().broadcast(arg().getValue(true),shape);
    }


    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String functionName() {
        return "broadcast";
    }

}
