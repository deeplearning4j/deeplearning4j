package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Broadcast extends AbstractUnaryFunction<ArrayField> {

    public Broadcast(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] shape) {
        super(sameDiff, i_v, shape,OpState.OpType.SHAPE,new Object[]{shape});
        this.shape = shape;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().broadcast(arg().getValue(true),shape);
    }

    @Override
    public double getReal() {
        return Math.floor(arg().getReal());
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String functionName() {
        return "broadcast";
    }

}
