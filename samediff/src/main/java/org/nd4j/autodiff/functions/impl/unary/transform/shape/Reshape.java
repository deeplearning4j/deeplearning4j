package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.Shape;

public class Reshape extends AbstractUnaryFunction<ArrayField> {
    public Reshape(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v,int[] shape) {
        super(sameDiff,i_v, Shape.resolveNegativeShapeIfNeccessary(shape),
                OpState.OpType.SHAPE,
                new Object[]{Shape.resolveNegativeShapeIfNeccessary(shape)});
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().reshape(arg().getValue(true),shape);
    }

    @Override
    public double getReal() {
        return Math.floor(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return this;
    }

    @Override
    public String functionName() {
        return "reshape";
    }

}
