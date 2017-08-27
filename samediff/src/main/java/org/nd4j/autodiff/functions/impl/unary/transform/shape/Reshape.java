package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Collections;
import java.util.List;

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
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(this);
    }

    @Override
    public String functionName() {
        return "reshape";
    }

}
