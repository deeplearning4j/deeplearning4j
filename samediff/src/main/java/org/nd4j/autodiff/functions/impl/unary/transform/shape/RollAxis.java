package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class RollAxis extends AbstractUnaryFunction<ArrayField> {

   private int axis;


    public RollAxis(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int axis) {
        super(sameDiff,i_v,null,
                OpState.OpType.SHAPE,
                new Object[]{axis});
        this.axis = axis;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().rollAxis(arg().getValue(true),axis);
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
        return "rollAxis";
    }

}
