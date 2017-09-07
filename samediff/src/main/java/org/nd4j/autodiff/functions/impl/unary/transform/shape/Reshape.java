package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import com.google.common.base.Preconditions;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.List;

public class Reshape extends AbstractUnaryFunction {
    public Reshape(SameDiff sameDiff, DifferentialFunction i_v,int[] shape) {
        super(sameDiff,i_v, Shape.resolveNegativeShapeIfNeccessary(shape,i_v.getResultShape()),
                OpState.OpType.SHAPE,
                new Object[]{Shape.resolveNegativeShapeIfNeccessary(shape,i_v.getResultShape())});
        Preconditions.checkState(ArrayUtil.prod(i_v.getResultShape()) == ArrayUtil.prod(Shape.resolveNegativeShapeIfNeccessary(shape,i_v.getResultShape())),"Invalid shape. Shape must be of equal length.");
    }

    @Override
    public ArrayField doGetValue() {
        return a().reshape(arg().getValue(true),shape);
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = this;
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }

    @Override
    public String functionName() {
        return "reshape";
    }

}
