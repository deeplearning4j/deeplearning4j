package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import com.google.common.base.Preconditions;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.List;

public class Permute extends AbstractUnaryFunction {
    protected  int[] dimensions;

    public Permute(SameDiff sameDiff,DifferentialFunction iX, int... dimensions) {
        super(sameDiff, iX, ArrayUtil.reverseCopy(iX.getValue(true)
                .getInput().getShape()), OpState.OpType.SHAPE, new Object[]{dimensions});
        this.dimensions = dimensions;
        Preconditions.checkState(dimensions != null,"Dimensions must not be null.");
    }

    @Override
    public ArrayField doGetValue() {
        if(dimensions == null && extraArgs != null) {
            this.dimensions = (int[]) extraArgs[0];
        }

        return sameDiff.getArrayFactory().permute(arg().getValue(true),dimensions);
    }

    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        return Collections.singletonList(this);
    }



    @Override
    public String functionName() {
        return "permute";
    }

}
