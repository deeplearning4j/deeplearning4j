package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.RSubOp;

import java.util.ArrayList;
import java.util.List;

public class RSub extends AbstractBinaryFunction {

    public RSub(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2,boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace, OpState.OpType.TRANSFORM);
    }


    public RSub(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }


    @Override
    public ArrayField doGetValue() {
        if(!isInPlace())
            return larg().getValue(true).rsub(rarg().getValue(true));
        else
            return larg().getValue(true).rsubi(rarg().getValue(true));
    }

    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction gradWrtX = f().div(i_v.get(0),rarg());
        DifferentialFunction gradWrtY = f().mul(f().neg(gradWrtX),f().div(larg(),rarg()));
        larg().setGradient(gradWrtX);
        rarg().setGradient(gradWrtY);
        List<DifferentialFunction> ret = new ArrayList<>(2);
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
    }



    @Override
    public String functionName() {
        return new RSubOp().name();
    }
}
