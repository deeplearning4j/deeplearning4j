package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;

import java.util.ArrayList;
import java.util.List;

public class Mul extends AbstractBinaryFunction {
    public Mul(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return larg().getValue(true).mul(rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction g = sameDiff.setupFunction(i_v.get(0));
        DifferentialFunction gradWrtX = f().mul(g,rarg());
        DifferentialFunction gradWrtY = f().mul(g,larg());
        List<DifferentialFunction> ret = new ArrayList<>();
        larg().setGradient(gradWrtX);
        rarg().setGradient(gradWrtY);
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
    }


    @Override
    public String functionName() {
        return new MulOp().name();
    }
}
