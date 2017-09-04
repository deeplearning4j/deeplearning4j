package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.RDivOp;

import java.util.ArrayList;
import java.util.List;

public class RDiv extends AbstractBinaryFunction {
    public RDiv(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return larg().getValue(true).rdiv(rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction gradWrtX = f().div(i_v.get(0),larg());
        DifferentialFunction gradWrtY = f().mul(f().neg(gradWrtX),f().div(rarg(),larg()));
        List<DifferentialFunction> ret = new ArrayList<>(2);
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        larg().setGradient(gradWrtX);
        rarg().setGradient(gradWrtY);
        return ret;
    }



    @Override
    public String functionName() {
        return new RDivOp().name();
    }
}
