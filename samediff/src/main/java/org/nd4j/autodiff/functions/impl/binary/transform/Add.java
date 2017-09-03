package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.Differential;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;

import java.util.ArrayList;
import java.util.List;

public class Add extends AbstractBinaryFunction {

    public Add(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return larg().getValue(true).add(rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction g = sameDiff.setupFunction(i_v.get(0));
        larg().setGradient(g);
        rarg().setGradient(g);
        List<DifferentialFunction> ret = new ArrayList<>();
        for(int i = 0; i < 2; i++)
            ret.add(g);

        return ret;
    }



    @Override
    public String functionName() {
        return new AddOp().name();
    }
}
