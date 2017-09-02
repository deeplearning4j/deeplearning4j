package org.nd4j.autodiff.functions.impl.binary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class CosineSimilarity extends AbstractBinaryReduceFunction {

    public CosineSimilarity(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2, int... dimensions) {
        super(sameDiff, i_v1, i_v2, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().cosineSimilarity(larg(), rarg(), dimensions);
    }

    private DifferentialFunction<ArrayField> formula() {
        DifferentialFunction<ArrayField> numerator = larg().mul(rarg());
        DifferentialFunction<ArrayField> denom = sameDiff.getFunctionFactory().sqrt(larg().pow(2).mul(rarg().pow(2)));

        return numerator.div(denom);
    }



    @Override
    public String doGetFormula(List<Variable > variables) {
        return larg().doGetFormula(variables) + " * " + rarg().doGetFormula(variables) + "/" +
                "sqrt(pow(" + larg().doGetFormula(variables) + ", 2) * pow(" + rarg().doGetFormula(variables) + ", 2))";
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity().name();
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        return formula().diff(i_v1);
    }
}
