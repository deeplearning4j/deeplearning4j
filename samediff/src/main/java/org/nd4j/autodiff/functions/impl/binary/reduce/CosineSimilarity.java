package org.nd4j.autodiff.functions.impl.binary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class CosineSimilarity extends AbstractBinaryReduceFunction {

    public CosineSimilarity(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int... dimensions) {
        super(sameDiff, i_v1, i_v2, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return a().cosineSimilarity(larg(), rarg(), dimensions);
    }

    private DifferentialFunction formula() {
        DifferentialFunction numerator = f().mul(larg(),rarg());
        DifferentialFunction denom = f().sqrt(f().mul(f().pow(larg(),2),f().pow(rarg(),2)));
        return f().div(numerator,denom);
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
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
        return formula().diff(i_v1);
    }
}
