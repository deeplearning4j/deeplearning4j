package org.nd4j.autodiff.functions.impl.binary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class ManhattanDistance  extends AbstractBinaryReduceFunction {

    public ManhattanDistance(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int... dimensions) {
        super(sameDiff, i_v1, i_v2, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().manhattanDistance(larg(),rarg(),dimensions);
    }


    @Override
    public String doGetFormula(List<Variable > variables) {
        return null;
    }

    @Override
    public String functionName() {
        return "manhattanDistance";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        throw new UnsupportedOperationException();
    }
}
