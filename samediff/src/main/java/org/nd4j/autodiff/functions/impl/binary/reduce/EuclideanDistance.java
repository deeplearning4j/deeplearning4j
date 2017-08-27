package org.nd4j.autodiff.functions.impl.binary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class EuclideanDistance  extends AbstractBinaryReduceFunction<ArrayField> {

    public EuclideanDistance(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2, int... dimensions) {
        super(sameDiff, i_v1, i_v2, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().euclideanDistance(larg(),rarg(),dimensions);
    }



    @Override
    public String doGetFormula(List<Variable<ArrayField> > variables) {
        return null;
    }

    @Override
    public String functionName() {
        return "euclideanDistance";
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        throw new UnsupportedOperationException();
    }
}
