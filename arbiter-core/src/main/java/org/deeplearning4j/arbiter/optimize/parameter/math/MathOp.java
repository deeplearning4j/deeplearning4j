package org.deeplearning4j.arbiter.optimize.parameter.math;

import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.List;

public class MathOp<T extends Number> extends AbstractParameterSpace<T> {

    private ParameterSpace<T> parameterSpace;
    private Op op;
    private T scalar;

    public MathOp(ParameterSpace<T> parameterSpace, Op op, T scalar){
        this.parameterSpace = parameterSpace;
        this.op = op;
        this.scalar = scalar;
    }

    @Override
    public T getValue(double[] parameterValues) {
        T u = parameterSpace.getValue(parameterValues);
        return op.doOp(u, scalar);
    }

    @Override
    public int numParameters() {
        return parameterSpace.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return parameterSpace.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        parameterSpace.setIndices(indices);
    }
}
