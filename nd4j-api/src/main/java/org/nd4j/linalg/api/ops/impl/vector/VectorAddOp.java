package org.nd4j.linalg.api.ops.impl.vector;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseVectorOp;

public class VectorAddOp extends BaseVectorOp {

    public VectorAddOp() {
    }

    public VectorAddOp(INDArray x, INDArray y, INDArray z, int dimension) {
        super(x, y, z, dimension);
    }


    @Override
    public String name() {
        return "vectoradd";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.add(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.add(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.add(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin + other;
    }

    @Override
    public double op(double origin, double other) {
        return origin + other;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }


}
