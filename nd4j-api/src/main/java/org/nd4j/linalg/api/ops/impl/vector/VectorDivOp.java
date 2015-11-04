package org.nd4j.linalg.api.ops.impl.vector;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseVectorOp;

public class VectorDivOp extends BaseVectorOp {

    public VectorDivOp() {
    }

    public VectorDivOp(INDArray x, INDArray y, INDArray z, int dimension) {
        super(x, y, z, dimension);
    }

    @Override
    public String name() {
        return "vectordiv";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.div(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.div(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.div(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin / other;
    }

    @Override
    public double op(double origin, double other) {
        return origin / other;
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
