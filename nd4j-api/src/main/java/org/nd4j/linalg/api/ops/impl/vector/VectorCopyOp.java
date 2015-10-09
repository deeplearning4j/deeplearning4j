package org.nd4j.linalg.api.ops.impl.vector;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseVectorOp;
import org.nd4j.linalg.factory.Nd4j;

public class VectorCopyOp extends BaseVectorOp {

    public VectorCopyOp() {
    }

    public VectorCopyOp(INDArray x, INDArray y, INDArray z, int dimension) {
        super(x, y, z, dimension);
    }

    @Override
    public String name() {
        return "vectorcopy";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return Nd4j.createComplexNumber(other, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return Nd4j.createComplexNumber(other, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return other;
    }

    @Override
    public float op(float origin, float other) {
        return other;
    }

    @Override
    public double op(double origin, double other) {
        return other;
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
