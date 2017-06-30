package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

public class BroadcastLessThan extends BaseBroadcastOp {

    public BroadcastLessThan() {}

    public BroadcastLessThan(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }


    @Override
    public int opNum() {
        return 10;
    }

    @Override
    public String name() {
        return "broadcast_lessthan";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float op(float origin, float other) {
        return origin < other ? 1.0f : 0.0f;
    }

    @Override
    public double op(double origin, double other) {
        return origin < other ? 1.0f : 0.0f;
    }

    @Override
    public double op(double origin) {
        return 1;
    }

    @Override
    public float op(float origin) {
        return 1;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }


}
