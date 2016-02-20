package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

public class BroadcastAddOp extends BaseBroadcastOp {

    public BroadcastAddOp() {
    }

    public BroadcastAddOp(INDArray x, INDArray y, INDArray z, int...dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastAddOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "broadcastadd";
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
