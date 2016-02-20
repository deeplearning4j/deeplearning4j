package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

public class BroadcastMulOp extends BaseBroadcastOp {

    public BroadcastMulOp() {
    }

    public BroadcastMulOp(INDArray x, INDArray y, INDArray z, int...dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastMulOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String name() {
        return "broadcastmul";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.mul(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin * other;
    }

    @Override
    public double op(double origin, double other) {
        return origin * other;
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
