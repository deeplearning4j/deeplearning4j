package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

public class BroadcastDivOp extends BaseBroadcastOp {

    public BroadcastDivOp() {
    }

    public BroadcastDivOp(INDArray x, INDArray y, INDArray z, int...dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastDivOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "broadcastdiv";
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
