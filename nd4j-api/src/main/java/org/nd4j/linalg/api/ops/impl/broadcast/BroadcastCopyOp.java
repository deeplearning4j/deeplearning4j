package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;
import org.nd4j.linalg.factory.Nd4j;

public class BroadcastCopyOp extends BaseBroadcastOp {

    public BroadcastCopyOp() {
    }

    public BroadcastCopyOp(INDArray x, INDArray y, INDArray z, int...dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastCopyOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String name() {
        return "broadcastcopy";
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
