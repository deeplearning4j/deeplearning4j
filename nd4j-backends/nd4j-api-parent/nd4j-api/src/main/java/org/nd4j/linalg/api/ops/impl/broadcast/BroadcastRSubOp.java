package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.List;

public class BroadcastRSubOp extends BaseBroadcastOp {
    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastRSubOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, boolean inPlace) {
        super(sameDiff, i_v, dimension, inPlace);
    }

    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, dimension, extraArgs);
    }

    public BroadcastRSubOp(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, dimension, extraArgs);
    }

    public BroadcastRSubOp() {}

    public BroadcastRSubOp(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }


    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String name() {
        return "broadcastrsub";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.rsub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.rsub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.rsub(other);
    }

    @Override
    public float op(float origin, float other) {
        return other - origin;
    }

    @Override
    public double op(double origin, double other) {
        return other - origin;
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

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
