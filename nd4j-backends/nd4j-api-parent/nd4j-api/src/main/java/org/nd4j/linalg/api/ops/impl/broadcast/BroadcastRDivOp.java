package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.List;

/**
 * Broadcast reverse divide
 * @author Adam Gibson
 */
public class BroadcastRDivOp extends BaseBroadcastOp {

    public BroadcastRDivOp() {}

    public BroadcastRDivOp(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastRDivOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, boolean inPlace) {
        super(sameDiff, i_v, dimension, inPlace);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, dimension, extraArgs);
    }

    public BroadcastRDivOp(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, dimension, extraArgs);
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String name() {
        return "broadcastrdiv";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.rdiv(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.rdiv(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.rdiv(other);
    }

    @Override
    public float op(float origin, float other) {
        return other / origin;
    }

    @Override
    public double op(double origin, double other) {
        return other / origin;
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
