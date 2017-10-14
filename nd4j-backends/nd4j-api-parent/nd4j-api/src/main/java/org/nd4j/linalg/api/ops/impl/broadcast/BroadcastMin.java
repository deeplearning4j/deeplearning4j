package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.List;

/**
 * Broadcast Min comparison op
 *
 * @author raver119@gmail.com
 */
public class BroadcastMin extends BaseBroadcastOp {

    public BroadcastMin() {}

    public BroadcastMin(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastMin(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, boolean inPlace) {
        super(sameDiff, i_v, dimension, inPlace);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, dimension, extraArgs);
    }

    public BroadcastMin(SameDiff sameDiff, DifferentialFunction i_v, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, dimension, extraArgs);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String name() {
        return "broadcast_min";
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
        return origin > other ? 1.0f : 0.0f;
    }

    @Override
    public double op(double origin, double other) {
        return origin > other ? 1.0f : 0.0f;
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


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
