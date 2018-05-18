package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.List;

/**
 * Broadcast Max comparison op
 *
 * @author raver119@gmail.com
 */
public class BroadcastMax extends BaseBroadcastOp {

    public BroadcastMax() {}

    public BroadcastMax(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastMax(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v, int[] dimension, boolean inPlace) {
        super(sameDiff, i_v, dimension, inPlace);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, dimension, extraArgs);
    }

    public BroadcastMax(SameDiff sameDiff, SDVariable i_v, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, dimension, extraArgs);
    }

    @Override
    public int opNum() {
        return 14;
    }

    @Override
    public String opName() {
        return "broadcast_max";
    }

    @Override
    public String onnxName() {
        return "Max";
    }

    @Override
    public String tensorflowName() {
        return "max";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
