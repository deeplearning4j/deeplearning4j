package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.Arrays;
import java.util.List;

public class BroadcastEqualTo extends BaseBroadcastOp {

    public BroadcastEqualTo() {}

    public BroadcastEqualTo(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastEqualTo(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v, int[] dimension, boolean inPlace) {
        super(sameDiff, i_v, dimension, inPlace);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, dimension, extraArgs);
    }

    public BroadcastEqualTo(SameDiff sameDiff, SDVariable i_v, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v, dimension, extraArgs);
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "broadcast_equalto";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(outputVariables()[0]);
    }

    @Override
    public String onnxName() {
        return "Equal";
    }

    @Override
    public String tensorflowName() {
        return "equal";
    }
}
