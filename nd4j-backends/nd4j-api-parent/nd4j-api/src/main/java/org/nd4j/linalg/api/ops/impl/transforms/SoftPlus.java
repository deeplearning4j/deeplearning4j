package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class SoftPlus extends BaseTransformOp {
    public SoftPlus(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public SoftPlus(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public SoftPlus(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public SoftPlus(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftPlus() {}

    public SoftPlus(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SoftPlus(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public SoftPlus(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String opName() {
        return "softplus";
    }

    @Override
    public String onnxName() {
        return "Softplus";
    }

    @Override
    public String tensorflowName() {
        return "softplus";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().sigmoid(arg());

        return Collections.singletonList(ret);
    }

}
