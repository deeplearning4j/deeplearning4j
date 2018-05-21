package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Set
 *
 * @author Adam Gibson
 */
public class Set extends BaseTransformOp {
    public Set(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Set(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Set(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Set(INDArray x, INDArray z) {
        super(x, z);
    }

    public Set() {
    }

    public Set(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Set(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Set(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 16;
    }

    @Override
    public String opName() {
        return "set";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        val shape = outputVariables()[0].getShape();
        SDVariable ym1 = f().rsub(rarg(), f().one(shape));
        SDVariable ret = f().mul(f().mul(rarg(), f().pow(larg(), 2.0)), larg());
        return Arrays.asList(ret);
    }

}
