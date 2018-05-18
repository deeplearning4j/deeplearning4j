package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * DropOut implementation as Op
 *
 * PLEASE NOTE: This is legacy DropOut implementation, please consider using op with the same opName from randomOps
 * @author raver119@gmail.com
 */
public class LegacyDropOut extends BaseTransformOp {

    private double p;

    public LegacyDropOut(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double p) {
        super(sameDiff, i_v, inPlace);
        this.p = p;
    }

    public LegacyDropOut(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs, double p) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.p = p;
    }

    public LegacyDropOut(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double p) {
        super(sameDiff, i_v, extraArgs);
        this.p = p;
    }

    public LegacyDropOut() {

    }

    public LegacyDropOut(INDArray x, double p) {
        super(x);
        this.p = p;
        init(x, null, x, x.length());
    }

    public LegacyDropOut(INDArray x, INDArray z, double p) {
        super(x, z);
        this.p = p;
        init(x, null, z, x.length());
    }

    public LegacyDropOut(INDArray x, INDArray z, double p, long n) {
        super(x, z, n);
        this.p = p;
        init(x, null, z, n);
    }

    @Override
    public int opNum() {
        return 43;
    }

    @Override
    public String opName() {
        return "legacy_dropout";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p, (double) n};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
