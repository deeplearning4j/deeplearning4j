package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

public class BiasAddGrad extends DynamicCustomOp {

    public BiasAddGrad(SameDiff sameDiff, SDVariable input, SDVariable bias, SDVariable gradient) {
        super(null, sameDiff, new SDVariable[]{input, bias, gradient});
    }

    public BiasAddGrad() {}

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "biasadd_bp";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported for op " + getClass().getSimpleName());
    }

    @Override
    public String onnxName() {
        return "BiasAddGrad";
    }

    @Override
    public String tensorflowName() {
        return "BiasAddGrad";
    }
}
