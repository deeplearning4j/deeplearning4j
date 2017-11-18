package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseGradientOp;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 *
 */
public class SigmoidDerivative extends BaseGradientOp  {
    public SigmoidDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public SigmoidDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public SigmoidDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SigmoidDerivative() {
    }

    public SigmoidDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SigmoidDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, z.lengthLong());
    }

    public SigmoidDerivative(INDArray x) {
        super(x);
    }

    /**
     * An op number
     *
     * @return
     */
    @Override
    public int opNum() {
        return 0;
    }

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    @Override
    public String opName() {
        return "sigmoidderivative";
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
    public void exec() {
        INDArray sigmoid = Nd4j.getExecutioner().execAndReturn(new Sigmoid(x));
        INDArray rsubbed = sigmoid.rsub(1);
        INDArray mulled = y.mul(sigmoid).muli(rsubbed);
        if(mulled != z)
            z.assign(mulled);

    }

    @Override
    public void exec(int... dimensions) {
        super.exec(dimensions);
    }




    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }


}
