package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseGradientOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class LogSoftMaxDerivative extends BaseGradientOp  {
    public LogSoftMaxDerivative(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public LogSoftMaxDerivative(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public LogSoftMaxDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public LogSoftMaxDerivative() {
    }

    public LogSoftMaxDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public LogSoftMaxDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, z.lengthLong());
    }

    public LogSoftMaxDerivative(INDArray x) {
        super(x);
    }

    public LogSoftMaxDerivative(INDArray indArray, INDArray indArray1, INDArray indArray2, int length) {
        super(indArray,indArray1,indArray2,length);
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
        return "logsoftmaxderivative";
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
        //TODO add dimension arg. For now: hardcoded along dimension 1...

        //Out = log(softmax(x)) = l(s(x))
        //dL/dIn = dL/dOut * dOut/dIn = dL/dOut * dl(s(x))/ds(x) * ds(x)/dx
        //       = (softmax deriv) * 1/s(x)

        INDArray softmax = Transforms.softmax(x, true);
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(x,y,z));
        z.divi(softmax);
    }

    @Override
    public void exec(int... dimensions) {
        super.exec(dimensions);
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(f().sub(i_v.get(0),f().sum(f().exp(larg()),1)));
    }

}
